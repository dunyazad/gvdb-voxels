#pragma once

#include <cuda_common.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <Serialization.hpp>

#include <HashMap.hpp>
#include <MortonCode.cuh>

#include <map>
#include <queue>
#include <vector>

using OctreeKey = uint64_t;

#ifndef OCTREE_NODE
#define OCTREE_NODE
struct OctreeNode
{
    OctreeKey key = UINT64_MAX;
    unsigned int parent = UINT32_MAX;
    unsigned int children[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
    float3 bmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 bmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
};
#endif

struct Octree
{
    __host__ __device__ inline static uint64_t GetDepth(uint64_t key)
    {
        return (key >> 58) & 0x3Full;
    }

    __host__ __device__ inline static uint64_t SetDepth(uint64_t key, uint64_t depth)
    {
        return (key & ~(0x3Full << 58)) | ((depth & 0x3Full) << 58);
    }

    __host__ __device__ inline static uint64_t GetCode(uint64_t key, uint64_t level)
    {
        auto shift = level * 3ull;
        return (shift >= 58ull) ? 0ull : ((key >> shift) & 0x7ull);
    }

    __host__ __device__ inline static uint64_t SetCode(uint64_t key, uint64_t level, uint64_t x, uint64_t y, uint64_t z)
    {
        auto code3 = (x & 1ull) << 2 | (y & 1ull) << 1 | (z & 1ull) << 0;

        auto shift = level * 3ull;
        if (shift >= 58ull) return key;

        auto mask = 0x7ull << shift;
        auto cleared = key & ~mask;
        return cleared | ((code3 & 0x7ull) << shift);
    }

    __host__ __device__ inline static uint64_t GetParentKey(uint64_t key)
    {
        auto depth = GetDepth(key);
        if (depth == 0) return key;

        auto newDepth = depth - 1;
        auto shift = (newDepth * 3ull);

        auto cleared = key & ~(0x7ull << shift);
        return SetDepth(cleared, newDepth);
    }

    __host__ __device__ inline static uint64_t GetChildKey(uint64_t key, uint64_t bx, uint64_t by, uint64_t bz)
    {
        auto depth = GetDepth(key);

        auto child = ((bx & 1ull) << 2) | ((by & 1ull) << 1) | (bz & 1ull);
        auto shift = depth * 3ull;

        auto extended = key | (child << shift);
        return SetDepth(extended, depth + 1);
    }

    __host__ __device__ inline static uint64_t ToKey(float3 position, float3 aabbMin, float3 aabbMax, uint64_t depth)
    {
        uint64_t payload = 0ull;
        float3 bbMin = aabbMin;
        float3 bbMax = aabbMax;

        for (uint64_t i = 0; i < depth; ++i)
        {
            float3 center
            {
                0.5f * (bbMin.x + bbMax.x),
                0.5f * (bbMin.y + bbMax.y),
                0.5f * (bbMin.z + bbMax.z)
            };

            auto bx = (position.x >= center.x) ? 1ull : 0ull;
            auto by = (position.y >= center.y) ? 1ull : 0ull;
            auto bz = (position.z >= center.z) ? 1ull : 0ull;

            auto child = (bx << 2) | (by << 1) | bz;

            payload |= (child << (i * 3ull));

            bbMin.x = bx ? center.x : bbMin.x;
            bbMax.x = bx ? bbMax.x : center.x;

            bbMin.y = by ? center.y : bbMin.y;
            bbMax.y = by ? bbMax.y : center.y;

            bbMin.z = bz ? center.z : bbMin.z;
            bbMax.z = bz ? bbMax.z : center.z;
        }

        return SetDepth(payload, depth);
    }

    __host__ __device__ inline static float3 ToPosition(uint64_t key, float3 aabbMin, float3 aabbMax)
    {
        unsigned depth = GetDepth(key);

        float3 bbMin = aabbMin;
        float3 bbMax = aabbMax;

        for (unsigned i = 0; i < depth; ++i)
        {
            float3 center
            {
                0.5f * (bbMin.x + bbMax.x),
                0.5f * (bbMin.y + bbMax.y),
                0.5f * (bbMin.z + bbMax.z)
            };

            unsigned code = GetCode(key, i);
            unsigned bx = (code >> 2) & 1ull;
            unsigned by = (code >> 1) & 1ull;
            unsigned bz = (code >> 0) & 1ull;

            bbMin.x = bx ? center.x : bbMin.x;
            bbMax.x = bx ? bbMax.x : center.x;

            bbMin.y = by ? center.y : bbMin.y;
            bbMax.y = by ? bbMax.y : center.y;

            bbMin.z = bz ? center.z : bbMin.z;
            bbMax.z = bz ? bbMax.z : center.z;
        }

        return float3
        {
            0.5f * (bbMin.x + bbMax.x),
            0.5f * (bbMin.y + bbMax.y),
            0.5f * (bbMin.z + bbMax.z)
        };
    }

    __host__ __device__ inline static float GetScale(OctreeKey key, uint64_t maxDepth, float unitLength)
    {
        auto level = Octree::GetDepth(key);
        const unsigned k = (maxDepth >= level) ? (maxDepth - level) : 0u;
        
#if defined(__CUDA_ARCH__)
        // device path
        return ldexpf(unitLength, static_cast<int>(k));
#else
        // host path
        return std::ldexp(unitLength, static_cast<int>(k));
#endif
    }

    __host__ __device__ inline static void SplitChildAABB(
        const float3& pmin,
        const float3& pmax,
        unsigned childCode3, // 0..7, (bx<<2)|(by<<1)|bz
        float3* cmin,
        float3* cmax)
    {
        float3 center
        {
            0.5f * (pmin.x + pmax.x),
            0.5f * (pmin.y + pmax.y),
            0.5f * (pmin.z + pmax.z)
        };

        const unsigned bx = (childCode3 >> 2) & 1u;
        const unsigned by = (childCode3 >> 1) & 1u;
        const unsigned bz = (childCode3 >> 0) & 1u;

        cmin->x = bx ? center.x : pmin.x;  cmax->x = bx ? pmax.x : center.x;
        cmin->y = by ? center.y : pmin.y;  cmax->y = by ? pmax.y : center.y;
        cmin->z = bz ? center.z : pmin.z;  cmax->z = bz ? pmax.z : center.z;
    }

    __host__ __device__ inline static float Dist2PointAABB(const float3& q, const float3& bmin, const float3& bmax)
    {
        const float dx = fmaxf(fmaxf(bmin.x - q.x, 0.0f), q.x - bmax.x);
        const float dy = fmaxf(fmaxf(bmin.y - q.y, 0.0f), q.y - bmax.y);
        const float dz = fmaxf(fmaxf(bmin.z - q.z, 0.0f), q.z - bmax.z);
        return dx * dx + dy * dy + dz * dz;
    }

    __host__ __device__ inline static float3 ClosestPointOnAABB(const float3& q, const float3& bmin, const float3& bmax)
    {
        float3 c;
        c.x = fminf(fmaxf(q.x, bmin.x), bmax.x);
        c.y = fminf(fmaxf(q.y, bmin.y), bmax.y);
        c.z = fminf(fmaxf(q.z, bmin.z), bmax.z);
        return c;
    }
};

struct HostOctree
{
    std::vector<OctreeNode> nodes;
    unsigned int rootIndex = UINT32_MAX;
    uint64_t leafDepth = UINT64_MAX;
    float3 aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float unitLength = 0.0f;

    void Initialize(
        float3* positions,
        unsigned int numberOfPositions,
        float3 aabbMin,
        float3 aabbMax,
        uint64_t leafDepth,
        float unitLength);

    void Terminate();

    OctreeNode* HostOctree::NN(float3 query)
    {
        if (rootIndex == UINT32_MAX || nodes.empty())
        {
            return nullptr;
        }

        struct Item
        {
            float d2;
            unsigned idx;
        };

        struct Cmp
        {
            bool operator()(const Item& a, const Item& b) const
            {
                return a.d2 > b.d2; // min-heap
            }
        };

        auto nodeDist2 = [&](unsigned idx) -> float
        {
            const auto& n = nodes[idx];
            return Octree::Dist2PointAABB(query, n.bmin, n.bmax);
        };

        std::priority_queue<Item, std::vector<Item>, Cmp> pq;
        pq.push({ nodeDist2(rootIndex), rootIndex });

        float bestD2 = FLT_MAX;
        OctreeNode* bestNode = nullptr;

        while (!pq.empty())
        {
            Item it = pq.top();
            pq.pop();

            // 하한이 현재 best 이상이면 이 서브트리 전체 prune
            if (it.d2 >= bestD2)
            {
                continue;
            }

            const OctreeNode& n = nodes[it.idx];

            bool isLeaf = true;
            for (int c = 0; c < 8; ++c)
            {
                unsigned ci = n.children[c];
                if (ci == UINT32_MAX) continue;
                isLeaf = false;

                float cd2 = nodeDist2(ci);
                if (cd2 < bestD2)
                {
                    pq.push({ cd2, ci });
                }
            }

            if (isLeaf)
            {
                // leaf 자체의 AABB 하한이 it.d2 이므로, 여기서 best 갱신
                if (it.d2 < bestD2)
                {
                    bestD2 = it.d2;
                    bestNode = const_cast<OctreeNode*>(&nodes[it.idx]);
                }
            }
        }

        return bestNode;
    }

};

struct DeviceOctree
{
    void Initialize(
        float3* positions,
        unsigned int numberOfPoints,
        float3 aabbMin,
        float3 aabbMax,
        uint64_t leafDepth,
        float unitLength);

    void Terminate();

    OctreeNode* nodes = nullptr;
    unsigned int* rootIndex = nullptr;
    uint64_t leafDepth = UINT64_MAX;
    float3 aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float unitLength = 0.0f;
    unsigned int allocatedNodes = 0;
    unsigned int numberOfNodes = 0;
    unsigned int* d_numberOfNodes = nullptr;

    HashMap<uint64_t, unsigned int> octreeKeys;

    __device__ static inline float nodeDist2(
        OctreeNode* nodes,
        unsigned int idx,
        const float3& q)
    {
        const OctreeNode& n = nodes[idx];
        return Octree::Dist2PointAABB(q, n.bmin, n.bmax);
    }

    __device__ static inline float nodeDist2_direct(
        const OctreeNode* nodes,
        unsigned int idx,
        const float3& q)
    {
        const OctreeNode& n = nodes[idx];
        return Octree::Dist2PointAABB(q, n.bmin, n.bmax);
    }

    __device__ static inline bool isLeaf(const OctreeNode& n)
    {
        return (n.children[0] == UINT32_MAX) &&
            (n.children[1] == UINT32_MAX) &&
            (n.children[2] == UINT32_MAX) &&
            (n.children[3] == UINT32_MAX) &&
            (n.children[4] == UINT32_MAX) &&
            (n.children[5] == UINT32_MAX) &&
            (n.children[6] == UINT32_MAX) &&
            (n.children[7] == UINT32_MAX);
    }

    __device__ static inline unsigned int DeviceOctree_NearestLeaf(
        const float3& query,
        OctreeNode* __restrict__ nodes,
        const unsigned int* __restrict__ rootIndex,
        float* outBestD2)
    {
        if (nodes == nullptr || *rootIndex == UINT32_MAX)
        {
            if (outBestD2) *outBestD2 = FLT_MAX;
            return UINT32_MAX;
        }

        const int STACK_CAP = 64;
        unsigned int  stackIdx[STACK_CAP];
        float         stackD2[STACK_CAP];
        int top = 0;

        const unsigned int root = *rootIndex;
        float rootD2 = nodeDist2_direct(nodes, root, query);
        stackIdx[top] = root;
        stackD2[top] = rootD2;
        ++top;

        float bestD2 = FLT_MAX;
        unsigned int bestIdx = UINT32_MAX;

        while (top > 0)
        {
            --top;
            const unsigned int idx = stackIdx[top];
            const float nd2 = stackD2[top];

            if (nd2 >= bestD2) continue;

            const OctreeNode& n = nodes[idx];
            if (isLeaf(n))
            {
                if (nd2 < bestD2) { bestD2 = nd2; bestIdx = idx; }
                continue;
            }

            // 자식 push (하한 프루닝)
#pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                const unsigned int ci = n.children[c];
                if (ci == UINT32_MAX) continue;

                const float cd2 = nodeDist2_direct(nodes, ci, query);
                if (cd2 < bestD2 && top < STACK_CAP)
                {
                    stackIdx[top] = ci;
                    stackD2[top] = cd2;
                    ++top;
                }
            }
        }

        if (outBestD2) *outBestD2 = bestD2;
        return bestIdx;
    }

    void NN(
        float3* queries,
        unsigned int numberOfQueries,
        unsigned int* outNearestIndex,
        float* outNearestD2);    

    bool NN_Single(
        const float3& query,
        unsigned int* h_index,
        float* h_d2);
};

__global__ void Kernel_DeviceOctree_NN_Batch(
    OctreeNode* __restrict__ nodes,
    const unsigned int* __restrict__ rootIndex,
    const float3* __restrict__ queries,
    unsigned int numberOfQueries,
    unsigned int* __restrict__ outIndices,
    float* __restrict__ outD2);