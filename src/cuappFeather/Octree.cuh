#pragma once

#include <cuda_common.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <Serialization.hpp>

#include <SimpleHashMap.hpp>
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
    unsigned int positionIndex = UINT32_MAX;
};
#endif

struct Octree
{
    __host__ __device__ static uint64_t GetDepth(uint64_t key);

    __host__ __device__ static uint64_t SetDepth(uint64_t key, uint64_t depth);

    __host__ __device__ static uint64_t GetCode(uint64_t key, uint64_t level);

    __host__ __device__ static uint64_t SetCode(uint64_t key, uint64_t level, uint64_t x, uint64_t y, uint64_t z);

    __host__ __device__ static uint64_t GetParentKey(uint64_t key);

    __host__ __device__ static uint64_t GetChildKey(uint64_t key, uint64_t bx, uint64_t by, uint64_t bz);

    __host__ __device__ static uint64_t ToKey(float3 position, float3 aabbMin, float3 aabbMax, uint64_t depth);

    __host__ __device__ static float3 ToPosition(uint64_t key, float3 aabbMin, float3 aabbMax);

    __host__ __device__ static float GetScale(OctreeKey key, uint64_t maxDepth, float unitLength);

    __host__ __device__ static void SplitChildAABB(
        const float3& pmin,
        const float3& pmax,
        unsigned childCode3, // 0..7, (bx<<2)|(by<<1)|bz
        float3* cmin,
        float3* cmax);

    __host__ __device__ static float Dist2PointAABB(const float3& q, const float3& bmin, const float3& bmax);

    __host__ __device__ static float3 ClosestPointOnAABB(const float3& q, const float3& bmin, const float3& bmax);
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

    OctreeNode* NN(float3 query);

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

    SimpleHashMap<uint64_t, unsigned int> octreeKeys;

    float3* positions = nullptr;
	unsigned int numberOfPositions = 0;

    void NN_H(
        float3* queries,
        unsigned int numberOfQueries,
        unsigned int* outNearestIndex,
        float* outNearestD2);

    void NN_D(
        float3* d_queries,
        unsigned int numberOfQueries,
        unsigned int* d_outNearestIndex,
        float* d_outNearestD2 = nullptr);

    bool NN_Single(
        const float3& query,
        unsigned int* h_index,
        float* h_d2);

    __device__ static float nodeDist2_direct(
        const OctreeNode* nodes,
        unsigned int idx,
        const float3& q);

    __device__ static bool isLeaf(const OctreeNode& n);

    __device__ static void preferred_child_order(
        const OctreeNode& n, const float3& q, int order[8]);

    __device__ static unsigned int DeviceOctree_NearestLeaf(
        const float3& query,
        const OctreeNode* __restrict__ nodes,
        const unsigned int* __restrict__ rootIndex,
        float* outBestD2 = nullptr);
};

__global__ void Kernel_DeviceOctree_NN_Batch(
    const OctreeNode* __restrict__ nodes,
    const unsigned int* __restrict__ rootIndex,
    const float3* __restrict__ positions,
    const float3* __restrict__ queries,
    unsigned int numberOfQueries,
    unsigned int* __restrict__ outIndices,
    float* __restrict__ outD2 = nullptr);