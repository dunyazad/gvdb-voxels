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
#include <vector>

#ifndef HOST_OCTREE_NODE
#define HOST_OCTREE_NODE
struct HostOctreeNode
{
    uint64_t key = UINT64_MAX;
    uint64_t parentKey = UINT64_MAX;
    unsigned int children[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
};
#endif

struct HostOctree
{
    vector<HostOctreeNode> nodes;

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
        auto code3 =  (x & 1ull) << 2 | (y & 1ull) << 1 | (z & 1ull) << 0;

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

        auto child = (bx << 2) | (by << 1) | bz;
        auto shift = depth * 3ull;

        auto extended = key | child << shift;
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

    void Initialize(float3* positions, unsigned int numberOfPositions, float3 aabbMin, float3 aabbMax, uint64_t leafDepth)
    {
        auto dimension = aabbMax - aabbMin;
        auto maxLength = fmaxf(dimension.x, fmaxf(dimension.y, dimension.z));
        auto center = (aabbMin + aabbMax) * 0.5f;

        auto bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
        auto bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

        std::map<uint64_t, uint64_t> codes;

        for (size_t i = 0; i < numberOfPositions; ++i)
        {
            uint64_t leafKey = ToKey(positions[i], bbMin, bbMax, leafDepth);
            codes[leafKey] = leafDepth;

            uint64_t code = leafKey;
            for (int j = leafDepth - 1; j >= 0; --j)
            {
                //auto childIndex = GetCode(leafKey, j);

                code = GetParentKey(code);

                auto [it, inserted] = codes.emplace(code, (uint64_t)j);
                if (!inserted)
                {
                    break;
                }
            }
        }

        nodes.reserve(codes.size());
        for (const auto& kvp : codes)
        {
            nodes.push_back({
                kvp.first,
                GetParentKey(UINT64_MAX),
                { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                  UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX }
                });
        }
    }

    void Terminate()
    {

    }
};


using OctreeKey = uint64_t;

#ifndef DEVICE_OCTREE_NODE
#define DEVICE_OCTREE_NODE
struct DeviceOctreeNode
{
    uint64_t mortonCode = UINT64_MAX;
    unsigned int level = UINT32_MAX;
    unsigned int parent = UINT32_MAX;
    unsigned int children[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
};
#endif

struct DeviceOctree
{
    static const int kDepthBits = 6;
    static const int kKeyBits = 58;
    static const int kBitsPerLevel = 3;
    static const int kMaxDepth = kKeyBits / kBitsPerLevel;     // 19

    static const uint32_t range = 1u << DeviceOctree::kMaxDepth;
    static __host__ __device__ inline int3 GridOffset()
    {
        int h = int(range / 2);
        return make_int3(h, h, h);
    }

    static const uint64_t kKeyMask = (1ull << kKeyBits) - 1;   // 0x03FFFFFFFFFFFFFF
    static const uint64_t kDepthMask = ~kKeyMask;              // 0xFC00000000000000

    DeviceOctreeNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;
    unsigned int numberOfNodes = 0;
    unsigned int* d_numberOfNodes = nullptr;

    HashMap<uint64_t, unsigned int> mortonCodeOctreeNodeMapping;
    HashMap<uint64_t, unsigned int> mortonCodes;

    unsigned int numberOfPoints = 0;
    float3 aabbMin = { FLT_MAX, FLT_MAX, FLT_MAX };
    float3 aabbMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    float voxelSize = 0.0f;

    void Initialize(
        float3* positions,
        unsigned numberOfPoints,
        float3 aabbMin,
        float3 aabbMax,
        float voxelSize);
    void Terminate();

    __host__ __device__
        inline static uint32_t clampu(uint32_t v, uint32_t lo, uint32_t hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    __host__ __device__
        inline static uint64_t Pack(uint8_t depth, uint64_t key)
    {
        return (uint64_t(depth) << kKeyBits) | (key & kKeyMask);
    }

    __host__ __device__
        inline static uint8_t UnpackDepth(uint64_t code)
    {
        return uint8_t(code >> kKeyBits);
    }

    __host__ __device__
        inline static uint64_t UnpackKey(uint64_t code)
    {
        return code & kKeyMask;
    }

    __host__ __device__
        inline static uint64_t EncodeKeyFromXYZ(uint32_t ix, uint32_t iy, uint32_t iz, uint8_t depth)
    {
        if (depth > kMaxDepth) depth = kMaxDepth;
        uint64_t key = 0;
        for (int l = depth - 1; l >= 0; --l)
        {
            uint64_t cx = (ix >> l) & 1u;
            uint64_t cy = (iy >> l) & 1u;
            uint64_t cz = (iz >> l) & 1u;
            uint64_t child = (cx << 2) | (cy << 1) | cz;
            key = (key << 3) | child;
        }
        return key;
    }

    __host__ __device__
        inline static uint64_t KeyFromPoint_AABB(float3 p, float3 bmin, float3 bmax, uint8_t depth)
    {
        if (depth > kMaxDepth) depth = kMaxDepth;
        const uint32_t S = 1u << depth;

        const float rx = bmax.x - bmin.x;
        const float ry = bmax.y - bmin.y;
        const float rz = bmax.z - bmin.z;

        uint32_t ix = rx > 0 ? uint32_t(::floorf((p.x - bmin.x) / rx * S)) : 0u;
        uint32_t iy = ry > 0 ? uint32_t(::floorf((p.y - bmin.y) / ry * S)) : 0u;
        uint32_t iz = rz > 0 ? uint32_t(::floorf((p.z - bmin.z) / rz * S)) : 0u;

        ix = clampu(ix, 0u, S - 1u);
        iy = clampu(iy, 0u, S - 1u);
        iz = clampu(iz, 0u, S - 1u);

        const uint64_t key = EncodeKeyFromXYZ(ix, iy, iz, depth);
        return Pack(depth, key);
    }

    //__host__ __device__
    //    inline static uint64_t KeyFromPoint_Voxel(float3 p,
    //        float3 center,
    //        float  voxelSize,
    //        uint8_t depth,
    //        int3   gridOffset)
    //{
    //    if (depth > kMaxDepth) depth = kMaxDepth;

    //    const float inv = voxelSize > 0 ? (1.0f / voxelSize) : 0.0f;
    //    const int32_t gx = int32_t(::floorf((p.x - center.x) * inv)) + gridOffset.x;
    //    const int32_t gy = int32_t(::floorf((p.y - center.y) * inv)) + gridOffset.y;
    //    const int32_t gz = int32_t(::floorf((p.z - center.z) * inv)) + gridOffset.z;

    //    const uint32_t shift = kMaxDepth - depth;
    //    const int32_t roundFix = (shift == 0u) ? 0 : int32_t((1u << shift) - 1u);
    //    const uint32_t ix = uint32_t((gx >= 0 ? gx : gx - roundFix) >> shift);
    //    const uint32_t iy = uint32_t((gy >= 0 ? gy : gy - roundFix) >> shift);
    //    const uint32_t iz = uint32_t((gz >= 0 ? gz : gz - roundFix) >> shift);

    //    const uint64_t key = EncodeKeyFromXYZ(ix, iy, iz, depth);
    //    return Pack(depth, key);
    //}

    __host__ __device__
        inline static uint64_t KeyFromPoint_Voxel(float3 p,
            float3 center,
            float  voxelSize,
            uint8_t depth,
            int3   gridOffset)
    {
        if (depth > kMaxDepth) depth = kMaxDepth;

        const float inv = voxelSize > 0 ? (1.0f / voxelSize) : 0.0f;
        const int32_t gx = int32_t(::floorf((p.x - center.x) * inv)) + gridOffset.x;
        const int32_t gy = int32_t(::floorf((p.y - center.y) * inv)) + gridOffset.y;
        const int32_t gz = int32_t(::floorf((p.z - center.z) * inv)) + gridOffset.z;

        const uint32_t shift = kMaxDepth - depth;
        const int32_t roundFix = (shift == 0u) ? 0 : int32_t((1u << shift) - 1u);

        // floor-division by 2^shift (음수 안전)
        int32_t ix_s = (gx >= 0 ? gx : gx - roundFix) >> shift;
        int32_t iy_s = (gy >= 0 ? gy : gy - roundFix) >> shift;
        int32_t iz_s = (gz >= 0 ? gz : gz - roundFix) >> shift;

        // [0, 2^depth - 1] 클램프 (언더/오버플로우 방지)
        const int32_t S_max = int32_t((1u << depth) - 1u);
        ix_s = ix_s < 0 ? 0 : (ix_s > S_max ? S_max : ix_s);
        iy_s = iy_s < 0 ? 0 : (iy_s > S_max ? S_max : iy_s);
        iz_s = iz_s < 0 ? 0 : (iz_s > S_max ? S_max : iz_s);

        const uint64_t key = EncodeKeyFromXYZ(uint32_t(ix_s), uint32_t(iy_s), uint32_t(iz_s), depth);
        return Pack(depth, key);
    }

    __host__ __device__
        inline static void DecodeXYZFromKey(uint64_t key, uint8_t depth, uint32_t& ix, uint32_t& iy, uint32_t& iz)
    {
        if (depth > kMaxDepth) depth = kMaxDepth;
        ix = iy = iz = 0u;

        // Encode 때: depth-1 .. 0 순서로 상위비트부터 child를 밀어넣었으므로
        // Decode는 l*3 위치의 상위 그룹부터 차례로 읽어 복원
        for (int l = depth - 1; l >= 0; --l)
        {
            const uint64_t child = (key >> (l * 3)) & 0x7ull; // [x y z] 비트
            ix = (ix << 1) | uint32_t((child >> 2) & 1u);
            iy = (iy << 1) | uint32_t((child >> 1) & 1u);
            iz = (iz << 1) | uint32_t((child >> 0) & 1u);
        }
    }

    __host__ __device__
        inline static int3 MaxDepthIndexFromDepthIndex(uint32_t ix, uint32_t iy, uint32_t iz, uint8_t depth, bool cellCenter)
    {
        if (depth > kMaxDepth) depth = kMaxDepth;

        const uint32_t scale = 1u << (kMaxDepth - depth); // 한 셀의 크기(최대 해상도 보xel 개수)
        int3 g;
        g.x = int32_t(ix) * int32_t(scale);
        g.y = int32_t(iy) * int32_t(scale);
        g.z = int32_t(iz) * int32_t(scale);

        if (cellCenter)
        {
            const int32_t half = int32_t(scale >> 1); // scale==1이면 0
            g.x += half;
            g.y += half;
            g.z += half;
        }
        return g;
    }

    //__host__ __device__
    //    inline static float3 PointFromCode_Voxel(uint64_t code,
    //        float3 center,
    //        float  voxelSize,
    //        int3   gridOffset,
    //        bool   cellCenter = true)
    //{
    //    uint8_t depth = UnpackDepth(code);
    //    uint64_t key = UnpackKey(code);

    //    uint32_t ix, iy, iz;
    //    DecodeXYZFromKey(key, depth, ix, iy, iz);

    //    const int3 g = MaxDepthIndexFromDepthIndex(ix, iy, iz, depth, cellCenter);

    //    float3 p;
    //    // forward 매핑: gx = floor((x - center.x)/voxelSize) + gridOffset.x
    //    // 역변환: x = center.x + (gx - gridOffset.x) * voxelSize
    //    p.x = center.x + (float(g.x - gridOffset.x) * voxelSize);
    //    p.y = center.y + (float(g.y - gridOffset.y) * voxelSize);
    //    p.z = center.z + (float(g.z - gridOffset.z) * voxelSize);
    //    return p;
    //}

    //__host__ __device__
    //    inline static float3 PointFromCode_Voxel(uint64_t code,
    //        float3 center,
    //        float  voxelSize,
    //        int3   gridOffset,
    //        bool   cellCenter = true)
    //{
    //    const uint8_t depth = UnpackDepth(code);
    //    const uint64_t key = UnpackKey(code);

    //    uint32_t ix, iy, iz;
    //    DecodeXYZFromKey(key, depth, ix, iy, iz);

    //    const float scale = float(1u << (kMaxDepth - depth)); // 이 leaf가 커버하는 max-res 보xel 수
    //    const float cx = float(ix) * scale + (cellCenter ? 0.5f * scale : 0.0f);
    //    const float cy = float(iy) * scale + (cellCenter ? 0.5f * scale : 0.0f);
    //    const float cz = float(iz) * scale + (cellCenter ? 0.5f * scale : 0.0f);

    //    float3 p;
    //    p.x = center.x + (cx - float(gridOffset.x)) * voxelSize;
    //    p.y = center.y + (cy - float(gridOffset.y)) * voxelSize;
    //    p.z = center.z + (cz - float(gridOffset.z)) * voxelSize;
    //    return p;
    //}

    __host__ __device__
        inline static float3 PointFromCode_Voxel(uint64_t code,
            float3 center,
            float  voxelSize,
            int3   gridOffset,
            bool   cellCenter = true)
    {
        const uint8_t depth = UnpackDepth(code);
        const uint64_t key = UnpackKey(code);

        uint32_t ix, iy, iz;
        DecodeXYZFromKey(key, depth, ix, iy, iz);

        const float scale = float(1u << (kMaxDepth - depth)); // 이 노드가 커버하는 max-res voxel 수
        const float cx = float(ix) * scale + (cellCenter ? 0.5f * scale : 0.0f);
        const float cy = float(iy) * scale + (cellCenter ? 0.5f * scale : 0.0f);
        const float cz = float(iz) * scale + (cellCenter ? 0.5f * scale : 0.0f);

        float3 p;
        p.x = center.x + (cx - float(gridOffset.x)) * voxelSize;
        p.y = center.y + (cy - float(gridOffset.y)) * voxelSize;
        p.z = center.z + (cz - float(gridOffset.z)) * voxelSize;
        return p;
    }

    __host__ __device__
        inline static uint64_t ParentCode(uint64_t code)
    {
        uint8_t d = UnpackDepth(code);
        if (d == 0) return Pack(0, 0ull);
        uint64_t k = UnpackKey(code) >> 3; // 말단 3비트 제거
        return Pack(uint8_t(d - 1), k);
    }

    __host__ __device__
        inline static uint32_t ChildIndex(uint64_t code)
    {
        // 말단 3비트가 이 노드의 자식 인덱스(0..7)
        return uint32_t(UnpackKey(code) & 0x7ull);
    }
};
