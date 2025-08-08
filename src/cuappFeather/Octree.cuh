#pragma once

#include <cuda_common.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <Serialization.hpp>

#include <HashMap.hpp>

using OctreeKey = uint64_t;

#ifndef OCTREE_NODE
#define OCTREE_NODE
struct OctreeNode
{
    uint64_t mortonCode = UINT64_MAX;
    unsigned int level = UINT32_MAX;
    unsigned int parent = UINT32_MAX;
    unsigned int children[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
};
#endif

struct Octree
{
    static const int kDepthBits = 6;
    static const int kKeyBits = 58;
    static const int kBitsPerLevel = 3;
    static const int kMaxDepth = kKeyBits / kBitsPerLevel;     // 19

    static const uint64_t kKeyMask = (1ull << kKeyBits) - 1;   // 0x03FFFFFFFFFFFFFF
    static const uint64_t kDepthMask = ~kKeyMask;              // 0xFC00000000000000

    OctreeNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;
    unsigned int numberOfNodes = 0;
    unsigned int* d_numberOfNodes = nullptr;

    HashMap<uint64_t, unsigned int> mortonCodeOctreeNodeMapping;

    void Initialize(float3* positions, unsigned numberOfPoints, float3 center, float voxelSize);
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

        const float scale = float(1u << (kMaxDepth - depth)); // 이 노드가 커버하는 max-res 보xel 수
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
