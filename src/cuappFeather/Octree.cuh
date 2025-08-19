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

using OctreeKey = uint64_t;

#ifndef OCTREE_NODE
#define OCTREE_NODE
struct OctreeNode
{
    OctreeKey key = UINT64_MAX;
    unsigned int parent = UINT32_MAX;
    unsigned int children[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
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
};

struct HostOctree
{
    vector<OctreeNode> nodes;

    void Initialize(float3* positions, unsigned int numberOfPositions, float3 aabbMin, float3 aabbMax, uint64_t leafDepth);

    void Terminate();
};

struct DeviceOctree
{
    void Initialize(
        float3* positions,
        unsigned int numberOfPoints,
        float3 aabbMin,
        float3 aabbMax,
        uint64_t leafDepth);

    void Terminate();

    OctreeNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;
    unsigned int numberOfNodes = 0;
    unsigned int* d_numberOfNodes = nullptr;

    HashMap<uint64_t, unsigned int> octreeKeys;
};
