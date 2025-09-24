#pragma once

#include <cuda_common.cuh>
#include <SimpleHashMap.hpp>
#include <MortonCode.cuh>

#include <stack>

struct MortonKey
{
    uint64_t code;
    unsigned int index;

    __host__ __device__ bool operator<(const MortonKey& other) const
    {
        if (code < other.code) return true;
        if (code > other.code) return false;
        return index < other.index;
    }

    __host__ __device__
        bool operator==(const MortonKey& other) const {
        return (code == other.code && index == other.index);
    }
};

struct SimpleBVHNode
{
    unsigned int parentNodeIndex = UINT32_MAX;
    unsigned int leftNodeIndex = UINT32_MAX;
    unsigned int rightNodeIndex = UINT32_MAX;
	unsigned pending = 2;

    cuAABB aabb = { make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
                    make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX) };
};

template<typename T>
struct SimpleBVH
{

};