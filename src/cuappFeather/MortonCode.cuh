#pragma once

#include <cuda_common.cuh>

struct MortonCode
{
    static constexpr uint32_t BITS_MAX = 21u;
    static constexpr uint64_t MASK_21 = (1ull << BITS_MAX) - 1ull;

    __host__ __device__ __forceinline__ static uint64_t expandBy3(uint32_t v)
    {
        uint64_t x = static_cast<uint64_t>(v) & 0x1fffffull;            // 21 bits
        x = (x | (x << 32)) & 0x1f00000000ffffull;
        x = (x | (x << 16)) & 0x1f0000ff0000ffull;
        x = (x | (x << 8)) & 0x100f00f00f00f00full;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
        x = (x | (x << 2)) & 0x1249249249249249ull;
        return x;
    }

    __host__ __device__ __forceinline__ static uint32_t compactBy3(uint64_t v)
    {
        v &= 0x1249249249249249ull;
        v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ull;
        v = (v ^ (v >> 4)) & 0x100f00f00f00f00full;
        v = (v ^ (v >> 8)) & 0x1f0000ff0000ffull;
        v = (v ^ (v >> 16)) & 0x1f00000000ffffull;
        v = (v ^ (v >> 32)) & 0x1fffffull;
        return static_cast<uint32_t>(v);
    }

    __host__ __device__ __forceinline__ static uint64_t encode21(uint32_t x, uint32_t y, uint32_t z)
    {
        x &= MASK_21; y &= MASK_21; z &= MASK_21;
        return expandBy3(x) | (expandBy3(y) << 1) | (expandBy3(z) << 2);
    }

    __host__ __device__ __forceinline__ static void decode21(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z)
    {
        x = compactBy3(morton);
        y = compactBy3(morton >> 1);
        z = compactBy3(morton >> 2);
    }

    __host__ __device__ __forceinline__ static int floor_to_int(float x)
    {
#if defined(__CUDA_ARCH__)
        return __float2int_rd(x); // device: round down
#else
        return static_cast<int>(floorf(x)); // host
#endif
    }

    __host__ __device__ __forceinline__ static float clampf(float v, float lo, float hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    __host__ __device__ __forceinline__ static int3 ComputeDims(const float3& aabbMin, const float3& aabbMax, float voxelSize)
    {
        const float ex = fmaxf(0.0f, aabbMax.x - aabbMin.x);
        const float ey = fmaxf(0.0f, aabbMax.y - aabbMin.y);
        const float ez = fmaxf(0.0f, aabbMax.z - aabbMin.z);

        const int dx = static_cast<int>(ceilf(ex / voxelSize));
        const int dy = static_cast<int>(ceilf(ey / voxelSize));
        const int dz = static_cast<int>(ceilf(ez / voxelSize));
        return make_int3(dx, dy, dz);
    }

    __host__ __device__ __forceinline__ static uint64_t MortonEncodeLimited(uint32_t x, uint32_t y, uint32_t z, unsigned int depth)
    {
        const uint32_t m = (depth >= 32u) ? 0xffffffffu : ((1u << depth) - 1u);
        return MortonCode::encode21(x & m, y & m, z & m);
    }

    __host__ __device__ __forceinline__ static bool PositionToMorton(
        const float3& p,
        const float3& aabbMin,
        const float3& aabbMax,
        float voxelSize,
        unsigned int maxDepth,
        uint64_t& outKey)
    {
        if (maxDepth == 0u || maxDepth > MortonCode::BITS_MAX)
        {
            return false;
        }

        if (p.x < aabbMin.x || p.y < aabbMin.y || p.z < aabbMin.z) return false;
        if (p.x > aabbMax.x || p.y > aabbMax.y || p.z > aabbMax.z) return false;

        const int3 dims = ComputeDims(aabbMin, aabbMax, voxelSize);
        const uint32_t cap = (1u << maxDepth);

        if (dims.x > static_cast<int>(cap) || dims.y > static_cast<int>(cap) || dims.z > static_cast<int>(cap))
        {
            return false;
        }

        const float eps = 1e-7f;
        const float rx = clampf((p.x - aabbMin.x) / voxelSize, 0.0f, fmaxf(0.0f, (float)dims.x - eps));
        const float ry = clampf((p.y - aabbMin.y) / voxelSize, 0.0f, fmaxf(0.0f, (float)dims.y - eps));
        const float rz = clampf((p.z - aabbMin.z) / voxelSize, 0.0f, fmaxf(0.0f, (float)dims.z - eps));

        const int ix = floor_to_int(rx);
        const int iy = floor_to_int(ry);
        const int iz = floor_to_int(rz);

        if (ix < 0 || iy < 0 || iz < 0) return false;
        if (ix >= dims.x || iy >= dims.y || iz >= dims.z) return false;

        if ((uint32_t)ix >= cap || (uint32_t)iy >= cap || (uint32_t)iz >= cap) return false;

        outKey = MortonEncodeLimited((uint32_t)ix, (uint32_t)iy, (uint32_t)iz, maxDepth);
        return true;
    }

    __host__ __device__ __forceinline__ static int3 MortonToIndex(uint64_t key, unsigned int maxDepth)
    {
        uint32_t x, y, z;
        MortonCode::decode21(key, x, y, z);
        if (maxDepth == 0u || maxDepth > MortonCode::BITS_MAX)
        {
            return make_int3((int)x, (int)y, (int)z);
        }
        const uint32_t m = (1u << maxDepth) - 1u;
        return make_int3((int)(x & m), (int)(y & m), (int)(z & m));
    }

    __host__ __device__ __forceinline__ static float3 MortonToPosition(
        uint64_t mortonCode,
        const float3& aabbMin,
        float voxelSize,
        unsigned int maxDepth)
    {
        const int3 idx = MortonToIndex(mortonCode, maxDepth);
        return make_float3(
            aabbMin.x + (static_cast<float>(idx.x) + 0.5f) * voxelSize,
            aabbMin.y + (static_cast<float>(idx.y) + 0.5f) * voxelSize,
            aabbMin.z + (static_cast<float>(idx.z) + 0.5f) * voxelSize
        );
    }
};
