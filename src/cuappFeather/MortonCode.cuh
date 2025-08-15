#pragma once

#include <cuda_common.cuh>

//struct MortonCode
//{
//#pragma once
//#include <cstdint>
//#include <cuda_runtime.h>
//
//    // ---- Morton helpers (21 bits per axis) ----
//    __host__ __device__ static inline uint64_t PartBits21(uint32_t n)
//    {
//        uint64_t x = (uint64_t)(n & 0x1fffff);                 // keep 21 bits
//        x = (x | (x << 32)) & 0x1f00000000ffffULL;
//        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
//        x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
//        x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
//        x = (x | (x << 2)) & 0x1249249249249249ULL;
//        return x;
//    }
//
//    __host__ __device__ static inline uint32_t CompactBits21(uint64_t n)
//    {
//        n &= 0x1249249249249249ULL;
//        n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3ULL;
//        n = (n ^ (n >> 4)) & 0x100f00f00f00f00fULL;
//        n = (n ^ (n >> 8)) & 0x1f0000ff0000ffULL;
//        n = (n ^ (n >> 16)) & 0x1f00000000ffffULL;
//        n = (n ^ (n >> 32)) & 0x1fffffULL;
//        return (uint32_t)n;
//    }
//
//    __host__ __device__ static inline uint64_t EncodeMorton3D(uint32_t x, uint32_t y, uint32_t z)
//    {
//        return (PartBits21(z) << 2) | (PartBits21(y) << 1) | PartBits21(x);
//    }
//
//    __host__ __device__ static inline void DecodeMorton3D(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z)
//    {
//        x = CompactBits21(code);
//        y = CompactBits21(code >> 1);
//        z = CompactBits21(code >> 2);
//    }
//
//    // ---- Utility (safe math, clamp) ----
//    __host__ __device__ static inline uint32_t uclamp_u32(int v, uint32_t lo, uint32_t hi)
//    {
//        int vv = v < (int)lo ? (int)lo : v;
//        vv = vv > (int)hi ? (int)hi : vv;
//        return (uint32_t)vv;
//    }
//
//    __host__ __device__ static inline uint32_t min_u32(uint32_t a, uint32_t b)
//    {
//        return a < b ? a : b;
//    }
//
//    // Compute usable grid dimension (number of voxels) along one axis.
//    // Limited by both AABB extent / voxelSize and 2^maxDepth.
//    __host__ __device__ static inline uint32_t AxisGridDim(float extent, float voxelSize, unsigned int maxDepth)
//    {
//        if (voxelSize <= 0.0f)
//        {
//            return 1u;
//        }
//        // Ceil to cover the full AABB, but keep at least 1
//        float g = extent / voxelSize;
//        uint32_t aabbDim = (uint32_t)((g > 0.0f) ? (int)ceilf(g) : 1);
//        uint32_t depthCap = (maxDepth >= 31u) ? 0x80000000u : (1u << maxDepth); // avoid UB on shift
//        if (depthCap == 0u)
//        {
//            depthCap = 0x80000000u; // fallback if maxDepth >= 31 (we'll still clamp to 21-bit below)
//        }
//        // Also cap to 21-bit capacity (2^21)
//        uint32_t cap21 = 1u << 21; // 2,097,152
//        uint32_t cap = min_u32(depthCap, cap21);
//        uint32_t dim = min_u32(aabbDim, cap);
//        return dim > 0u ? dim : 1u;
//    }
//
//    // ---- API: Position <-> Morton ----
//    __host__ __device__ inline static uint64_t PositionToMorton(
//        const float3& p,
//        const float3& aabbMin,
//        const float3& aabbMax,
//        float voxelSize,
//        unsigned int maxDepth)
//    {
//        // Grid dimensions along each axis
//        const float3 ext =
//        {
//            aabbMax.x - aabbMin.x,
//            aabbMax.y - aabbMin.y,
//            aabbMax.z - aabbMin.z
//        };
//        const uint32_t dimX = AxisGridDim(ext.x, voxelSize, maxDepth);
//        const uint32_t dimY = AxisGridDim(ext.y, voxelSize, maxDepth);
//        const uint32_t dimZ = AxisGridDim(ext.z, voxelSize, maxDepth);
//
//        // Convert position to integer voxel indices
//        // Subtract a tiny epsilon so points exactly on aabbMax fall into the last voxel.
//        const float eps = 1e-6f;
//        float fx = (p.x - aabbMin.x) / voxelSize;
//        float fy = (p.y - aabbMin.y) / voxelSize;
//        float fz = (p.z - aabbMin.z) / voxelSize;
//
//        int ix = (int)floorf(fx - eps);
//        int iy = (int)floorf(fy - eps);
//        int iz = (int)floorf(fz - eps);
//
//        // Clamp to valid range and to 21-bit capacity
//        const uint32_t maxX = (dimX > 0u) ? (dimX - 1u) : 0u;
//        const uint32_t maxY = (dimY > 0u) ? (dimY - 1u) : 0u;
//        const uint32_t maxZ = (dimZ > 0u) ? (dimZ - 1u) : 0u;
//
//        uint32_t x = uclamp_u32(ix, 0u, maxX) & 0x1fffff;
//        uint32_t y = uclamp_u32(iy, 0u, maxY) & 0x1fffff;
//        uint32_t z = uclamp_u32(iz, 0u, maxZ) & 0x1fffff;
//
//        return EncodeMorton3D(x, y, z);
//    }
//
//    __host__ __device__ inline static float3 MortonToPosition(
//        uint64_t mortonCode,
//        const float3& aabbMin,
//        float voxelSize,
//        unsigned int /*maxDepth*/)
//    {
//        // Decode indices
//        uint32_t x, y, z;
//        DecodeMorton3D(mortonCode, x, y, z);
//
//        // Return voxel center position
//        float3 pos;
//        pos.x = aabbMin.x + (float(x) + 0.5f) * voxelSize;
//        pos.y = aabbMin.y + (float(y) + 0.5f) * voxelSize;
//        pos.z = aabbMin.z + (float(z) + 0.5f) * voxelSize;
//        return pos;
//    }
//
//};

struct MortonCode
{
    // --- Configuration ---
    static constexpr uint32_t XYZ_BITS = 20; // change if needed (<= 21)
    static constexpr uint32_t DEPTH_BITS = 4;  // change if needed
    static_assert(3 * XYZ_BITS + DEPTH_BITS <= 64, "Bit budget overflow");

    static constexpr uint32_t MAX_XYZ = (XYZ_BITS == 32) ? 0xFFFFFFFFu : ((1u << XYZ_BITS) - 1u);
    static constexpr uint32_t MAX_D = (DEPTH_BITS == 32) ? 0xFFFFFFFFu : ((1u << DEPTH_BITS) - 1u);

    // Depth is stored in the lowest DEPTH_BITS.
    // [63 : DEPTH_BITS] : interleaved morton of x,y,z (3*XYZ_BITS bits)
    // [DEPTH_BITS-1 : 0] : depth

    // --- Interleave helpers (support up to 21 bits safely) ---
    __host__ __device__ static inline uint64_t Part1By2(uint32_t v)
    {
        // Keep only lower 21 bits; upper bits are zeroed and won't affect masks.
        uint64_t x = static_cast<uint64_t>(v & 0x1FFFFF);
        x = (x | (x << 32)) & 0x1F00000000FFFFULL;
        x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
        x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL; // 0b001001...
        return x;
    }

    __host__ __device__ static inline uint32_t Compact1By2(uint64_t v)
    {
        v &= 0x1249249249249249ULL;
        v = (v ^ (v >> 2)) & 0x10C30C30C30C30C3ULL;
        v = (v ^ (v >> 4)) & 0x100F00F00F00F00FULL;
        v = (v ^ (v >> 8)) & 0x1F0000FF0000FFULL;
        v = (v ^ (v >> 16)) & 0x1F00000000FFFFULL;
        v = (v ^ (v >> 32)) & 0x1FFFFFULL;
        return static_cast<uint32_t>(v);
    }

    // Clamp indices to XYZ_BITS and depth to DEPTH_BITS.
    __host__ __device__ static inline void ClampToBitBudget(uint32_t& ix, uint32_t& iy, uint32_t& iz, uint32_t& depth)
    {
        if (ix > MAX_XYZ) ix = MAX_XYZ;
        if (iy > MAX_XYZ) iy = MAX_XYZ;
        if (iz > MAX_XYZ) iz = MAX_XYZ;
        if (DEPTH_BITS < 32) depth &= MAX_D;
    }

    // Encode indices + depth to packed 64-bit code.
    __host__ __device__ static inline uint64_t Encode(uint32_t ix, uint32_t iy, uint32_t iz, uint32_t depth)
    {
        ClampToBitBudget(ix, iy, iz, depth);

        // Interleave full 21-bit capable pattern, but we will only *use* XYZ_BITS from each.
        const uint64_t morton =
            (Part1By2(ix) << 0) |
            (Part1By2(iy) << 1) |
            (Part1By2(iz) << 2);

        // Shift left to leave DEPTH_BITS for depth.
        const uint64_t morton_shifted = morton << DEPTH_BITS;
        return morton_shifted | static_cast<uint64_t>(depth);
    }

    // Decode packed code to indices + depth.
    __host__ __device__ static inline void Decode(uint64_t code, uint32_t& ix, uint32_t& iy, uint32_t& iz, uint32_t& depth)
    {
        depth = static_cast<uint32_t>(code & ((DEPTH_BITS == 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << DEPTH_BITS) - 1ULL)));
        const uint64_t morton = code >> DEPTH_BITS;

        ix = Compact1By2(morton >> 0);
        iy = Compact1By2(morton >> 1);
        iz = Compact1By2(morton >> 2);

        // Mask down to the configured XYZ_BITS (important when XYZ_BITS < 21)
        if (XYZ_BITS < 21)
        {
            const uint32_t mask = (1u << XYZ_BITS) - 1u;
            ix &= mask; iy &= mask; iz &= mask;
        }
    }

    // Convert world position -> (ix,iy,iz) at a specific voxel size.
    __host__ __device__ static inline bool PositionToIndices(const float3& p,
        const float3& aabbMin,
        const float3& aabbMax,
        float voxelSize,
        uint32_t& ix, uint32_t& iy, uint32_t& iz)
    {
        if (p.x < aabbMin.x || p.x > aabbMax.x ||
            p.y < aabbMin.y || p.y > aabbMax.y ||
            p.z < aabbMin.z || p.z > aabbMax.z)
        {
            return false;
        }

        const float3 local = make_float3(p.x - aabbMin.x, p.y - aabbMin.y, p.z - aabbMin.z);

        // Nudge max-boundary points inward to keep them inside the last voxel.
        const float eps = 1e-6f;
        const float fx = (local.x / voxelSize);// - eps;
        const float fy = (local.y / voxelSize);// - eps;
        const float fz = (local.z / voxelSize);// - eps;

        const int64_t ixi = static_cast<int64_t>(floorf(fx));
        const int64_t iyi = static_cast<int64_t>(floorf(fy));
        const int64_t izi = static_cast<int64_t>(floorf(fz));

        if (ixi < 0 || iyi < 0 || izi < 0) return false;
        if (ixi > MAX_XYZ || iyi > MAX_XYZ || izi > MAX_XYZ) return false;

        ix = static_cast<uint32_t>(ixi);
        iy = static_cast<uint32_t>(iyi);
        iz = static_cast<uint32_t>(izi);
        return true;
    }

    // depth-aware: world position -> packed code
    // Convention: depth==maxDepth is leaf (smallest voxels).
    __host__ __device__ static inline uint64_t PositionToCode(const float3& p,
        const float3& aabbMin,
        const float3& aabbMax,
        float leafVoxelSize,
        uint32_t depth,
        uint32_t maxDepth)
    {
        // voxel size at this depth
        if (depth > maxDepth) return UINT64_MAX;
        const uint32_t levelsDown = maxDepth - depth;
        const float voxelSize = leafVoxelSize * static_cast<float>(1u << levelsDown);

		//printf("voxelSize : %f, levelsDown : %u, depth : %u\n", voxelSize, levelsDown, depth);

        uint32_t ix, iy, iz;
        if (!PositionToIndices(p, aabbMin, aabbMax, leafVoxelSize, ix, iy, iz))
        {
            return UINT64_MAX;
        }
        return Encode(ix, iy, iz, depth);
    }

    // depth-aware: packed code -> world position at voxel center
    __host__ __device__ static inline float3 CodeToPosition(uint64_t code,
        const float3& aabbMin,
        float leafVoxelSize,
        uint32_t maxDepth)
    {
        uint32_t ix, iy, iz, depth;
        Decode(code, ix, iy, iz, depth);

        const uint32_t levelsDown = (depth <= maxDepth) ? (maxDepth - depth) : 0u;
        const float voxelSize = leafVoxelSize * static_cast<float>(1u << levelsDown);

        float3 pos;
        pos.x = aabbMin.x + (static_cast<float>(ix) + 0.5f) * leafVoxelSize;
        pos.y = aabbMin.y + (static_cast<float>(iy) + 0.5f) * leafVoxelSize;
        pos.z = aabbMin.z + (static_cast<float>(iz) + 0.5f) * leafVoxelSize;
        return pos;
    }
};
