#pragma once

#include <cuda_common.cuh>

#include <unordered_set>
#include <queue>

/*
struct MortonCode
{
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

    // ---- Morton helpers (21 bits per axis) ----
    __host__ __device__ static inline uint64_t PartBits21(uint32_t n)
    {
        uint64_t x = (uint64_t)(n & 0x1fffff);                 // keep 21 bits
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;
        return x;
    }

    __host__ __device__ static inline uint32_t CompactBits21(uint64_t n)
    {
        n &= 0x1249249249249249ULL;
        n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3ULL;
        n = (n ^ (n >> 4)) & 0x100f00f00f00f00fULL;
        n = (n ^ (n >> 8)) & 0x1f0000ff0000ffULL;
        n = (n ^ (n >> 16)) & 0x1f00000000ffffULL;
        n = (n ^ (n >> 32)) & 0x1fffffULL;
        return (uint32_t)n;
    }

    __host__ __device__ static inline uint64_t EncodeMorton3D(uint32_t x, uint32_t y, uint32_t z)
    {
        return (PartBits21(z) << 2) | (PartBits21(y) << 1) | PartBits21(x);
    }

    __host__ __device__ static inline void DecodeMorton3D(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z)
    {
        x = CompactBits21(code);
        y = CompactBits21(code >> 1);
        z = CompactBits21(code >> 2);
    }

    // ---- Utility (safe math, clamp) ----
    __host__ __device__ static inline uint32_t uclamp_u32(int v, uint32_t lo, uint32_t hi)
    {
        int vv = v < (int)lo ? (int)lo : v;
        vv = vv > (int)hi ? (int)hi : vv;
        return (uint32_t)vv;
    }

    __host__ __device__ static inline uint32_t min_u32(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    // Compute usable grid dimension (number of voxels) along one axis.
    // Limited by both AABB extent / voxelSize and 2^maxDepth.
    __host__ __device__ static inline uint32_t AxisGridDim(float extent, float voxelSize, unsigned int maxDepth)
    {
        if (voxelSize <= 0.0f)
        {
            return 1u;
        }
        // Ceil to cover the full AABB, but keep at least 1
        float g = extent / voxelSize;
        uint32_t aabbDim = (uint32_t)((g > 0.0f) ? (int)ceilf(g) : 1);
        uint32_t depthCap = (maxDepth >= 31u) ? 0x80000000u : (1u << maxDepth); // avoid UB on shift
        if (depthCap == 0u)
        {
            depthCap = 0x80000000u; // fallback if maxDepth >= 31 (we'll still clamp to 21-bit below)
        }
        // Also cap to 21-bit capacity (2^21)
        uint32_t cap21 = 1u << 21; // 2,097,152
        uint32_t cap = min_u32(depthCap, cap21);
        uint32_t dim = min_u32(aabbDim, cap);
        return dim > 0u ? dim : 1u;
    }

    // ---- API: Position <-> Morton ----
    __host__ __device__ inline static uint64_t PositionToMorton(
        const float3& p,
        const float3& aabbMin,
        const float3& aabbMax,
        float voxelSize,
        unsigned int maxDepth)
    {
        // Grid dimensions along each axis
        const float3 ext =
        {
            aabbMax.x - aabbMin.x,
            aabbMax.y - aabbMin.y,
            aabbMax.z - aabbMin.z
        };
        const uint32_t dimX = AxisGridDim(ext.x, voxelSize, maxDepth);
        const uint32_t dimY = AxisGridDim(ext.y, voxelSize, maxDepth);
        const uint32_t dimZ = AxisGridDim(ext.z, voxelSize, maxDepth);

        // Convert position to integer voxel indices
        // Subtract a tiny epsilon so points exactly on aabbMax fall into the last voxel.
        const float eps = 1e-6f;
        float fx = (p.x - aabbMin.x) / voxelSize;
        float fy = (p.y - aabbMin.y) / voxelSize;
        float fz = (p.z - aabbMin.z) / voxelSize;

        int ix = (int)floorf(fx - eps);
        int iy = (int)floorf(fy - eps);
        int iz = (int)floorf(fz - eps);

        // Clamp to valid range and to 21-bit capacity
        const uint32_t maxX = (dimX > 0u) ? (dimX - 1u) : 0u;
        const uint32_t maxY = (dimY > 0u) ? (dimY - 1u) : 0u;
        const uint32_t maxZ = (dimZ > 0u) ? (dimZ - 1u) : 0u;

        uint32_t x = uclamp_u32(ix, 0u, maxX) & 0x1fffff;
        uint32_t y = uclamp_u32(iy, 0u, maxY) & 0x1fffff;
        uint32_t z = uclamp_u32(iz, 0u, maxZ) & 0x1fffff;

        return EncodeMorton3D(x, y, z);
    }

    __host__ __device__ inline static float3 MortonToPosition(
        uint64_t mortonCode,
        const float3& aabbMin,
        float voxelSize,
        unsigned int maxDepth)
    {
        // Decode indices
        uint32_t x, y, z;
        DecodeMorton3D(mortonCode, x, y, z);

        // Return voxel center position
        float3 pos;
        pos.x = aabbMin.x + (float(x) + 0.5f) * voxelSize;
        pos.y = aabbMin.y + (float(y) + 0.5f) * voxelSize;
        pos.z = aabbMin.z + (float(z) + 0.5f) * voxelSize;
        return pos;
    }

};
*/

/*
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
*/

/*
//==============================================================================
// 21-bit per axis limits for 64-bit 3D Morton
//==============================================================================
#define MORTON21_MASK (0x1FFFFFu) // 21 bits
#define MORTON21_BIAS (1 << 20)   // maps [-2^20, 2^20-1] -> [0, 2^21-1]

//==============================================================================
// Bit interleave helpers (3D): "spread" 21 bits so that each original bit
// sits 2 zeros apart: b -> b00b00b...
// Reference masks are standard for 64-bit 3-way interleave.
//==============================================================================

struct MortonCode
{
    __host__ __device__ __forceinline__ static uint64_t spreadBy2_21(uint32_t v)
    {
        uint64_t x = (uint64_t)(v & MORTON21_MASK);
        x = (x | (x << 32)) & 0x1F00000000FFFFull;
        x = (x | (x << 16)) & 0x1F0000FF0000FFull;
        x = (x | (x << 8)) & 0x100F00F00F00F00Full;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
        x = (x | (x << 2)) & 0x1249249249249249ull; // final: ...001001001...
        return x;
    }

    __host__ __device__ __forceinline__ static uint32_t compactBy2_21(uint64_t v)
    {
        uint64_t x = v & 0x1249249249249249ull;
        x = (x ^ (x >> 2)) & 0x10C30C30C30C30C3ull;
        x = (x ^ (x >> 4)) & 0x100F00F00F00F00Full;
        x = (x ^ (x >> 8)) & 0x1F0000FF0000FFull;
        x = (x ^ (x >> 16)) & 0x1F00000000FFFFull;
        x = (x ^ (x >> 32)) & MORTON21_MASK;
        return (uint32_t)x;
    }

    //==============================================================================
    // Encode/Decode 3D (signed int inputs; bias 적용)
    //==============================================================================

    __host__ __device__ __forceinline__ static uint64_t morton3D_encode_biased(int x, int y, int z)
    {
        // Map to unsigned 21-bit with bias and clamp (optional 안전장치)
        uint32_t ux = (uint32_t)((x + MORTON21_BIAS) < 0 ? 0 : (x + MORTON21_BIAS));
        uint32_t uy = (uint32_t)((y + MORTON21_BIAS) < 0 ? 0 : (y + MORTON21_BIAS));
        uint32_t uz = (uint32_t)((z + MORTON21_BIAS) < 0 ? 0 : (z + MORTON21_BIAS));

        ux &= MORTON21_MASK;
        uy &= MORTON21_MASK;
        uz &= MORTON21_MASK;

        return (spreadBy2_21(ux) << 0) |
            (spreadBy2_21(uy) << 1) |
            (spreadBy2_21(uz) << 2);
    }

    __host__ __device__ __forceinline__ static void morton3D_decode_biased(uint64_t code, int& x, int& y, int& z)
    {
        uint32_t ux = compactBy2_21(code >> 0);
        uint32_t uy = compactBy2_21(code >> 1);
        uint32_t uz = compactBy2_21(code >> 2);

        x = (int)ux - MORTON21_BIAS;
        y = (int)uy - MORTON21_BIAS;
        z = (int)uz - MORTON21_BIAS;
    }

    //==============================================================================
    // Encode/Decode 3D (비편향: 입력이 이미 [0, 2^21-1] 범위인 경우)
    //==============================================================================

    __host__ __device__ __forceinline__ static uint64_t morton3D_encode_u21(uint32_t x, uint32_t y, uint32_t z)
    {
        x &= MORTON21_MASK; y &= MORTON21_MASK; z &= MORTON21_MASK;
        return (spreadBy2_21(x) << 0) |
            (spreadBy2_21(y) << 1) |
            (spreadBy2_21(z) << 2);
    }

    __host__ __device__ __forceinline__ static void morton3D_decode_u21(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z)
    {
        x = compactBy2_21(code >> 0);
        y = compactBy2_21(code >> 1);
        z = compactBy2_21(code >> 2);
    }

    //==============================================================================
    // 2D variant (up to 32-bit code; 축당 최대 16-bit 권장)
    //==============================================================================

    __host__ __device__ __forceinline__ static uint32_t spreadBy1_16(uint32_t v)
    {
        uint32_t x = v & 0x0000FFFFu;
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        x = (x | (x << 2)) & 0x33333333u;
        x = (x | (x << 1)) & 0x55555555u; // ...010101...
        return x;
    }

    __host__ __device__ __forceinline__ static uint32_t compactBy1_16(uint32_t v)
    {
        uint32_t x = v & 0x55555555u;
        x = (x ^ (x >> 1)) & 0x33333333u;
        x = (x ^ (x >> 2)) & 0x0F0F0F0Fu;
        x = (x ^ (x >> 4)) & 0x00FF00FFu;
        x = (x ^ (x >> 8)) & 0x0000FFFFu;
        return x;
    }

    __host__ __device__ __forceinline__ static uint32_t morton2D_encode_u16(uint32_t x, uint32_t y)
    {
        return (spreadBy1_16(x) << 0) | (spreadBy1_16(y) << 1);
    }

    __host__ __device__ __forceinline__ static void morton2D_decode_u16(uint32_t code, uint32_t& x, uint32_t& y)
    {
        x = compactBy1_16(code >> 0);
        y = compactBy1_16(code >> 1);
    }

    __host__ __device__ __forceinline__ static int morton_max_depth(
        float domainExtent,
        float voxelSize,
        int maxBitsPerAxis = 21)
    {
        if (domainExtent <= 0.0 || voxelSize <= 0.0)
        {
            return 0;
        }

        float ratio = domainExtent / voxelSize;
        int depth = static_cast<int>(std::ceil(std::log2(ratio)));
        return std::min(depth, maxBitsPerAxis);
    }

    __host__ __device__ __forceinline__ static int morton_max_depth(
        float3 aabbMin,
        float3 aabbMax,
        float voxelSize,
        int maxBitsPerAxis = 21)
    {
        float dx = aabbMax.x - aabbMin.x;
        float dy = aabbMax.y - aabbMin.y;
        float dz = aabbMax.z - aabbMin.z;
        float domainExtent = fmaxf(dx, fmaxf(dy, dz));

        if (domainExtent <= 0.0 || voxelSize <= 0.0)
        {
            return 0;
        }

        float ratio = domainExtent / voxelSize;
        int depth = static_cast<int>(std::ceil(std::log2(ratio)));
        return std::min(depth, maxBitsPerAxis);
    }

    //==============================================================================
    // 예시 커널: 좌표 배열 -> Morton code, 그리고 역변환 검증
    //==============================================================================
};

__global__ void Kernel_MortonCode_Encode(const uint3* in, uint64_t* out, int n);

__global__ void Kernel_MortonCode_Decode(const uint64_t* codes, uint3* out, int n);
*/

struct MortonCode
{
    __host__ __device__ __forceinline__ static cuAABB GetDomainAABB(float3 aabbMin, float3 aabbMax)
    {
        const float mx = fminf(aabbMin.x, aabbMax.x);
        const float my = fminf(aabbMin.y, aabbMax.y);
        const float mz = fminf(aabbMin.z, aabbMax.z);

        const float Mx = fmaxf(aabbMin.x, aabbMax.x);
        const float My = fmaxf(aabbMin.y, aabbMax.y);
        const float Mz = fmaxf(aabbMin.z, aabbMax.z);

        const float cx = 0.5f * (Mx + mx);
        const float cy = 0.5f * (My + my);
        const float cz = 0.5f * (Mz + mz);

        const float dx = Mx - mx;
        const float dy = My - my;
        const float dz = Mz - mz;

        const float extent = fmaxf(dx, fmaxf(dy, dz));
        const float half_extent = extent * 0.5f;

        return {
            make_float3(cx - half_extent, cy - half_extent, cz - half_extent),
            make_float3(cx + half_extent, cy + half_extent, cz + half_extent) };
    }

    __host__ __device__ __forceinline__ static cuAABB GetDomainAABB(float3 aabbMin, float3 aabbMax, float& domainExtent)
    {
        const float mx = fminf(aabbMin.x, aabbMax.x);
        const float my = fminf(aabbMin.y, aabbMax.y);
        const float mz = fminf(aabbMin.z, aabbMax.z);

        const float Mx = fmaxf(aabbMin.x, aabbMax.x);
        const float My = fmaxf(aabbMin.y, aabbMax.y);
        const float Mz = fmaxf(aabbMin.z, aabbMax.z);

        const float cx = 0.5f * (Mx + mx);
        const float cy = 0.5f * (My + my);
        const float cz = 0.5f * (Mz + mz);

        const float dx = Mx - mx;
        const float dy = My - my;
        const float dz = Mz - mz;

        domainExtent = fmaxf(dx, fmaxf(dy, dz));
        const float half_extent = 0.5f * domainExtent;

        return {
            make_float3(cx - half_extent, cy - half_extent, cz - half_extent),
            make_float3(cx + half_extent, cy + half_extent, cz + half_extent) };
    }

    __host__ __device__ __forceinline__ static float GetDomainExtent(cuAABB aabb)
    {
        const float dx = fabsf(aabb.max.x - aabb.min.x);
        const float dy = fabsf(aabb.max.y - aabb.min.y);
        const float dz = fabsf(aabb.max.z - aabb.min.z);
        return fmaxf(dx, fmaxf(dy, dz));
    }
    
    __host__ __device__ __forceinline__ static unsigned int GetMaxDepth()
    {
        return 21u;
    }

    __host__ __device__ __forceinline__ static float GetDomainVoxelSize(
        float3 aabbMin, float3 aabbMax)
    {
        const float extent = GetDomainExtent({ aabbMin, aabbMax });
        if (!(extent > 0.0f))
        {
            return 0.0f;
        }
        constexpr double N = (double)(1ull << 21);
        return (float)(extent / N);
    }

    __host__ __device__ __forceinline__ static uint64_t Part1By2(uint32_t x)
    {
        // 21 bits만 사용
        x &= 0x001fffffU;

        // 64비트로 승격해서 시프트/마스크 수행
        uint64_t v = (uint64_t)x;
        v = (v | (v << 32)) & 0x001f00000000ffffULL;
        v = (v | (v << 16)) & 0x001f0000ff0000ffULL;
        v = (v | (v << 8)) & 0x100f00f00f00f00FULL;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2)) & 0x1249249249249249ULL;
        return v;
    }

    __host__ __device__ __forceinline__ static uint32_t Compact1By2(uint64_t x)
    {
        x &= 0x1249249249249249ULL;
        x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3ULL;
        x = (x ^ (x >> 4)) & 0x100f00f00f00f00FULL;
        x = (x ^ (x >> 8)) & 0x001f0000ff0000ffULL;
        x = (x ^ (x >> 16)) & 0x001f00000000ffffULL;
        x = (x ^ (x >> 32)) & 0x00000000001ffffFULL;
        return (uint32_t)x;
    }

    __host__ __device__ __forceinline__ static uint64_t EncodeMorton3D_21b(
        uint32_t ix, uint32_t iy, uint32_t iz)
    {
        return (Part1By2(ix) << 0) | (Part1By2(iy) << 1) | (Part1By2(iz) << 2);
    }

    __host__ __device__ __forceinline__ static void DecodeMorton3D_21b(
        uint64_t code, uint32_t& ix, uint32_t& iy, uint32_t& iz)
    {
        ix = Compact1By2(code >> 0);
        iy = Compact1By2(code >> 1);
        iz = Compact1By2(code >> 2);
    }

    __host__ __device__ __forceinline__ static uint64_t FromPosition(
        const float3 position, const float3 aabbMin, const float3 aabbMax)
    {
        float domainExtent = 0.0f;
        const cuAABB dom = GetDomainAABB(aabbMin, aabbMax, domainExtent);

        if (!(domainExtent > 0.0f))
        {
            return 0ull;
        }

        // 2) [0, 1] 정규화 (double로 중간 계산 안정화)
        const double ex = (double)domainExtent;
        const double tx = ((double)position.x - (double)dom.min.x) / ex;
        const double ty = ((double)position.y - (double)dom.min.y) / ex;
        const double tz = ((double)position.z - (double)dom.min.z) / ex;

        // 경계 클램프
        const double nx = tx < 0.0 ? 0.0 : (tx > 1.0 ? 1.0 : tx);
        const double ny = ty < 0.0 ? 0.0 : (ty > 1.0 ? 1.0 : ty);
        const double nz = tz < 0.0 ? 0.0 : (tz > 1.0 ? 1.0 : tz);

        // 3) 21bit 정수 격자에 매핑
        //    (1<<21)-1 = 2097151. max 포함되도록 floor 후 클램프
        const uint32_t mask = 0x001fffff; // 21 bits
        const double    s = (double)mask;

        uint32_t ix = (uint32_t)floor(nx * s);
        uint32_t iy = (uint32_t)floor(ny * s);
        uint32_t iz = (uint32_t)floor(nz * s);

        if (ix > mask) ix = mask;
        if (iy > mask) iy = mask;
        if (iz > mask) iz = mask;

        // 4) interleave
        return EncodeMorton3D_21b(ix, iy, iz);
    }

    __host__ __device__ __forceinline__ static float3 ToPosition(
        const uint64_t code, const float3 aabbMin, const float3 aabbMax)
    {
        // 1) 도메인 AABB (정방형) 확보
        float domainExtent = 0.0f;
        const cuAABB dom = GetDomainAABB(aabbMin, aabbMax, domainExtent);

        if (!(domainExtent > 0.0f))
        {
            return make_float3(0.f, 0.f, 0.f);
        }

        // 2) 디코드 → 정수 격자 인덱스(ix,iy,iz) in [0, 2^21-1]
        uint32_t ix, iy, iz;
        DecodeMorton3D_21b(code, ix, iy, iz);

        const double mask = double(0x001fffff); // 21 bits
        const double nx = double(ix) / mask;    // [0,1]
        const double ny = double(iy) / mask;
        const double nz = double(iz) / mask;

        // 3) 정규화 해제: 로어 코너(셀 최소 좌표)
        const double ex = double(domainExtent);
        const double px = double(dom.min.x) + nx * ex;
        const double py = double(dom.min.y) + ny * ex;
        const double pz = double(dom.min.z) + nz * ex;

        return make_float3((float)px, (float)py, (float)pz);
    }

    __host__ __device__ __forceinline__ static float3 ToCellCenter(
        const uint64_t code, const float3 aabbMin, const float3 aabbMax)
    {
        float domainExtent = 0.0f;
        const cuAABB dom = GetDomainAABB(aabbMin, aabbMax, domainExtent);
        if (!(domainExtent > 0.0f))
        {
            return make_float3(0.f, 0.f, 0.f);
        }

        // 디코드
        uint32_t ix, iy, iz;
        DecodeMorton3D_21b(code, ix, iy, iz);

        // N = 2^21
        constexpr double N = (double)(1ull << 21);
        const double cellSize = (double)domainExtent / N;

        const double cx = (double)dom.min.x + ((double)ix + 0.5) * cellSize;
        const double cy = (double)dom.min.y + ((double)iy + 0.5) * cellSize;
        const double cz = (double)dom.min.z + ((double)iz + 0.5) * cellSize;

        return make_float3((float)cx, (float)cy, (float)cz);
    }

    __host__ __device__ __forceinline__
        static size_t LowerBound(const uint64_t* data, size_t n, uint64_t key)
    {
        size_t lo = 0, hi = n;
        while (lo < hi)
        {
            size_t mid = lo + ((hi - lo) >> 1);
            uint64_t v = data[mid];
            if (v < key) lo = mid + 1;
            else         hi = mid;
        }
        return lo; // 첫 번째 >= key 의 인덱스 (없으면 n)
    }

    __host__ __device__ __forceinline__
        static size_t UpperBound(const uint64_t* data, size_t n, uint64_t key)
    {
        size_t lo = 0, hi = n;
        while (lo < hi)
        {
            size_t mid = lo + ((hi - lo) >> 1);
            uint64_t v = data[mid];
            if (v <= key) lo = mid + 1;
            else          hi = mid;
        }
        return lo; // 첫 번째 > key 의 인덱스 (없으면 n)
    }

    __host__ __device__ __forceinline__
        static bool BinarySearch(const uint64_t* data, size_t n, uint64_t key, size_t& idx)
    {
        idx = LowerBound(data, n, key);
        return (idx < n&& data[idx] == key);
    }

    __host__ __device__ __forceinline__
        static void EqualRange(const uint64_t* data, size_t n, uint64_t key, size_t& first, size_t& last)
    {
        first = LowerBound(data, n, key);
        last = UpperBound(data, n, key); // [first,last)
    }

    struct NearestResult
    {
        int idx;
        uint64_t code;
        float dist2;
    };


// ============================================================================
// MortonCode::CodeBVH  (code-only exact NN; domain = GetDomainAABB 정방화)
// 빌드 O(N log N), 질의 평균 O(log N). 셀 중심 == 포인트 가정.
// ============================================================================
    struct MortonCode::CodeBVH
    {
        struct Node
        {
            uint32_t begin;   // index[] 구간 [begin, end)
            uint32_t end;
            uint32_t left;    // children index (or UINT32_MAX if leaf)
            uint32_t right;
            float3   bmin;    // AABB of cell-centers in world coords
            float3   bmax;
            uint8_t  axis;    // split axis (0:x,1:y,2:z)
            bool     leaf;
        };

        // 입력 참조(외부 소유). codes는 정렬되어 있을 필요 없음.
        const uint64_t* codes = nullptr;
        size_t          n = 0;

        // 내부 버퍼
        std::vector<uint32_t> idx;   // 재배열용 인덱스
        std::vector<Node>     nodes;

        // 도메인(정방화) 정보
        float3 domMin;
        double cellSize;

        // -------------------- 유틸 --------------------
        static __host__ __device__ __forceinline__ double aabbDist2(const float3 p, const float3 bmin, const float3 bmax)
        {
            const double px = (double)p.x, py = (double)p.y, pz = (double)p.z;
            const double x = (px < bmin.x) ? (bmin.x - px) : ((px > bmax.x) ? (px - bmax.x) : 0.0);
            const double y = (py < bmin.y) ? (bmin.y - py) : ((py > bmax.y) ? (py - bmax.y) : 0.0);
            const double z = (pz < bmin.z) ? (bmin.z - pz) : ((pz > bmax.z) ? (pz - bmax.z) : 0.0);
            return x * x + y * y + z * z;
        }

        __host__ __device__ __forceinline__ float3 centerOf(uint64_t code) const
        {
            uint32_t ix, iy, iz;
            MortonCode::DecodeMorton3D_21b(code, ix, iy, iz);
            const double cx = (double)domMin.x + ((double)ix + 0.5) * cellSize;
            const double cy = (double)domMin.y + ((double)iy + 0.5) * cellSize;
            const double cz = (double)domMin.z + ((double)iz + 0.5) * cellSize;
            return make_float3((float)cx, (float)cy, (float)cz);
        }

        void computeBounds(uint32_t begin, uint32_t end, float3& bmin, float3& bmax) const
        {
            float3 mn = make_float3(+FLT_MAX, +FLT_MAX, +FLT_MAX);
            float3 mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for (uint32_t k = begin; k < end; ++k)
            {
                const float3 c = centerOf(codes[idx[k]]);
                mn.x = fminf(mn.x, c.x);  mn.y = fminf(mn.y, c.y);  mn.z = fminf(mn.z, c.z);
                mx.x = fmaxf(mx.x, c.x);  mx.y = fmaxf(mx.y, c.y);  mx.z = fmaxf(mx.z, c.z);
            }
            bmin = mn; bmax = mx;
        }

        struct CmpAxis
        {
            const MortonCode::CodeBVH* self;
            int axis;
            bool operator()(uint32_t a, uint32_t b) const
            {
                const float3 ca = self->centerOf(self->codes[a]);
                const float3 cb = self->centerOf(self->codes[b]);
                if (axis == 0) return ca.x < cb.x;
                if (axis == 1) return ca.y < cb.y;
                return ca.z < cb.z;
            }
        };

        uint32_t buildRecursive(uint32_t begin, uint32_t end, uint32_t leafSize)
        {
            Node node;
            node.begin = begin;
            node.end = end;
            node.left = node.right = UINT32_MAX;
            node.leaf = ((end - begin) <= leafSize);

            computeBounds(begin, end, node.bmin, node.bmax);

            const uint32_t id = (uint32_t)nodes.size();
            nodes.push_back(node);

            if (node.leaf)
            {
                nodes[id].axis = 0;
                return id;
            }

            const float3 ext = make_float3(node.bmax.x - node.bmin.x,
                node.bmax.y - node.bmin.y,
                node.bmax.z - node.bmin.z);
            int axis = 0;
            if (ext.y > ext.x && ext.y >= ext.z) axis = 1;
            else if (ext.z > ext.x && ext.z >= ext.y) axis = 2;
            nodes[id].axis = (uint8_t)axis;

            const uint32_t mid = begin + ((end - begin) >> 1);
            std::nth_element(idx.begin() + begin, idx.begin() + mid, idx.begin() + end,
                [&](uint32_t ia, uint32_t ib)
                {
                    const float3 ca = centerOf(codes[ia]);
                    const float3 cb = centerOf(codes[ib]);
                    if (axis == 0) return ca.x < cb.x;
                    if (axis == 1) return ca.y < cb.y;
                    return ca.z < cb.z;
                });

            const uint32_t L = buildRecursive(begin, mid, leafSize);
            const uint32_t R = buildRecursive(mid, end, leafSize);
            nodes[id].left = L;
            nodes[id].right = R;
            return id;
        }

        // -------------------- 공개 API --------------------
        void Build(const uint64_t* inCodes, size_t count, float3 aabbMin, float3 aabbMax, uint32_t leafSize = 8)
        {
            codes = inCodes; n = (uint32_t)count;
            idx.resize(n);
            for (uint32_t i = 0; i < n; ++i) idx[i] = i;

            float ex;
            const cuAABB dom = MortonCode::GetDomainAABB(aabbMin, aabbMax, ex);
            domMin = dom.min;
            cellSize = (double)ex / (double)(1ull << 21);

            nodes.clear();
            nodes.reserve(n * 2);
            if (n > 0)
            {
                buildRecursive(0, (uint32_t)n, leafSize);
            }
        }

        MortonCode::NearestResult Nearest(const float3 q) const
        {
            MortonCode::NearestResult out{ -1, 0ull, FLT_MAX };
            if (n == 0 || nodes.empty())
            {
                return out;
            }

            struct QNode
            {
                uint32_t id;
                double   bound2;
                bool operator<(const QNode& o) const { return bound2 > o.bound2; } // min-heap
            };
            std::priority_queue<QNode> pq;
            pq.push({ 0u, aabbDist2(q, nodes[0].bmin, nodes[0].bmax) });

            double best = std::numeric_limits<double>::infinity();
            int    bestIdx = -1;
            uint64_t bestCode = 0;

            while (!pq.empty())
            {
                const QNode cur = pq.top(); pq.pop();
                if (cur.bound2 >= best) continue;

                const Node& nd = nodes[cur.id];
                if (nd.leaf)
                {
                    for (uint32_t k = nd.begin; k < nd.end; ++k)
                    {
                        const uint32_t i = idx[k];
                        const uint64_t code = codes[i];
                        const float3 c = centerOf(code);
                        const double dx = (double)c.x - (double)q.x;
                        const double dy = (double)c.y - (double)q.y;
                        const double dz = (double)c.z - (double)q.z;
                        const double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < best)
                        {
                            best = d2;
                            bestIdx = (int)i;     // codes의 원래 인덱스
                            bestCode = code;
                        }
                    }
                    continue;
                }

                const uint32_t L = nd.left, R = nd.right;
                const double dl = aabbDist2(q, nodes[L].bmin, nodes[L].bmax);
                const double dr = aabbDist2(q, nodes[R].bmin, nodes[R].bmax);

                // 더 유망한 쪽을 먼저 탐색
                if (dl < dr)
                {
                    if (dl < best) pq.push({ L, dl });
                    if (dr < best) pq.push({ R, dr });
                }
                else
                {
                    if (dr < best) pq.push({ R, dr });
                    if (dl < best) pq.push({ L, dl });
                }
            }

            if (bestIdx >= 0)
            {
                out.idx = bestIdx;
                out.code = bestCode;
                out.dist2 = (float)best;
            }
            return out;
        }
    };

};
