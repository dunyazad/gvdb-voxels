#pragma once

#include <cuda_common.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <Serialization.hpp>

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator/(const float3& a, float s) {
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__host__ __device__ inline float3 make_float3(const uint3& u) {
    return make_float3((float)u.x, (float)u.y, (float)u.z);
}

__host__ __device__ inline float3 make_float3_uniform(float v) {
    return make_float3(v, v, v);
}

#ifndef OCTREE_NODE
#define OCTREE_NODE
struct OctreeNode {
    uint64_t mortonCode;
    int level;
    int parent;
    std::vector<int> children;
    int pointCount;
};
#endif

struct AABB {
    float3 minBound;
    float3 maxBound;
};

// ==============================
// CUDA Morton Utils
// ==============================
__device__ __forceinline__ uint64_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
    return (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
}

__device__ __forceinline__ uint32_t compactBits(uint64_t x) {
    x &= 0x9249249249249249;
    x = (x ^ (x >> 2)) & 0x30C30C30C30C30C3;
    x = (x ^ (x >> 4)) & 0xF00F00F00F00F00F;
    x = (x ^ (x >> 8)) & 0x00FF0000FF0000FF;
    x = (x ^ (x >> 16)) & 0xFFFF00000000FFFF;
    return (uint32_t)(x & 0x1FFFFF);
}

__device__ __forceinline__ uint3 mortonDecode(uint64_t code) {
    return make_uint3(compactBits(code >> 2), compactBits(code >> 1), compactBits(code));
}

__global__ void Kernel_MortonEncode(const float3* __restrict__ d_points,
    uint64_t* __restrict__ d_codes,
    int N, float voxelSize, float3 minBound) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float3 p = d_points[idx];
    uint3 gridCoord = make_uint3(
        static_cast<unsigned int>((p.x - minBound.x) / voxelSize),
        static_cast<unsigned int>((p.y - minBound.y) / voxelSize),
        static_cast<unsigned int>((p.z - minBound.z) / voxelSize)
    );
    d_codes[idx] = mortonEncode(gridCoord.x, gridCoord.y, gridCoord.z);
}

__global__ void Kernel_DecodeVoxelCenters(const uint64_t* __restrict__ d_codes,
    float3* __restrict__ d_centers,
    int numLeaves, float voxelSize, float3 minBound) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numLeaves) return;

    uint3 grid = mortonDecode(d_codes[idx]);
    float3 gridf = make_float3((float)grid.x, (float)grid.y, (float)grid.z);

    float half = voxelSize * 0.5f;
    d_centers[idx] = gridf * voxelSize + make_float3(half, half, half) + minBound;
}

inline int commonPrefixLen(uint64_t a, uint64_t b) {
    return _lzcnt_u64(a ^ b);
}

std::vector<OctreeNode> BuildOctreeFromMortonCodes(
    const std::vector<uint64_t>& sortedMortonCodes,
    const std::vector<int>& pointCounts,
    int maxDepth = 63) // 63-bit Morton
{
    int N = static_cast<int>(sortedMortonCodes.size());
    std::vector<OctreeNode> nodes(N);

    for (int i = 0; i < N; ++i) {
        nodes[i].mortonCode = sortedMortonCodes[i];
        nodes[i].pointCount = pointCounts[i];
        nodes[i].level = 0;
        nodes[i].parent = -1;
    }

    for (int i = 1; i < N; ++i) {
        int prefixLen = commonPrefixLen(sortedMortonCodes[i - 1], sortedMortonCodes[i]);
        nodes[i].level = maxDepth - prefixLen;

        if (nodes[i - 1].level < nodes[i].level) {
            nodes[i].parent = i - 1;
            nodes[i - 1].children.push_back(i);
        }
    }

    return nodes;
}


inline uint32_t compactBits_CPU(uint64_t x) {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x1f00000000ffff;
    x = (x ^ (x >> 32)) & 0x1fffff;
    return (uint32_t)x;
}

inline uint3 mortonDecode_CPU(uint64_t code) {
    return make_uint3(
        compactBits_CPU(code >> 2),
        compactBits_CPU(code >> 1),
        compactBits_CPU(code >> 0)
    );
}

inline uchar3 colormap(int level) {
    const uchar3 colors[] = {
        {255, 0, 0}, {255, 127, 0}, {255, 255, 0},
        {0, 255, 0}, {0, 0, 255}, {75, 0, 130}, {143, 0, 255}
    };
    return colors[level % 7];
}

void AddBoxLinesToPLY(PLYFormat& ply, const float3& center, float size, uint8_t r, uint8_t g, uint8_t b)
{
    float half = size * 0.5f;

    float3 corners[8] = {
        {center.x - half, center.y - half, center.z - half},
        {center.x + half, center.y - half, center.z - half},
        {center.x + half, center.y + half, center.z - half},
        {center.x - half, center.y + half, center.z - half},
        {center.x - half, center.y - half, center.z + half},
        {center.x + half, center.y - half, center.z + half},
        {center.x + half, center.y + half, center.z + half},
        {center.x - half, center.y + half, center.z + half}
    };

    static const int edgeIndices[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // bottom square
        {4,5},{5,6},{6,7},{7,4}, // top square
        {0,4},{1,5},{2,6},{3,7}  // vertical connections
    };

    int baseIndex = ply.GetPoints().size() / 3;

    for (int i = 0; i < 8; ++i)
    {
        ply.AddPoint(corners[i].x, corners[i].y, corners[i].z);
        ply.AddColor((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
    }

    for (int i = 0; i < 12; ++i)
    {
        ply.AddLineIndex(baseIndex + edgeIndices[i][0]);
        ply.AddLineIndex(baseIndex + edgeIndices[i][1]);
    }
}

// ==============================
// Main CUDA Octree Builder
// ==============================
std::vector<OctreeNode> BuildMortonOctree(const float3* d_inputPoints, int N, float voxelSize, float3 minBound)
{
    thrust::device_vector<uint64_t> mortonCodes(N);
    Kernel_MortonEncode <<<(N + 255) / 256, 256 >>> (
        d_inputPoints,
        thrust::raw_pointer_cast(mortonCodes.data()),
        N, voxelSize, minBound
        );
    cudaDeviceSynchronize();

    thrust::device_vector<float3> sortedPoints(d_inputPoints, d_inputPoints + N);
    thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), sortedPoints.begin());

    thrust::device_vector<uint64_t> uniqueCodes(N);
    thrust::device_vector<int> counts(N);
    auto end = thrust::reduce_by_key(
        mortonCodes.begin(), mortonCodes.end(),
        thrust::make_constant_iterator(1),
        uniqueCodes.begin(), counts.begin()
    );
    int numLeaves = end.first - uniqueCodes.begin();

    thrust::device_vector<float3> voxelCenters(numLeaves);
    Kernel_DecodeVoxelCenters << <(numLeaves + 255) / 256, 256 >> > (
        thrust::raw_pointer_cast(uniqueCodes.data()),
        thrust::raw_pointer_cast(voxelCenters.data()),
        numLeaves, voxelSize, minBound
        );
    cudaDeviceSynchronize();

    std::vector<float3> h_centers(numLeaves);
    std::vector<uint64_t> h_codes(numLeaves);
    std::vector<int> h_counts(numLeaves);

    cudaMemcpy(h_centers.data(), thrust::raw_pointer_cast(voxelCenters.data()), sizeof(float3) * numLeaves, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_codes.data(), thrust::raw_pointer_cast(uniqueCodes.data()), sizeof(uint64_t) * numLeaves, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), thrust::raw_pointer_cast(counts.data()), sizeof(int) * numLeaves, cudaMemcpyDeviceToHost);

    std::vector<OctreeNode> octree = BuildOctreeFromMortonCodes(h_codes, h_counts);

    //PLYFormat ply;
    //for (size_t i = 0; i < octree.size(); i++)
    //{
    //    const OctreeNode& node = octree[i];

    //    uint3 fullGrid = mortonDecode_CPU(node.mortonCode);
    //    uint3 shifted = make_uint3(
    //        fullGrid.x >> node.level,
    //        fullGrid.y >> node.level,
    //        fullGrid.z >> node.level
    //    );

    //    float voxelScale = voxelSize * (1 << node.level);
    //    float3 gridf = make_float3((float)shifted.x, (float)shifted.y, (float)shifted.z);
    //    float3 center = gridf * voxelScale + make_float3_uniform(voxelScale * 0.5f) + minBound;

    //    uchar3 color = colormap(node.level); // 계층 깊이에 따라 색 지정
    //    AddBoxLinesToPLY(ply, center, voxelScale, color.x, color.y, color.z);
    //}
    //ply.Serialize("../../res/3D/Octree.ply");

    return octree;
}
//
//__global__ void Kernel_MortonEncode(const float3* __restrict__ d_points,
//    uint64_t* __restrict__ d_codes,
//    int N, float voxelSize, float3 minBound) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx >= N) return;
//
//    float3 p = d_points[idx];
//    float3 diff;
//    diff.x = p.x - minBound.x;
//    diff.y = p.y - minBound.y;
//    diff.z = p.z - minBound.z;
//
//    uint3 gridCoord = make_uint3(diff.x / voxelSize,
//        diff.y / voxelSize,
//        diff.z / voxelSize);
//
//    d_codes[idx] = mortonEncode(gridCoord.x, gridCoord.y, gridCoord.z);
//}
//
//__global__ void Kernel_DecodeVoxelCenters(const uint64_t* __restrict__ d_codes,
//    float3* __restrict__ d_centers,
//    int numLeaves,
//    float voxelSize,
//    float3 minBound) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx >= numLeaves) return;
//
//    uint64_t code = d_codes[idx];
//    uint3 grid = mortonDecode(code);
//    float3 center = make_float3(
//        grid.x * voxelSize + voxelSize * 0.5f + minBound.x,
//        grid.y * voxelSize + voxelSize * 0.5f + minBound.y,
//        grid.z * voxelSize + voxelSize * 0.5f + minBound.z);
//    d_centers[idx] = center;
//}
//
//void BuildMortonOctree(const float3* d_inputPoints, int N, float voxelSize, AABB bbox)
//{
//    // 1. Morton 코드 생성
//    thrust::device_vector<uint64_t> mortonCodes(N);
//    Kernel_MortonEncode << <(N + 255) / 256, 256 >> > (
//        d_inputPoints,
//        thrust::raw_pointer_cast(mortonCodes.data()),
//        N, voxelSize, bbox.minBound
//        );
//    cudaDeviceSynchronize();
//
//    // 2. 정렬
//    thrust::device_vector<float3> sortedPoints(d_inputPoints, d_inputPoints + N);
//    thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), sortedPoints.begin());
//
//    // 3. unique leaf voxel 추출
//    thrust::device_vector<uint64_t> uniqueCodes(N);
//    thrust::device_vector<int> counts(N);
//    auto end = thrust::reduce_by_key(
//        mortonCodes.begin(), mortonCodes.end(),
//        thrust::make_constant_iterator(1),
//        uniqueCodes.begin(),
//        counts.begin()
//    );
//    int numLeaves = end.first - uniqueCodes.begin();
//
//    // 4. voxel 중심 좌표 디코딩
//    thrust::device_vector<float3> voxelCenters(numLeaves);
//    Kernel_DecodeVoxelCenters << <(numLeaves + 255) / 256, 256 >> > (
//        thrust::raw_pointer_cast(uniqueCodes.data()),
//        thrust::raw_pointer_cast(voxelCenters.data()),
//        numLeaves,
//        voxelSize,
//        bbox.minBound
//        );
//    cudaDeviceSynchronize();
//
//    std::vector<float3> h_centers(numLeaves);
//    cudaMemcpy(h_centers.data(), thrust::raw_pointer_cast(voxelCenters.data()),
//        sizeof(float3) * numLeaves, cudaMemcpyDeviceToHost);
//
//    PLYFormat ply;
//    for (size_t i = 0; i < h_centers.size(); i++)
//    {
//        ply.AddPoint(h_centers[i].x, h_centers[i].y, h_centers[i].z);
//    }
//    ply.Serialize("../../res/3D/Morton.ply");
//}