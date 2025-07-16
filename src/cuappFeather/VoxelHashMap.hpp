
#pragma once

#include <cuda_runtime.h>
#include <cuda_vector_math.cuh>

#ifndef LaunchKernel
#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)
#endif

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[%s] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

#include <Serialization.hpp>

using VoxelKey = uint64_t;
#define EMPTY_KEY UINT64_MAX
#define VALID_KEY(k) ((k) != EMPTY_KEY)

struct Voxel
{
    uint3 coordinate = make_uint3(UINT32_MAX, UINT32_MAX, UINT32_MAX);
    float3 normalSum = make_float3(0, 0, 0);
    float3 colorSum = make_float3(0, 0, 0);
    unsigned int count = 0;
};

struct VoxelHashEntry
{
    VoxelKey key;
    Voxel voxel;
};

struct VoxelHashMapInfo
{
    VoxelHashEntry* entries = nullptr;
    size_t capacity = 1 << 24; // default 16M slots
    unsigned int maxProbe = 64;
};

__global__ void Kernel_ClearHashMap(VoxelHashMapInfo info)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= info.capacity) return;

    info.entries[idx].key = EMPTY_KEY;
    info.entries[idx].voxel = {};
}

__global__ void Kernel_OccupyVoxelHashMap(
    VoxelHashMapInfo info,
    float3* positions,
    float3* normals,
    float3* colors,
    float voxelSize,
    unsigned int numberOfPoints);

struct VoxelHashMap
{
    VoxelHashMapInfo info;

    void Initialize(size_t capacity = 1 << 24, unsigned int maxProbe = 64)
    {
        info.capacity = capacity;
        info.maxProbe = maxProbe;
        cudaMalloc(&info.entries, sizeof(VoxelHashEntry) * info.capacity);

        LaunchKernel(Kernel_ClearHashMap, info.capacity, info);
    }

    void Terminate()
    {
        if (nullptr != info.entries)
        {
            cudaFree(info.entries);
        }
        info.entries = nullptr;
    }

    void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, float voxelSize, unsigned int numberOfPoints)
    {
        LaunchKernel(Kernel_OccupyVoxelHashMap, numberOfPoints,
            info, d_positions, d_normals, d_colors, voxelSize, numberOfPoints);
    }

    void Serialize(const std::string& filename, float voxelSize) const
    {
        std::vector<VoxelHashEntry> hostEntries(info.capacity);
        cudaMemcpy(hostEntries.data(), info.entries, sizeof(VoxelHashEntry) * info.capacity, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        PLYFormat ply;
        for (const auto& entry : hostEntries)
        {
            if (!VALID_KEY(entry.key) || entry.voxel.count == 0) continue;

            int3 index = VoxelKeyToIndex(entry.key);
            float3 center = IndexToPosition(index, voxelSize);

            float invCount = 1.0f / static_cast<float>(entry.voxel.count);
            float3 normal = make_float3(
                entry.voxel.normalSum.x * invCount,
                entry.voxel.normalSum.y * invCount,
                entry.voxel.normalSum.z * invCount);

            float3 color = make_float3(
                entry.voxel.colorSum.x * invCount,
                entry.voxel.colorSum.y * invCount,
                entry.voxel.colorSum.z * invCount);

            ply.AddPoint(center.x, center.y, center.z);
            ply.AddNormal(normal.x, normal.y, normal.z);
            ply.AddColor(fminf(color.x, 1.0f), fminf(color.y, 1.0f), fminf(color.z, 1.0f));
        }

        ply.Serialize(filename);
    }

    __host__ __device__ inline static uint64_t expandBits(uint32_t v)
    {
        uint64_t x = v & 0x1fffff; // 21 bits
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }

    __host__ __device__ inline static uint32_t compactBits(uint64_t x)
    {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
        x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
        x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
        x = (x ^ (x >> 16)) & 0x1f00000000ffff;
        x = (x ^ (x >> 32)) & 0x1fffff;
        return static_cast<uint32_t>(x);
    }

    __host__ __device__ inline static VoxelKey IndexToVoxelKey(const int3& coord)
    {
        // Add offset to handle negative values
        const int OFFSET = 1 << 20; // 2^20 = 1048576
        return (expandBits(coord.z + OFFSET) << 2) |
            (expandBits(coord.y + OFFSET) << 1) |
            expandBits(coord.x + OFFSET);
    }

    __host__ __device__ inline static int3 VoxelKeyToIndex(VoxelKey key)
    {
        const int OFFSET = 1 << 20;
        int x = static_cast<int>(compactBits(key));
        int y = static_cast<int>(compactBits(key >> 1));
        int z = static_cast<int>(compactBits(key >> 2));
        return make_int3(x - OFFSET, y - OFFSET, z - OFFSET);
    }

    __host__ inline static int3 PositionToIndex_host(float3 pos, float voxelSize)
    {
        return make_int3(
            static_cast<int>(std::floor(pos.x / voxelSize)),
            static_cast<int>(std::floor(pos.y / voxelSize)),
            static_cast<int>(std::floor(pos.z / voxelSize)));
    }

    __device__ inline static int3 PositionToIndex_device(float3 pos, float voxelSize)
    {
        return make_int3(
            __float2int_rd(pos.x / voxelSize),
            __float2int_rd(pos.y / voxelSize),
            __float2int_rd(pos.z / voxelSize));
    }

    __host__ __device__ inline static int3 VoxelHashMap::PositionToIndex(float3 pos, float voxelSize)
    {
#if defined(__CUDA_ARCH__)
        return PositionToIndex_device(pos, voxelSize);
#else
        return PositionToIndex_host(pos, voxelSize);
#endif
    }

    __host__ __device__ inline static float3 VoxelHashMap::IndexToPosition(int3 index, float voxelSize)
    {
        return make_float3(
            (index.x + 0.5f) * voxelSize,
            (index.y + 0.5f) * voxelSize,
            (index.z + 0.5f) * voxelSize);
    }

    __host__ __device__ static inline size_t hash(VoxelKey key, size_t capacity)
    {
        // Use a simple multiplicative hash (can be replaced with better ones)
        key ^= (key >> 33);
        key *= 0xff51afd7ed558ccd;
        key ^= (key >> 33);
        key *= 0xc4ceb9fe1a85ec53;
        key ^= (key >> 33);
        return static_cast<size_t>(key) % capacity;
    }
};

__global__ void Kernel_OccupyVoxelHashMap(
    VoxelHashMapInfo info,
    float3* positions,
    float3* normals,
    float3* colors,
    float voxelSize,
    unsigned int numberOfPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    float3 pos = positions[tid];
    int3 index = VoxelHashMap::PositionToIndex(pos, voxelSize);
    VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
    float3 n = normals ? normals[tid] : make_float3(0, 0, 0);
    float3 c = colors ? colors[tid] : make_float3(0, 0, 0);

    size_t h = VoxelHashMap::hash(key, info.capacity);

    for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
    {
        size_t slot = (h + probe) % info.capacity;
        VoxelHashEntry* entry = &info.entries[slot];

        VoxelKey old = atomicCAS(reinterpret_cast<unsigned long long*>(&entry->key),
            EMPTY_KEY, key);

        if (old == EMPTY_KEY || old == key)
        {
            atomicAdd(&entry->voxel.normalSum.x, n.x);
            atomicAdd(&entry->voxel.normalSum.y, n.y);
            atomicAdd(&entry->voxel.normalSum.z, n.z);

            atomicAdd(&entry->voxel.colorSum.x, c.x);
            atomicAdd(&entry->voxel.colorSum.y, c.y);
            atomicAdd(&entry->voxel.colorSum.z, c.z);

            atomicAdd(&entry->voxel.count, 1u);

            entry->voxel.coordinate = make_uint3(index.x, index.y, index.z);

            return;
        }
    }
}

