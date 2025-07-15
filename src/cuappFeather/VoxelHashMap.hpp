
#pragma once

#include <cuda_runtime.h>

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

using VoxelKey = unsigned long long;
#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL
#define VALID_KEY(k) ((k) != EMPTY_KEY)
#define OFFSET_21BIT (1 << 20) // 1048576

struct Voxel
{
    int3 coordinate = make_int3(0, 0, 0);
    float3 normalSum = make_float3(0, 0, 0);
    float3 colorSum = make_float3(0, 0, 0);
    unsigned int count = 0;
};

struct alignas(8) VoxelHashEntry
{
    VoxelKey key;
    Voxel voxel;
};

struct VoxelHashMapInfo
{
    VoxelHashEntry* entries = nullptr;
    size_t capacity = 1 << 24; // default 16M slots
    uint8_t maxProbe = 64;
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
    unsigned int numberOfPoints,
    unsigned int* failedCount);

struct VoxelHashMap
{
    VoxelHashMapInfo info;

    void Initialize(size_t capacity = 1 << 24, uint8_t maxProbe = 64)
    {
        info.capacity = capacity;
        info.maxProbe = maxProbe;
        cudaMalloc(&info.entries, sizeof(VoxelHashEntry) * info.capacity);
        LaunchKernel(Kernel_ClearHashMap, info.capacity, info);
        cudaDeviceSynchronize();
    }

    void Terminate()
    {
        if (info.entries)
            cudaFree(info.entries);
        info.entries = nullptr;
    }

    void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, float voxelSize, unsigned int numberOfPoints)
    {
        unsigned int* d_failedCount;
        cudaMalloc(&d_failedCount, sizeof(unsigned int));
        cudaMemset(d_failedCount, 0, sizeof(unsigned int));

        LaunchKernel(Kernel_OccupyVoxelHashMap, numberOfPoints,
            info, d_positions, d_normals, d_colors, voxelSize, numberOfPoints, d_failedCount);

        unsigned int h_failedCount = 0;
        cudaMemcpy(&h_failedCount, d_failedCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_failedCount);

        printf("[!] Failed insertions: %u (%.2f%%)\n", h_failedCount, 100.0f * h_failedCount / numberOfPoints);
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

    __host__ __device__ static inline VoxelKey IndexToVoxelKey(const int3& coord)
    {
        uint64_t x = static_cast<uint64_t>(coord.x + OFFSET_21BIT) & 0x1FFFFF;
        uint64_t y = static_cast<uint64_t>(coord.y + OFFSET_21BIT) & 0x1FFFFF;
        uint64_t z = static_cast<uint64_t>(coord.z + OFFSET_21BIT) & 0x1FFFFF;
        return (x << 42) | (y << 21) | z;
    }

    __host__ __device__ static inline int3 VoxelKeyToIndex(VoxelKey key)
    {
        int x = static_cast<int>((key >> 42) & 0x1FFFFF) - OFFSET_21BIT;
        int y = static_cast<int>((key >> 21) & 0x1FFFFF) - OFFSET_21BIT;
        int z = static_cast<int>(key & 0x1FFFFF) - OFFSET_21BIT;
        return make_int3(x, y, z);
    }

    __host__ __device__ static inline int3 PositionToIndex(float3 pos, float voxelSize)
    {
        return make_int3(
            static_cast<int>(floorf(pos.x / voxelSize)),
            static_cast<int>(floorf(pos.y / voxelSize)),
            static_cast<int>(floorf(pos.z / voxelSize)));
    }

    __host__ __device__ static inline float3 IndexToPosition(int3 index, float voxelSize)
    {
        return make_float3(
            (index.x + 0.5f) * voxelSize,
            (index.y + 0.5f) * voxelSize,
            (index.z + 0.5f) * voxelSize);
    }

    __host__ __device__ static inline uint64_t mix64(uint64_t x)
    {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    __host__ __device__ static inline size_t hash(VoxelKey key, size_t capacity)
    {
        return mix64(key) & (capacity - 1);
    }

    __device__ static bool insert_accumulate(VoxelHashMapInfo info, VoxelKey key, float3 n, float3 c)
    {
        size_t idx = hash(key, info.capacity);
        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            unsigned long long* slot_key = reinterpret_cast<unsigned long long*>(&info.entries[slot].key);
            unsigned long long prev_key = atomicCAS(slot_key, EMPTY_KEY, key);

            if (prev_key == EMPTY_KEY)
            {
                info.entries[slot].voxel.coordinate = VoxelKeyToIndex(key);
                info.entries[slot].voxel.normalSum = n;
                info.entries[slot].voxel.colorSum = c;
                info.entries[slot].voxel.count = 1;
                return true;
            }
            else if (prev_key == key)
            {
                atomicAdd(&info.entries[slot].voxel.normalSum.x, n.x);
                atomicAdd(&info.entries[slot].voxel.normalSum.y, n.y);
                atomicAdd(&info.entries[slot].voxel.normalSum.z, n.z);

                atomicAdd(&info.entries[slot].voxel.colorSum.x, c.x);
                atomicAdd(&info.entries[slot].voxel.colorSum.y, c.y);
                atomicAdd(&info.entries[slot].voxel.colorSum.z, c.z);

                atomicAdd(&info.entries[slot].voxel.count, 1u);
                return true;
            }
        }
        return false;
    }
};

__global__ void Kernel_OccupyVoxelHashMap(
    VoxelHashMapInfo info,
    float3* positions,
    float3* normals,
    float3* colors,
    float voxelSize,
    unsigned int numberOfPoints,
    unsigned int* failedCount)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    float3 pos = positions[tid];
    int3 index = VoxelHashMap::PositionToIndex(pos, voxelSize);
    VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);

    float3 n = normals ? normals[tid] : make_float3(0, 0, 0);
    float3 c = colors ? colors[tid] : make_float3(0, 0, 0);

    if (!VoxelHashMap::insert_accumulate(info, key, n, c) && failedCount)
        atomicAdd(failedCount, 1);
}
