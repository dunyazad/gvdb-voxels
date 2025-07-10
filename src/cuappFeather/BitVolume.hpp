#pragma once

#include <cuda_runtime.h>

struct BitVolumeInfo
{
    dim3 dimensions = dim3(1000, 1000, 1000);
    unsigned int numberOfVoxels = 1000 * 1000 * 1000;

    dim3 allocated_dimensions = dim3(125, 125, 125);  // will be recomputed
    unsigned int numberOfAllocatedWords = 0;

    float3 volumeMin;
    uint32_t* volume = nullptr;
};

struct BitVolume
{
    BitVolumeInfo info;

    void Initialize(dim3 dimensions = dim3(1000, 1000, 1000))
    {
        info.dimensions = dimensions;
        info.numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

        unsigned int x = static_cast<unsigned int>(ceilf(dimensions.x / 32.0f));
        unsigned int y = dimensions.y;
        unsigned int z = dimensions.z;

        info.allocated_dimensions = dim3(x, y, z);
        info.numberOfAllocatedWords = x * y * z;

        cudaMalloc(&info.volume, sizeof(uint32_t) * info.numberOfAllocatedWords);
        cudaMemset(info.volume, 0, sizeof(uint32_t) * info.numberOfAllocatedWords);
    }

    void Terminate()
    {
        if (info.volume) cudaFree(info.volume);
        info.volume = nullptr;
    }

    void Clear()
    {
        cudaMemset(info.volume, 0, sizeof(uint32_t) * info.numberOfAllocatedWords);
    }

    __device__ bool isOccupied(const BitVolumeInfo& info, dim3 voxelIndex)
    {
        if (voxelIndex.x >= info.dimensions.x ||
            voxelIndex.y >= info.dimensions.y ||
            voxelIndex.z >= info.dimensions.z)
            return false;

        unsigned int wordX = voxelIndex.x / 32;
        unsigned int bitX = voxelIndex.x % 32;

        unsigned int wordIndex = voxelIndex.z * info.allocated_dimensions.y * info.allocated_dimensions.x +
            voxelIndex.y * info.allocated_dimensions.x +
            wordX;

        uint32_t word = info.volume[wordIndex];
        return (word >> bitX) & 1;
    }

    __device__ void setOccupied(BitVolumeInfo& info, dim3 voxelIndex)
    {
        if (voxelIndex.x >= info.dimensions.x ||
            voxelIndex.y >= info.dimensions.y ||
            voxelIndex.z >= info.dimensions.z)
            return;

        unsigned int wordX = voxelIndex.x / 32;
        unsigned int bitX = voxelIndex.x % 32;

        unsigned int wordIndex = voxelIndex.z * info.allocated_dimensions.y * info.allocated_dimensions.x +
            voxelIndex.y * info.allocated_dimensions.x +
            wordX;

        atomicOr(&info.volume[wordIndex], 1u << bitX);
    }
};
