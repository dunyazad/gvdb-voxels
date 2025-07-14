#pragma once

#include <cuda_runtime.h>
#include <Serialization.hpp>

#ifndef LaunchKernel
#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)
#endif

template<typename Voxel>
struct DenseGridInfo {
    Voxel* d_voxels = nullptr;
    unsigned int* d_visitedFlags = nullptr;

    dim3 dimensions = dim3(400, 400, 400);
    unsigned int numberOfVoxels = 400 * 400 * 400;
    float voxelSize = 0.1f;
    float3 gridMin = make_float3(-200, -200, -200);

    unsigned int* d_numberOfOccupiedVoxels = nullptr;
    uint3* d_occupiedVoxelIndices = nullptr;
    unsigned int h_numberOfOccupiedVoxels = 0;
    unsigned int h_occupiedCapacity = 0;
};

template<typename Voxel>
__global__ void Kernel_Occupy(
    DenseGridInfo<Voxel> info,
    float3* d_positions,
    unsigned int numberOfPositions,
    Voxel* d_voxels)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPositions) return;

    float3 pos = d_positions[tid];

    int iX = static_cast<int>((pos.x - info.gridMin.x) / info.voxelSize);
    int iY = static_cast<int>((pos.y - info.gridMin.y) / info.voxelSize);
    int iZ = static_cast<int>((pos.z - info.gridMin.z) / info.voxelSize);

    if (0 > iX || info.dimensions.x <= iX ||
        0 > iY || info.dimensions.y <= iY ||
        0 > iZ || info.dimensions.z <= iZ)
        return;

    unsigned int flattenIndex = iZ * info.dimensions.y * info.dimensions.x + iY * info.dimensions.x + iX;

    if (atomicExch(&info.d_visitedFlags[flattenIndex], 1) == 0)
    {
        unsigned int index = atomicAdd(info.d_numberOfOccupiedVoxels, 1);
        info.d_occupiedVoxelIndices[index] = make_uint3(iX, iY, iZ);
        if (d_voxels)
        {
            info.d_voxels[flattenIndex] = d_voxels[tid];
        }
    }
}

template<typename Voxel>
struct DenseGrid {
    DenseGridInfo<Voxel> info;

    void Initialize(float3 gridMin, dim3 dimensions = dim3(400, 400, 400), float voxelSize = 0.1f)
    {
        if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0) return;

        info.gridMin = gridMin;
        info.dimensions = dimensions;
        info.voxelSize = voxelSize;
        info.numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

        cudaMalloc(&info.d_voxels, sizeof(Voxel) * info.numberOfVoxels);
        cudaMemset(info.d_voxels, 0, sizeof(Voxel) * info.numberOfVoxels);

        cudaMalloc(&info.d_visitedFlags, sizeof(unsigned int) * info.numberOfVoxels);
        cudaMemset(info.d_visitedFlags, 0, sizeof(unsigned int) * info.numberOfVoxels);
    }

    void Terminate()
    {
        if (info.d_voxels) cudaFree(info.d_voxels);
        if (info.d_numberOfOccupiedVoxels) cudaFree(info.d_numberOfOccupiedVoxels);
        if (info.d_occupiedVoxelIndices) cudaFree(info.d_occupiedVoxelIndices);
        if (info.d_visitedFlags) cudaFree(info.d_visitedFlags);

        info.d_voxels = nullptr;
        info.d_numberOfOccupiedVoxels = nullptr;
        info.d_occupiedVoxelIndices = nullptr;
        info.d_visitedFlags = nullptr;

        info.h_numberOfOccupiedVoxels = 0;
        info.h_occupiedCapacity = 0;
    }

    void Clear()
    {
        if (info.d_voxels) cudaMemset(info.d_voxels, 0, sizeof(Voxel) * info.numberOfVoxels);
        if (info.d_visitedFlags) cudaMemset(info.d_visitedFlags, 0, sizeof(unsigned int) * info.numberOfVoxels);

        if (info.d_numberOfOccupiedVoxels)
        {
            cudaFree(info.d_numberOfOccupiedVoxels);
            info.d_numberOfOccupiedVoxels = nullptr;
        }
        if (info.d_occupiedVoxelIndices)
        {
            cudaFree(info.d_occupiedVoxelIndices);
            info.d_occupiedVoxelIndices = nullptr;
        }

        info.h_numberOfOccupiedVoxels = 0;
        info.h_occupiedCapacity = 0;
    }

    void CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy)
    {
        if (numberOfVoxelsToOccupy == 0) return;

        if (!info.d_occupiedVoxelIndices)
        {
            cudaMalloc(&info.d_occupiedVoxelIndices, sizeof(uint3) * numberOfVoxelsToOccupy);
            cudaMalloc(&info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
            unsigned int zero = 0;
            cudaMemcpy(info.d_numberOfOccupiedVoxels, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
            info.h_occupiedCapacity = numberOfVoxelsToOccupy;
        }
        else
        {
            cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            unsigned int required = info.h_numberOfOccupiedVoxels + numberOfVoxelsToOccupy;
            if (required > info.h_occupiedCapacity)
            {
                required = min(required, info.numberOfVoxels);
                uint3* d_new = nullptr;
                cudaMalloc(&d_new, sizeof(uint3) * required);
                cudaMemcpy(d_new, info.d_occupiedVoxelIndices,
                    sizeof(uint3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToDevice);
                cudaFree(info.d_occupiedVoxelIndices);
                info.d_occupiedVoxelIndices = d_new;
                info.h_occupiedCapacity = required;
            }
        }

        printf("Dense Grid numberOfOccupiedVoxels capacity: %u\n", info.h_occupiedCapacity);
    }

    void Occupy(float3* d_positions, unsigned int numberOfPositions, Voxel* d_voxels = nullptr)
    {
        CheckOccupiedIndicesLength(numberOfPositions);
        LaunchKernel(Kernel_Occupy, numberOfPositions, info, d_positions, numberOfPositions, d_voxels);
        cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
    }

    void Serialize(const std::string& filename)
    {
        if (info.h_numberOfOccupiedVoxels == 0) return;

        uint3* h_occupiedVoxelIndices = new uint3[info.h_numberOfOccupiedVoxels];
        cudaMemcpy(h_occupiedVoxelIndices, info.d_occupiedVoxelIndices,
            sizeof(uint3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);

        PLYFormat ply;
        for (unsigned int i = 0; i < info.h_numberOfOccupiedVoxels; ++i)
        {
            uint3 idx = h_occupiedVoxelIndices[i];
            float x = (float)idx.x * info.voxelSize + info.gridMin.x + info.voxelSize * 0.5f;
            float y = (float)idx.y * info.voxelSize + info.gridMin.y + info.voxelSize * 0.5f;
            float z = (float)idx.z * info.voxelSize + info.gridMin.z + info.voxelSize * 0.5f;
            ply.AddPoint(x, y, z);
        }

        ply.Serialize(filename);
        delete[] h_occupiedVoxelIndices;
    }
};
