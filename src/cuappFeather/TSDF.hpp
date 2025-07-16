#pragma once

#include <cuda_common.cuh>

struct TSDFVoxel
{
    float tsdfValue;
    float3 normal;
    float3 color;
    unsigned int count;
};

struct TSDFInfo
{
    TSDFVoxel* d_voxels = nullptr;

    dim3 dimensions = dim3(400, 400, 400);
    unsigned int numberOfVoxels = 400 * 400 * 400;
    float voxelSize = 0.1f;
    float3 gridMin = make_float3(-200, -200, -200);

    unsigned int* d_numberOfOccupiedVoxels = nullptr;
    uint3* d_occupiedVoxelIndices = nullptr;
    unsigned int h_numberOfOccupiedVoxels = 0;
    unsigned int h_occupiedCapacity = 0;
};

__global__ void Kernel_OccupyTSDF(
    TSDFInfo info,
    float3* d_positions,
    float3* d_normals,
    float3* d_colors,
    unsigned int numberOfPositions);

__global__ void Kernel_SerializeTSDF(
    TSDFInfo info,
    float3* d_positions,
    float3* d_normals,
    float3* d_colors);

struct TSDF
{
    TSDFInfo info;

    void Initialize(float3 gridMin, dim3 dimensions = dim3(400, 400, 400), float voxelSize = 0.1f)
    {
        if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0) return;

        info.gridMin = gridMin;
        info.dimensions = dimensions;
        info.voxelSize = voxelSize;
        info.numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

        size_t allocated = 0;

        cudaMalloc(&info.d_voxels, sizeof(TSDFVoxel) * info.numberOfVoxels);
        cudaMemset(info.d_voxels, 0, sizeof(TSDFVoxel) * info.numberOfVoxels);

        allocated += sizeof(TSDFVoxel) * info.numberOfVoxels;

        printf("[TSDF] Allocated : %f GB\n", (float)allocated / 1000000000.0f);
    }

    void Terminate()
    {
        if (info.d_voxels) cudaFree(info.d_voxels);
        if (info.d_numberOfOccupiedVoxels) cudaFree(info.d_numberOfOccupiedVoxels);
        if (info.d_occupiedVoxelIndices) cudaFree(info.d_occupiedVoxelIndices);

        info.d_voxels = nullptr;
        info.d_numberOfOccupiedVoxels = nullptr;
        info.d_occupiedVoxelIndices = nullptr;

        info.h_numberOfOccupiedVoxels = 0;
        info.h_occupiedCapacity = 0;
    }

    void Clear()
    {
        if (info.d_voxels) cudaMemset(info.d_voxels, 0, sizeof(TSDFVoxel) * info.numberOfVoxels);

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

    void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPositions)
    {
        CheckOccupiedIndicesLength(numberOfPositions);
        LaunchKernel(Kernel_OccupyTSDF, numberOfPositions, info, d_positions, d_normals, d_colors, numberOfPositions);
        cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
    }

    void Serialize(const std::string& filename)
    {
        if (info.h_numberOfOccupiedVoxels == 0) return;

        float3* d_positions = nullptr;
        cudaMalloc(&d_positions, sizeof(float3) * info.h_numberOfOccupiedVoxels);
        float3* d_normals = nullptr;
        cudaMalloc(&d_normals, sizeof(float3) * info.h_numberOfOccupiedVoxels);
        float3* d_colors = nullptr;
        cudaMalloc(&d_colors, sizeof(float3) * info.h_numberOfOccupiedVoxels);

        LaunchKernel(Kernel_SerializeTSDF, info.h_numberOfOccupiedVoxels, info, d_positions, d_normals, d_colors);

        float3* h_positions = new float3[info.h_numberOfOccupiedVoxels];
        float3* h_normals = new float3[info.h_numberOfOccupiedVoxels];
        float3* h_colors = new float3[info.h_numberOfOccupiedVoxels];

        cudaMemcpy(h_positions, d_positions, sizeof(float3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals, d_normals, sizeof(float3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colors, d_colors, sizeof(float3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(d_positions);
        cudaFree(d_normals);
        cudaFree(d_colors);

        PLYFormat ply;
        for (unsigned int i = 0; i < info.h_numberOfOccupiedVoxels; ++i)
        {
            auto& p = h_positions[i];
            auto& n = h_normals[i];
            auto& c = h_colors[i];
            
            ply.AddPoint(p.x, p.y, p.z);
            ply.AddNormal(n.x, n.y, n.z);
            ply.AddColor(c.x, c.y, c.z);
        }

        ply.Serialize(filename);

        delete[] h_positions;
        delete[] h_normals;
        delete[] h_colors;
    }
};

__global__ void Kernel_OccupyTSDF(
    TSDFInfo info,
    float3* d_positions,
    float3* d_normals,
    float3* d_colors,
    unsigned int numberOfPositions)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPositions) return;

    float3 position = d_positions[tid];

    int iX = static_cast<int>((position.x - info.gridMin.x) / info.voxelSize);
    int iY = static_cast<int>((position.y - info.gridMin.y) / info.voxelSize);
    int iZ = static_cast<int>((position.z - info.gridMin.z) / info.voxelSize);

    if (0 > iX || info.dimensions.x <= iX ||
        0 > iY || info.dimensions.y <= iY ||
        0 > iZ || info.dimensions.z <= iZ)
        return;

    float3 normal = d_normals[tid];
    float3 color = d_colors[tid];

    unsigned int flattenIndex = iZ * info.dimensions.y * info.dimensions.x + iY * info.dimensions.x + iX;
    auto& voxel = info.d_voxels[flattenIndex];

    unsigned int count = atomicAdd(&voxel.count, 1);
    if (count == 0)
    {
        unsigned int index = atomicAdd(info.d_numberOfOccupiedVoxels, 1);
        if (index < info.h_occupiedCapacity)
            info.d_occupiedVoxelIndices[index] = make_uint3(iX, iY, iZ);
    }

    // 모든 thread가 atomic하게 누적
    atomicAdd(&voxel.normal.x, normal.x);
    atomicAdd(&voxel.normal.y, normal.y);
    atomicAdd(&voxel.normal.z, normal.z);
    atomicAdd(&voxel.color.x, color.x);
    atomicAdd(&voxel.color.y, color.y);
    atomicAdd(&voxel.color.z, color.z);
}

__global__ void Kernel_SerializeTSDF(
    TSDFInfo info,
    float3* d_positions,
    float3* d_normals,
    float3* d_colors)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *info.d_numberOfOccupiedVoxels) return;

    auto voxelIndex = info.d_occupiedVoxelIndices[tid];

    float x = info.gridMin.x + (float)voxelIndex.x * info.voxelSize;
    float y = info.gridMin.y + (float)voxelIndex.y * info.voxelSize;
    float z = info.gridMin.z + (float)voxelIndex.z * info.voxelSize;

    unsigned int flattenIndex = voxelIndex.z * info.dimensions.y * info.dimensions.x + voxelIndex.y * info.dimensions.x + voxelIndex.x;
    auto& voxel = info.d_voxels[flattenIndex];

    float3 normal = voxel.normal;
    float3 color = voxel.color;

    normal.x /= (float)voxel.count;
    normal.y /= (float)voxel.count;
    normal.z /= (float)voxel.count;

    color.x /= (float)voxel.count;
    color.y /= (float)voxel.count;
    color.z /= (float)voxel.count;

    d_positions[tid].x = x;
    d_positions[tid].y = y;
    d_positions[tid].z = z;

    d_normals[tid].x = normal.x;
    d_normals[tid].y = normal.y;
    d_normals[tid].z = normal.z;

    d_colors[tid].x = color.x;
    d_colors[tid].y = color.y;
    d_colors[tid].z = color.z;
}
