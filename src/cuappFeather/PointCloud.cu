#pragma once

#include <PointCloud.cuh>

__global__ void Kernel_DevicePointCloudCompactValidPoints(
    float3* in_positions, float3* in_normals, float3* in_colors,
    float3* out_positions, float3* out_normals, float3* out_colors,
    unsigned int* valid_counter, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 p = in_positions[idx];
    bool valid = (p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX);

    if (valid)
    {
        unsigned int writeIdx = atomicAdd(valid_counter, 1);
        out_positions[writeIdx] = p;
        out_normals[writeIdx] = in_normals[idx];
        out_colors[writeIdx] = in_colors[idx];
    }
}