#pragma once

#include <cuda_common.cuh>

struct KDNode
{
    float3 point;
    unsigned int left = UINT32_MAX;
    unsigned int right = UINT32_MAX;
    unsigned int axis = 0;
};

__global__ void Kernel_Scequence(unsigned int* target, unsigned int numberOfElement)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfElement) return;

    target[threadid] = threadid;
}

struct KDTree
{
    void Intialize(unsigned int numberOfPoints)
    {

    }

    void Terminate()
    {

    }

    void Build(float* d_points, unsigned int numberOfPoints)
    {
    }

    void Serialize(const string& filename)
    {
    }
};
