#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>

#ifndef LaunchKernel
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); int NOT = 512; int NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)
#endif

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
