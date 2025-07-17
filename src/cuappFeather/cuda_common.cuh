#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <cuda_vector_math.cuh>
#include <marching_cubes_constants.cuh>

#include <Serialization.hpp>

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

struct HostPointCloud;
struct DevicePointCloud;

struct HostPointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    void Intialize(unsigned int numberOfPoints)
    {
        if (numberOfPoints == 0) return;

        this->numberOfPoints = numberOfPoints;
        positions = new float3[numberOfPoints];
        normals = new float3[numberOfPoints];
        colors = new float3[numberOfPoints];
    }

    void Terminate()
    {
        if (numberOfPoints > 0)
        {
            delete[] positions;
            delete[] normals;
            delete[] colors;

            positions = nullptr;
            normals = nullptr;
            colors = nullptr;
            numberOfPoints = 0;
        }
    }

    HostPointCloud();
    HostPointCloud(const DevicePointCloud& other);
    HostPointCloud& operator=(const DevicePointCloud& other);
};

struct DevicePointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    void Intialize(unsigned int numberOfPoints)
    {
        if (numberOfPoints == 0) return;

        this->numberOfPoints = numberOfPoints;
        cudaMalloc(&positions, sizeof(float3) * numberOfPoints);
        cudaMalloc(&normals, sizeof(float3) * numberOfPoints);
        cudaMalloc(&colors, sizeof(float3) * numberOfPoints);
        cudaDeviceSynchronize();
    }

    void Terminate()
    {
        if (numberOfPoints > 0)
        {
            cudaFree(positions);
            cudaFree(normals);
            cudaFree(colors);

            positions = nullptr;
            normals = nullptr;
            colors = nullptr;
            numberOfPoints = 0;

            cudaDeviceSynchronize();
        }
    }

    DevicePointCloud();
    DevicePointCloud(const HostPointCloud& other);
    DevicePointCloud& operator=(const HostPointCloud& other);
};

inline HostPointCloud::HostPointCloud() {}

inline HostPointCloud::HostPointCloud(const DevicePointCloud& other)
{
    operator =(other);
}

inline HostPointCloud& HostPointCloud::operator=(const DevicePointCloud& other)
{
    Terminate();
    Intialize(other.numberOfPoints);

    cudaMemcpy(positions, other.positions, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);
    cudaMemcpy(normals, other.normals, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);
    cudaMemcpy(colors, other.colors, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);

    return *this;
}

inline DevicePointCloud::DevicePointCloud() {}

inline DevicePointCloud::DevicePointCloud(const HostPointCloud& other)
{
    operator =(other);
}

inline DevicePointCloud& DevicePointCloud::operator=(const HostPointCloud& other)
{
    Terminate();
    Intialize(other.numberOfPoints);

    cudaMemcpy(positions, other.positions, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(normals, other.normals, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(colors, other.colors, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);

    return *this;
}
