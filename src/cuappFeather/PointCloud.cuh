#pragma once

#include <cuda_common.cuh>

struct HostPointCloud;
struct DevicePointCloud;

struct HostPointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    HostPointCloud();
    HostPointCloud(const HostPointCloud& other);
    HostPointCloud& operator=(const HostPointCloud& other);
    HostPointCloud(const DevicePointCloud& other);
    HostPointCloud& operator=(const DevicePointCloud& other);

    void Intialize(unsigned int numberOfPoints);
    void Terminate();

    void CompactValidPoints();
};

struct DevicePointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    DevicePointCloud();
    DevicePointCloud(const DevicePointCloud& other);
    DevicePointCloud& operator=(const DevicePointCloud& other);
    DevicePointCloud(const HostPointCloud& other);
    DevicePointCloud& operator=(const HostPointCloud& other);

    void Intialize(unsigned int numberOfPoints);
    void Terminate();

    void CompactValidPoints();
};
