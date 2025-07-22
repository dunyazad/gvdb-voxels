#pragma once

#include <cuda_common.cuh>

struct HostMesh;
struct DeviceMesh;

struct HostMesh
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    HostMesh();
    HostMesh(const DeviceMesh& other);
    HostMesh& operator=(const DeviceMesh& other);

    void Intialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CompactValidPoints();
};

struct DeviceMesh
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    DeviceMesh();
    DeviceMesh(const HostMesh& other);
    DeviceMesh& operator=(const HostMesh& other);

    void Intialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CompactValidPoints();
};
