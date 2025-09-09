#pragma once

#include <cuda_common.cuh>
#include <SimpleHashMap.hpp>

struct ThrustHalfEdge
{
    unsigned int vertexIndex = UINT32_MAX;
    unsigned int oppositeIndex = UINT32_MAX;
    unsigned int nextIndex = UINT32_MAX;
    unsigned int faceIndex = UINT32_MAX;
};

struct ThrustHalfEdgeFace
{
    unsigned int halfEdgeIndex = UINT32_MAX;
};

struct ThrustHostHalfEdgeMesh;
struct ThrustDeviceHalfEdgeMesh;

struct ThrustHostHalfEdgeMesh
{
    unsigned int numberOfPoints = 0;
    thrust::host_vector<float3> positions;
    thrust::host_vector<float3> normals;
    thrust::host_vector<float3> colors;

    unsigned int numberOfFaces = 0;
    thrust::host_vector<uint3> faces;

    thrust::host_vector<ThrustHalfEdge> halfEdges;
    thrust::host_vector<ThrustHalfEdgeFace> halfEdgeFaces;
    thrust::host_vector<unsigned int> vertexToHalfEdge;

    ThrustHostHalfEdgeMesh();
    ThrustHostHalfEdgeMesh(const ThrustHostHalfEdgeMesh& other);
    ThrustHostHalfEdgeMesh& operator=(const ThrustHostHalfEdgeMesh& other);
    ThrustHostHalfEdgeMesh(const ThrustDeviceHalfEdgeMesh& other);
    ThrustHostHalfEdgeMesh& operator=(const ThrustDeviceHalfEdgeMesh& other);

    void Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CopyFromDevice(const ThrustDeviceHalfEdgeMesh& deviceMesh);
    void CopyToDevice(ThrustDeviceHalfEdgeMesh& deviceMesh) const;

    uint64_t PackEdge(unsigned int v0, unsigned int v1);
    bool PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const;

    void BuildHalfEdges();

    bool SerializePLY(const string& filename, bool useAlpha = false);
    bool DeserializePLY(const string& filename);

    std::vector<unsigned int> GetOneRingVertices(unsigned int v) const;
};

struct ThrustDeviceHalfEdgeMesh
{
    unsigned int numberOfPoints = 0;
    thrust::device_vector<float3> positions;
    thrust::device_vector<float3> normals;
    thrust::device_vector<float3> colors;

    unsigned int numberOfFaces = 0;
    thrust::device_vector<uint3> faces;

    thrust::device_vector<ThrustHalfEdge> halfEdges;
    thrust::device_vector<ThrustHalfEdgeFace> halfEdgeFaces;
    thrust::device_vector<unsigned int> vertexToHalfEdge;

    ThrustDeviceHalfEdgeMesh();
    ThrustDeviceHalfEdgeMesh(const ThrustHostHalfEdgeMesh& other);
    ThrustDeviceHalfEdgeMesh& operator=(const ThrustHostHalfEdgeMesh& other);
    ThrustDeviceHalfEdgeMesh(const ThrustDeviceHalfEdgeMesh& other);
    ThrustDeviceHalfEdgeMesh& operator=(const ThrustDeviceHalfEdgeMesh& other);

    void Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CopyFromHost(const ThrustHostHalfEdgeMesh& hostMesh);
    void CopyToHost(ThrustHostHalfEdgeMesh& hostMesh) const;

    __device__ static uint64_t PackEdge(unsigned int v0, unsigned int v1);
    bool PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const;

    void BuildHalfEdges();

    //std::vector<unsigned int> GetOneRingVertices(unsigned int v);

    __device__ static bool SimpleHashMapInsert(SimpleHashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value);
    __device__ static bool SimpleHashMapFind(const SimpleHashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue);
};
