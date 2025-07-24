#pragma once

#include <cuda_common.cuh>
#include <HashMap.hpp>

struct HalfEdgeVertex
{
    unsigned int pointIndex = UINT32_MAX;
    unsigned int halfEdgeIndex = UINT32_MAX;
};

struct HalfEdge
{
    unsigned int vertexIndex = UINT32_MAX;
    unsigned int oppositeIndex = UINT32_MAX;
    unsigned int nextIndex = UINT32_MAX;
    unsigned int faceIndex = UINT32_MAX;
};

struct HalfEdgeFace
{
    unsigned int halfEdgeIndex = UINT32_MAX;
};

struct HostHalfEdgeMesh;
struct DeviceHalfEdgeMesh;

struct HostHalfEdgeMesh
{
    // Vertex/Face buffer (HostMesh-compatible)
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    // Half-edge structure
    HalfEdge* halfEdges = nullptr;
    HalfEdgeFace* halfEdgeFaces = nullptr;

    HostHalfEdgeMesh();
    HostHalfEdgeMesh(const HostHalfEdgeMesh& other);
    HostHalfEdgeMesh& operator=(const HostHalfEdgeMesh& other);
    HostHalfEdgeMesh(const DeviceHalfEdgeMesh& other);
    HostHalfEdgeMesh& operator=(const DeviceHalfEdgeMesh& other);
    ~HostHalfEdgeMesh();

    void Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CopyFromDevice(const DeviceHalfEdgeMesh& deviceMesh);
    void CopyToDevice(DeviceHalfEdgeMesh& deviceMesh) const;

    void BuildHalfEdges();

    bool SerializePLY(const string& filename, bool useAlpha = false);
    bool DeserializePLY(const string& filename);

    bool PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const;
};

struct DeviceHalfEdgeMesh
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    HalfEdge* halfEdges = nullptr;
    HalfEdgeFace* halfEdgeFaces = nullptr;

    unsigned int* vertexToHalfEdge = nullptr;

    DeviceHalfEdgeMesh();
    DeviceHalfEdgeMesh(const HostHalfEdgeMesh& other);
    DeviceHalfEdgeMesh& operator=(const HostHalfEdgeMesh& other);
    DeviceHalfEdgeMesh(const DeviceHalfEdgeMesh& other);
    DeviceHalfEdgeMesh& operator=(const DeviceHalfEdgeMesh& other);

    void Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CopyFromHost(const HostHalfEdgeMesh& hostMesh);
    void CopyToHost(HostHalfEdgeMesh& hostMesh) const;

    void BuildHalfEdges();

    bool PickFace(const float3& rayOrigin, const float3& rayDir,int& outHitIndex, float& outHitT) const;

    void LaplacianSmoothing(unsigned int iterations = 1, float lambda = 0.5f);
    void LaplacianSmoothingNRing(unsigned int iterations, float lambda, int nRing);

    __host__ __device__ static uint64_t PackEdge(unsigned int v0, unsigned int v1);
    __device__ static bool HashMapInsert(HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value);
    __device__ static bool HashMapFind(const HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue);

    __host__ __device__
        static bool RayTriangleIntersect(const float3& orig, const float3& dir,
            const float3& v0, const float3& v1, const float3& v2,
            float& t, float& u, float& v);
    __device__ static void atomicMinF(float* addr, float val, int* idx, int myIdx);
    __device__ static void atomicMinWithIndex(float* address, float val, int* idxAddress, int idx);
};

__global__ void Kernel_DeviceHalfEdgeMesh_BuildHalfEdges(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HalfEdgeFace* outFaces,
    HashMapInfo<uint64_t, unsigned int> info);

__global__ void Kernel_DeviceHalfEdgeMesh_LinkOpposites(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HashMapInfo<uint64_t, unsigned int> info);

__global__ void Kernel_LaplacianSmooth(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    float lambda);

__global__ void Kernel_LaplacianSmoothNRing(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    float lambda,
    int maxRing);

__global__ void Kernel_DeviceHalfEdgeMesh_PickFace(
    const float3 rayOrigin,
    const float3 rayDir,
    const float3* positions,
    const uint3* faces,
    unsigned int numberOfFaces,
    int* outHitIndex,
    float* outHitT);