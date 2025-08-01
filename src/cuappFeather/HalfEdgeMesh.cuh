#pragma once

#include <cuda_common.cuh>
#include <HashMap.hpp>

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
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    HalfEdge* halfEdges = nullptr;
    HalfEdgeFace* halfEdgeFaces = nullptr;
    unsigned int* vertexToHalfEdge = nullptr;

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

    uint64_t PackEdge(unsigned int v0, unsigned int v1);
    
    void BuildHalfEdges();
    void BuildVertexToHalfEdgeMapping();

    bool SerializePLY(const string& filename, bool useAlpha = false);
    bool DeserializePLY(const string& filename);

    bool PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const;

    std::vector<unsigned int> GetOneRingVertices(unsigned int v) const;
    std::vector<unsigned int> GetVerticesInRadius(unsigned int startVertex, float radius);

    void ComputeFeatureWeights(std::vector<float>& featureWeights, float sharpAngleDeg);

    void BilateralNormalSmoothing(
        std::vector<float3>& outNormals,
        const std::vector<float>& featureWeights,
        float sigma_s,
        float sigma_n);

    void TangentPlaneProjection(
        std::vector<float3>& outPositions,
        const std::vector<float3>& smoothNormals,
        const std::vector<float>& featureWeights,
        float sigma_proj);

    void RobustSmooth(int iterations, float sigma_s, float sigma_n, float sigma_proj, float sharpAngleDeg);
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
    void BuildVertexToHalfEdgeMapping();

    bool PickFace(const float3& rayOrigin, const float3& rayDir,int& outHitIndex, float& outHitT) const;

    std::vector<unsigned int> GetOneRingVertices(unsigned int v, bool fixBorderVertices = false) const;
    std::vector<unsigned int> GetVerticesInRadius(unsigned int startVertex, float radius);

    void LaplacianSmoothing(unsigned int iterations = 1, float lambda = 0.5f, bool fixBorderVertices = false);

    __host__ __device__ static uint64_t PackEdge(unsigned int v0, unsigned int v1);
    __device__ static bool HashMapInsert(HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value);
    __device__ static bool HashMapFind(const HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue);

    __host__ __device__
        static bool RayTriangleIntersect(const float3& orig, const float3& dir,
            const float3& v0, const float3& v1, const float3& v2,
            float& t, float& u, float& v);
    __device__ static void atomicMinF(float* addr, float val, int* idx, int myIdx);
    __device__ static void atomicMinWithIndex(float* address, float val, int* idxAddress, int idx);

    __device__ static void GetOneRingVertices_Device(
            unsigned int v,
            const HalfEdge* halfEdges,
            const unsigned int* vertexToHalfEdge,
            unsigned int numberOfPoints,
            bool fixBorderVertices,
            unsigned int* outNeighbors,
            unsigned int& outCount,
            unsigned int maxNeighbors = 32);
};

__global__ void Kernel_DeviceHalfEdgeMesh_BuildHalfEdges(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HalfEdgeFace* halfEdgeFaces,
    HashMapInfo<uint64_t, unsigned int> info);

__global__ void Kernel_DeviceHalfEdgeMesh_LinkOpposites(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HashMapInfo<uint64_t, unsigned int> info);

__global__ void Kernel_DeviceHalfEdgeMesh_ValidateOppositeEdges(const HalfEdge* halfEdges, size_t numHalfEdges);

__global__ void Kernel_DeviceHalfEdgeMesh_BuildVertexToHalfEdgeMapping(
    const HalfEdge* halfEdges,
    unsigned int* vertexToHalfEdge,
    unsigned int numberOfHalfEdges);

__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    bool fixeborderVertices,
    const HalfEdge* halfEdges,
    unsigned int numberOfHalfEdges,
    unsigned int* vertexToHalfEdge,
    float lambda);

__global__ void Kernel_DeviceHalfEdgeMesh_PickFace(
    const float3 rayOrigin,
    const float3 rayDir,
    const float3* positions,
    const uint3* faces,
    unsigned int numberOfFaces,
    int* outHitIndex,
    float* outHitT);

__global__ void Kernel_DeviceHalfEdgeMesh_OneRing(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int numberOfPoints,
    unsigned int v, // 조사할 vertex index
    bool fixBorderVertices,
    unsigned int* outBuffer, // outBuffer[0:count-1]에 결과 저장
    unsigned int* outCount);

__global__ void Kernel_DeviceHalfEdgeMesh_GetVerticesInRadius(
    const float3* positions,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    const unsigned int* frontier,
    unsigned int frontierSize,
    unsigned int* visited,
    unsigned int* nextFrontier,
    unsigned int* nextFrontierSize,
    unsigned int* result,
    unsigned int* resultSize,
    unsigned int startVertex,
    float radius);