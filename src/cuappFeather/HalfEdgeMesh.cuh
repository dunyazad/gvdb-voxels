#pragma once

#include <cuda_common.cuh>
#include <HashMap.hpp>
#include <VoxelKey.hpp>

struct cuAABB
{
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
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

struct FaceNode
{
    unsigned int faceIndex = UINT32_MAX;
    unsigned int nextNodeIndex = UINT32_MAX;
};

struct FaceNodeHashMapEntry
{
    VoxelKey key = UINT64_MAX;
    unsigned int length = 0;
    unsigned int headIndex = UINT32_MAX;
    unsigned int tailIndex = UINT32_MAX;
    int lock = 0;
};

struct HostHalfEdgeMesh;
struct DeviceHalfEdgeMesh;

struct HostHalfEdgeMesh
{
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

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

    void RecalcAABB();

    uint64_t PackEdge(unsigned int v0, unsigned int v1);
    bool PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const;
    
    void BuildHalfEdges();
    void BuildVertexToHalfEdgeMapping();

    bool SerializePLY(const string& filename, bool useAlpha = false);
    bool DeserializePLY(const string& filename);

    std::vector<unsigned int> GetOneRingVertices(unsigned int v) const;
    
    bool CollapseEdge(unsigned int heIdx);
};

struct DeviceHalfEdgeMesh
{
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    uint3* faces = nullptr;
    unsigned int numberOfFaces = 0;

    HalfEdge* halfEdges = nullptr;
    HalfEdgeFace* halfEdgeFaces = nullptr;
    unsigned int* vertexToHalfEdge = nullptr;

    HashMap<uint64_t, FaceNodeHashMapEntry> faceNodeHashMap;
    FaceNode* faceNodes = nullptr;

    DeviceHalfEdgeMesh();
    DeviceHalfEdgeMesh(const HostHalfEdgeMesh& other);
    DeviceHalfEdgeMesh& operator=(const HostHalfEdgeMesh& other);
    DeviceHalfEdgeMesh(const DeviceHalfEdgeMesh& other);
    DeviceHalfEdgeMesh& operator=(const DeviceHalfEdgeMesh& other);

    void Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces);
    void Terminate();

    void CopyFromHost(const HostHalfEdgeMesh& hostMesh);
    void CopyToHost(HostHalfEdgeMesh& hostMesh) const;

    void RecalcAABB();

    void BuildHalfEdges();
    void RemoveIsolatedVertices();

    void BuildFaceNodeHashMap();
    vector<float3> GetFaceNodePositions();
    void DebugPrintFaceNodeHashMap();
    std::vector<unsigned int> FindUnlinkedFaceNodes();
    std::vector<unsigned int> FindNearestTriangleIndices(float3* d_positions, unsigned int numberOfInputPoints);

    bool PickFace(const float3& rayOrigin, const float3& rayDir,int& outHitIndex, float& outHitT) const;

    std::vector<unsigned int> GetOneRingVertices(unsigned int v, bool fixBorderVertices = false) const;
    std::vector<unsigned int> GetVerticesInRadius(unsigned int startVertex, float radius);

    void LaplacianSmoothing(unsigned int iterations = 1, float lambda = 0.5f, bool fixBorderVertices = false);
    void RadiusLaplacianSmoothing(float radius = 0.3f, unsigned int iterations = 1, float lambda = 0.5f);

    void GetAABBs(vector<cuAABB>& result, cuAABB& mMaabbs);
    vector<uint64_t> GetMortonCodes();

    __host__ __device__ static uint64_t PackEdge(unsigned int v0, unsigned int v1);
    __device__ static bool HashMapInsert(HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value);
    __device__ static bool HashMapFind(const HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue);

    __device__ static void GetOneRingVertices_Device(
            unsigned int v,
            const HalfEdge* halfEdges,
            const unsigned int* vertexToHalfEdge,
            unsigned int numberOfPoints,
            bool fixBorderVertices,
            unsigned int* outNeighbors,
            unsigned int& outCount,
            unsigned int maxNeighbors = 32);

    __device__ static void GetAllVerticesInRadius_Device(
        unsigned int vid,
        const float3* positions,
        unsigned int numberOfPoints,
        const HalfEdge* halfEdges,
        const unsigned int* vertexToHalfEdge,
        unsigned int* neighbors,        // [MAX_NEIGHBORS] output
        unsigned int& outCount,         // output count
        unsigned int maxNeighbors,
        float radius);
};

__global__ void Kernel_DeviceHalfEdgeMesh_RecalcAABB(float3* positions, float3* min, float3* max, unsigned int numberOfPoints);

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

__global__ void Kernel_DeviceHalfEdgeMesh_CompactVertices(
    unsigned int* vertexToHalfEdge,
    unsigned int* vertexIndexMapping,
    unsigned int* vertexIndexMappingIndex,
    const float3* oldPositions,
    const float3* oldNormals,
    const float3* oldColors,
    float3* newPositions,
    float3* newNormals,
    float3* newColors,
    unsigned int numberOfPoints);

__global__ void Kernel_DeviceHalfEdgeMesh_RemapVertexToHalfEdge(
    unsigned int* vertexToHalfEdge,
    unsigned int* newVertexToHalfEdge,
    unsigned int* vertexIndexMapping,
    unsigned int numberOfPoints);

__global__ void Kernel_DeviceHalfEdgeMesh_RemapVertexIndexOfFacesAndHalfEdges(
    uint3* faces, HalfEdge* halfEdges, unsigned int numberOfFaces, unsigned int* vertexIndexMapping);

__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    bool fixborderVertices,
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

__global__ void Kernel_GetAllVerticesInRadius(
    const float3* positions,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int* allNeighbors,   // [numberOfPoints * MAX_NEIGHBORS]
    unsigned int* allNeighborSizes, // [numberOfPoints]
    unsigned int maxNeighbors,
    float radius
);

__global__ void Kernel_RadiusLaplacianSmooth(
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int maxNeighbors,
    float radius,
    float lambda);
