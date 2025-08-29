#pragma once

#include <cuda_common.cuh>
#include <HashMap.hpp>
#include <VoxelKey.hpp>
#include <LBVH.cuh>

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

    DeviceLBVH bvh;
    MortonKey* mortonKeys = nullptr;

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
    void UpdateBVH();

    void BuildHalfEdges();
    void RemoveIsolatedVertices();

    void BuildFaceNodeHashMap();
    vector<float3> GetFaceNodePositions();
    void DebugPrintFaceNodeHashMap();
    std::vector<unsigned int> FindUnlinkedFaceNodes();
    std::vector<unsigned int> FindNearestTriangleIndices(float3* d_positions, unsigned int numberOfInputPoints);
    std::vector<unsigned int> FindNearestTriangleIndicesAndClosestPoints(
        float3* d_positions, unsigned int numberOfInputPoints, int offset, std::vector<float3>& outClosestPoints);

    bool PickFace(const float3& rayOrigin, const float3& rayDir,int& outHitIndex, float& outHitT) const;

    std::vector<unsigned int> GetOneRingVertices(unsigned int v, bool fixBorderVertices = false) const;
    std::vector<unsigned int> GetVerticesInRadius(unsigned int startVertex, float radius);

    std::vector<unsigned int> GetOneRingFaces(unsigned int f) const;
    std::vector<unsigned int> GetFacesInRadius(unsigned int f, float radius);

    void LaplacianSmoothing(unsigned int iterations = 1, float lambda = 0.5f, bool fixBorderVertices = false);
    void RadiusLaplacianSmoothing(float radius = 0.3f, unsigned int iterations = 1, float lambda = 0.5f);

    void GetAABBs(vector<cuAABB>& result, cuAABB& mMaabbs);
    vector<uint64_t> GetMortonCodes();

    vector<float> GetFaceCurvatures();

	void FindDegenerateFaces(vector<unsigned int>& outFaceIndices) const;

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
        unsigned int* neighbors,
        unsigned int& outCount,
        unsigned int maxNeighbors,
        float radius);

    __device__ static void GetOneRingFaces_Device(
        unsigned int f,
        const HalfEdge* halfEdges,
        unsigned int numberOfFaces,
        unsigned int* outFaces,
        unsigned int& outCount);

    __device__ static float3 calcFaceNormal(const float3* positions, uint3 face)
    {
        float3 v0 = positions[face.x];
        float3 v1 = positions[face.y];
        float3 v2 = positions[face.z];
        return normalize(cross(v1 - v0, v2 - v0));
    }
};
