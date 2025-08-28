#include <HalfEdgeMesh.cuh>
#include <Mesh.cuh>

#include <set>

#define MAX_FRONTIER (1 << 16)
#define MAX_RESULT   (1 << 20)
#define MAX_NEIGHBORS 64

#pragma region HostHalfEdgeMesh
HostHalfEdgeMesh::HostHalfEdgeMesh() {}

HostHalfEdgeMesh::HostHalfEdgeMesh(const HostHalfEdgeMesh& other)
{
    *this = other;
}

HostHalfEdgeMesh& HostHalfEdgeMesh::operator=(const HostHalfEdgeMesh& other)
{
    if (this == &other) return *this;
    Terminate();
    Initialize(other.numberOfPoints, other.numberOfFaces);
    memcpy(positions, other.positions, sizeof(float3) * numberOfPoints);
    memcpy(normals, other.normals, sizeof(float3) * numberOfPoints);
    memcpy(colors, other.colors, sizeof(float3) * numberOfPoints);
    memcpy(faces, other.faces, sizeof(uint3) * numberOfFaces);
    memcpy(halfEdges, other.halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
    memcpy(halfEdgeFaces, other.halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
    memcpy(vertexToHalfEdge, other.vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);
    min = other.min;
    max = other.max;
    return *this;
}

HostHalfEdgeMesh::HostHalfEdgeMesh(const DeviceHalfEdgeMesh& other)
{
    *this = other;
}

HostHalfEdgeMesh& HostHalfEdgeMesh::operator=(const DeviceHalfEdgeMesh& other)
{
    CopyFromDevice(other);
    return *this;
}

HostHalfEdgeMesh::~HostHalfEdgeMesh()
{
    Terminate();
}

void HostHalfEdgeMesh::Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
    this->numberOfPoints = numberOfPoints;
    this->numberOfFaces = numberOfFaces;
    if (numberOfPoints > 0)
    {
        positions = new float3[numberOfPoints];
        normals = new float3[numberOfPoints];
        colors = new float3[numberOfPoints];
    }
    else
    {
        positions = nullptr;
        normals = nullptr;
        colors = nullptr;
    }
    if (numberOfFaces > 0)
    {
        faces = new uint3[numberOfFaces];
        halfEdges = new HalfEdge[numberOfFaces * 3];
        halfEdgeFaces = new HalfEdgeFace[numberOfFaces];
        vertexToHalfEdge = new unsigned int[numberOfPoints];
    }
    else
    {
        faces = nullptr;
        halfEdges = nullptr;
        halfEdgeFaces = nullptr;
        vertexToHalfEdge = nullptr;
    }

    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
}

void HostHalfEdgeMesh::Terminate()
{
    if (positions) delete[] positions;
    if (normals) delete[] normals;
    if (colors) delete[] colors;
    if (faces) delete[] faces;
    if (halfEdges) delete[] halfEdges;
    if (halfEdgeFaces) delete[] halfEdgeFaces;
    if (vertexToHalfEdge) delete[] vertexToHalfEdge;

    positions = nullptr;
    normals = nullptr;
    colors = nullptr;
    faces = nullptr;
    halfEdges = nullptr;
    halfEdgeFaces = nullptr;
    vertexToHalfEdge = nullptr;
    numberOfPoints = 0;
    numberOfFaces = 0;

    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
}

void HostHalfEdgeMesh::CopyFromDevice(const DeviceHalfEdgeMesh& deviceMesh)
{
    Terminate();
    Initialize(deviceMesh.numberOfPoints, deviceMesh.numberOfFaces);

    CUDA_COPY_D2H(positions, deviceMesh.positions, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(normals, deviceMesh.normals, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(colors, deviceMesh.colors, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(faces, deviceMesh.faces, sizeof(uint3) * numberOfFaces);
    CUDA_COPY_D2H(halfEdges, deviceMesh.halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
    CUDA_COPY_D2H(halfEdgeFaces, deviceMesh.halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
    CUDA_COPY_D2H(vertexToHalfEdge, deviceMesh.vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);

    CUDA_SYNC();

    min = deviceMesh.min;
    max = deviceMesh.max;
}

void HostHalfEdgeMesh::CopyToDevice(DeviceHalfEdgeMesh& deviceMesh) const
{
    deviceMesh.Terminate();
    deviceMesh.Initialize(numberOfPoints, numberOfFaces);

    CUDA_COPY_H2D(deviceMesh.positions, positions, sizeof(float3) * numberOfPoints);
    CUDA_COPY_H2D(deviceMesh.normals, normals, sizeof(float3) * numberOfPoints);
    CUDA_COPY_H2D(deviceMesh.colors, colors, sizeof(float3) * numberOfPoints);
    CUDA_COPY_H2D(deviceMesh.faces, faces, sizeof(uint3) * numberOfFaces);
    CUDA_COPY_H2D(deviceMesh.halfEdges, halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
    CUDA_COPY_H2D(deviceMesh.halfEdgeFaces, halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
    CUDA_COPY_H2D(deviceMesh.vertexToHalfEdge, vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);

    CUDA_SYNC();

    deviceMesh.min = min;
    deviceMesh.max = max;
}

void HostHalfEdgeMesh::RecalcAABB()
{
    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (size_t i = 0; i < numberOfPoints; i++)
    {
        auto& p = positions[i];

        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);

        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }
}

uint64_t HostHalfEdgeMesh::PackEdge(unsigned int v0, unsigned int v1)
{
    return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
}

bool HostHalfEdgeMesh::PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const
{
    float minT = std::numeric_limits<float>::max();
    int minIdx = -1;

    for (unsigned int i = 0; i < numberOfFaces; ++i)
    {
        const uint3& tri = faces[i];
        const float3& v0 = positions[tri.x];
        const float3& v1 = positions[tri.y];
        const float3& v2 = positions[tri.z];

        float t, u, v;
        if (RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
        {
            if (t < minT)
            {
                minT = t;
                minIdx = static_cast<int>(i);
            }
        }
    }

    outHitIndex = minIdx;
    outHitT = minT;
    return (minIdx >= 0);
}

void HostHalfEdgeMesh::BuildHalfEdges()
{
    if (numberOfFaces == 0 || faces == nullptr)
        return;

    std::unordered_map<uint64_t, unsigned int> edgeMap;

    for (unsigned int i = 0; i < numberOfFaces; ++i)
    {
        auto& face = faces[i];

        auto& e0 = halfEdges[i * 3 + 0];
        auto& e1 = halfEdges[i * 3 + 1];
        auto& e2 = halfEdges[i * 3 + 2];

        auto& f = halfEdgeFaces[i];

        e0.faceIndex = i; e1.faceIndex = i; e2.faceIndex = i;
        e0.nextIndex = i * 3 + 1; e1.nextIndex = i * 3 + 2; e2.nextIndex = i * 3 + 0;

        e0.vertexIndex = face.x;
        e1.vertexIndex = face.y;
        e2.vertexIndex = face.z;
        e0.oppositeIndex = e1.oppositeIndex = e2.oppositeIndex = UINT32_MAX;
        f.halfEdgeIndex = i * 3;

        edgeMap[PackEdge(face.x, face.y)] = i * 3 + 0;
        edgeMap[PackEdge(face.y, face.z)] = i * 3 + 1;
        edgeMap[PackEdge(face.z, face.x)] = i * 3 + 2;
    }

    for (unsigned int i = 0; i < numberOfFaces; ++i)
    {
        auto& face = faces[i];
        if (auto it = edgeMap.find(PackEdge(face.y, face.x)); it != edgeMap.end())
            halfEdges[i * 3 + 0].oppositeIndex = it->second;
        if (auto it = edgeMap.find(PackEdge(face.z, face.y)); it != edgeMap.end())
            halfEdges[i * 3 + 1].oppositeIndex = it->second;
        if (auto it = edgeMap.find(PackEdge(face.x, face.z)); it != edgeMap.end())
            halfEdges[i * 3 + 2].oppositeIndex = it->second;
    }

    BuildVertexToHalfEdgeMapping();
}

void HostHalfEdgeMesh::BuildVertexToHalfEdgeMapping()
{
    if (vertexToHalfEdge) delete[] vertexToHalfEdge;
    vertexToHalfEdge = new unsigned int[numberOfPoints];
    for (unsigned int i = 0; i < numberOfPoints; ++i)
        vertexToHalfEdge[i] = UINT32_MAX;

    for (unsigned int i = 0; i < numberOfFaces * 3; ++i)
    {
        unsigned int v = halfEdges[i].vertexIndex;
        if (v < numberOfPoints && vertexToHalfEdge[v] == UINT32_MAX)
            vertexToHalfEdge[v] = i;
    }
}

bool HostHalfEdgeMesh::SerializePLY(const std::string& filename, bool useAlpha)
{
    PLYFormat ply;

    for (unsigned int i = 0; i < numberOfPoints; ++i)
    {
        ply.AddPoint(positions[i].x, positions[i].y, positions[i].z);
        if (normals) ply.AddNormal(normals[i].x, normals[i].y, normals[i].z);
        if (colors)
        {
            if (useAlpha)
                ply.AddColor(colors[i].x, colors[i].y, colors[i].z, 1.0f);
            else
                ply.AddColor(colors[i].x, colors[i].y, colors[i].z);
        }
    }

    for (unsigned int f = 0; f < numberOfFaces; ++f)
    {
        unsigned int he0 = halfEdgeFaces[f].halfEdgeIndex;
        if (he0 == UINT32_MAX) continue; // skip if uninitialized
        const HalfEdge& e0 = halfEdges[he0];
        const HalfEdge& e1 = halfEdges[e0.nextIndex];
        const HalfEdge& e2 = halfEdges[e1.nextIndex];
        // e0.faceIndex == e1.faceIndex == e2.faceIndex == f

        ply.AddFace(e0.vertexIndex, e1.vertexIndex, e2.vertexIndex);
    }

    return ply.Serialize(filename);
}

bool HostHalfEdgeMesh::DeserializePLY(const std::string& filename)
{
    PLYFormat ply;
    if (!ply.Deserialize(filename))
        return false;

    unsigned int n = static_cast<unsigned int>(ply.GetPoints().size() / 3);
    unsigned int nf = static_cast<unsigned int>(ply.GetTriangleIndices().size() / 3);
    Terminate();
    Initialize(n, nf);

    const auto& pts = ply.GetPoints();
    for (unsigned int i = 0; i < n; ++i)
    {
        positions[i].x = pts[i * 3 + 0];
        positions[i].y = pts[i * 3 + 1];
        positions[i].z = pts[i * 3 + 2];
    }

    if (normals && !ply.GetNormals().empty())
    {
        const auto& ns = ply.GetNormals();
        for (unsigned int i = 0; i < n; ++i)
        {
            normals[i].x = ns[i * 3 + 0];
            normals[i].y = ns[i * 3 + 1];
            normals[i].z = ns[i * 3 + 2];
        }
    }

    if (colors && !ply.GetColors().empty())
    {
        const auto& cs = ply.GetColors();
        size_t stride = cs.size() / n == 4 ? 4 : 3;
        for (unsigned int i = 0; i < n; ++i)
        {
            colors[i].x = cs[i * stride + 0];
            colors[i].y = cs[i * stride + 1];
            colors[i].z = cs[i * stride + 2];
        }
    }

    const auto& idx = ply.GetTriangleIndices();
    for (unsigned int i = 0; i < nf; ++i)
    {
        faces[i].x = idx[i * 3 + 0];
        faces[i].y = idx[i * 3 + 1];
        faces[i].z = idx[i * 3 + 2];
    }

    BuildHalfEdges();

    auto [mx, my, mz] = ply.GetAABBMin();
    min.x = mx; min.y = my; min.z = mz;
    auto [Mx, My, Mz] = ply.GetAABBMax();
    max.x = Mx; max.y = My; max.z = Mz;

    return true;
}

std::vector<unsigned int> HostHalfEdgeMesh::GetOneRingVertices(unsigned int v) const
{
    std::vector<unsigned int> neighbors;
    if (!vertexToHalfEdge || v >= numberOfPoints) return neighbors;

    unsigned int ishe = vertexToHalfEdge[v];
    if (ishe == UINT32_MAX) return neighbors;

    auto ihe = ishe;
    auto lihe = ihe;
    bool borderFound = false;

    // cw
    do
    {
        auto he = halfEdges[ihe];
        if (UINT32_MAX == he.nextIndex) break;

        auto ioe = he.oppositeIndex;
        if (UINT32_MAX == ioe)
        {
            lihe = ihe;
            borderFound = true;
            break;
        }
        auto oe = halfEdges[he.oppositeIndex];
        if (UINT32_MAX == oe.vertexIndex)
        {
            printf("[Fatal] Oppsite halfedge has no vertex index.\n");
            break;
        }
        neighbors.push_back(oe.vertexIndex);

        if (UINT32_MAX == oe.nextIndex)
        {
            printf("[Fatal] Oppsite halfedge has no next halfedge index.\n");
            break;
        }

        ihe = oe.nextIndex;
    } while (ihe != ishe && UINT32_MAX != ihe);

    // ccw
    if (borderFound)
    {
        neighbors.clear();
        ishe = lihe;
        ihe = lihe;
        do
        {
            auto he = halfEdges[ihe];
            if (UINT32_MAX == he.nextIndex)
            {
                printf("[Fatal] Halfedge has no next halfedge index.\n");
                break;
            }

            auto ne = halfEdges[he.nextIndex];
            if (UINT32_MAX == ne.nextIndex)
            {
                printf("[Fatal] Next halfedge has no next halfedge index.\n");
                break;
            }
            if (ishe == ihe)
            {
                neighbors.push_back(ne.vertexIndex);
            }

            auto pe = halfEdges[ne.nextIndex];
            neighbors.push_back(pe.vertexIndex);

            ihe = pe.oppositeIndex;
        } while (ihe != ishe && UINT32_MAX != ihe);
    }
    return neighbors;
}

bool HostHalfEdgeMesh::CollapseEdge(unsigned int heIdx)
{
    if (heIdx >= numberOfFaces * 3)
        return false;

    HalfEdge& he = halfEdges[heIdx];
    unsigned int v0 = he.vertexIndex;
    unsigned int oppIdx = he.oppositeIndex;
    if (oppIdx == UINT32_MAX)
        return false;

    HalfEdge& oppHe = halfEdges[oppIdx];
    unsigned int v1 = oppHe.vertexIndex;
    if (v0 == v1)
        return false;

    for (unsigned int i = 0; i < numberOfFaces * 3; ++i)
    {
        if (halfEdges[i].vertexIndex == v1)
            halfEdges[i].vertexIndex = v0;
    }
    for (unsigned int fi = 0; fi < numberOfFaces; ++fi)
    {
        uint3& f = faces[fi];
        if (f.x == v1) f.x = v0;
        if (f.y == v1) f.y = v0;
        if (f.z == v1) f.z = v0;
    }
    if (vertexToHalfEdge)
    {
        for (unsigned int i = 0; i < numberOfPoints; ++i)
        {
            if (vertexToHalfEdge[i] != UINT32_MAX)
            {
                unsigned int hei = vertexToHalfEdge[i];
                if (hei < numberOfFaces * 3 && halfEdges[hei].vertexIndex == v1)
                    vertexToHalfEdge[i] = heIdx;
            }
        }
    }

    positions[v0] = (positions[v0] + positions[v1]) * 0.5f;
    if (normals)
        normals[v0] = normalize(normals[v0] + normals[v1]);
    if (colors)
        colors[v0] = (colors[v0] + colors[v1]) * 0.5f;

    std::vector<uint3> newFaces;
    for (unsigned int fi = 0; fi < numberOfFaces; ++fi)
    {
        uint3 f = faces[fi];
        if (f.x != f.y && f.y != f.z && f.z != f.x)
            newFaces.push_back(f);
    }
    unsigned int newFaceCount = static_cast<unsigned int>(newFaces.size());
    memcpy(faces, newFaces.data(), sizeof(uint3) * newFaceCount);
    numberOfFaces = newFaceCount;

    BuildHalfEdges();

    return true;
}
#pragma endregion

#pragma region DeviceHalfEdgeMesh Kernels Forward Declarations
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

__global__ void Kernel_DeviceHalfEdgeMesh_BuildFaceNodeHashMap(
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> info,
    float3* positions,
    uint3* faces,
    FaceNode* faceNodes,
    float voxelSize,
    unsigned int numberOfFaces,
    unsigned int* d_numDropped);

__global__ void Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap(
    const float3* points,
    unsigned int numPoints,
    const float3* positions,
    const uint3* faces,
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> faceNodeHashMap,
    const FaceNode* faceNodes,
    float voxelSize,
    unsigned int* outIndices);

__global__ void Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap_ClosestPoint(
    const float3* points,
    unsigned int numPoints,
    const float3* positions,
    const uint3* faces,
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> faceNodeHashMap,
    const FaceNode* faceNodes,
    float voxelSize,
    int offset,
    unsigned int* outIndices,
    float3* outClosestPoints);

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
    unsigned int v,
    bool fixBorderVertices,
    unsigned int* outBuffer,
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

__global__ void Kernel_DeviceHalfEdgeMesh_OneRingFaces(
    unsigned int f,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    unsigned int* outFaces,
    unsigned int* outCount);

__global__ void Kernel_DeviceHalfEdgeMesh_GetFacesInRadius(
    const float3* positions,
    const uint3* faces,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    const unsigned int* frontier,
    unsigned int frontierSize,
    unsigned int* visited,
    unsigned int* nextFrontier,
    unsigned int* nextFrontierSize,
    unsigned int* result,
    unsigned int* resultSize,
    unsigned int startFace,
    float radius);

__global__ void Kernel_DeviceHalfEdgeMesh_RadiusLaplacianSmooth(
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int maxNeighbors,
    float radius,
    float lambda);

__global__ void Kernel_DeviceHalfEdgeMesh_GetAABB(
    float3* positions,
    uint3* faces,
    cuAABB* aabbs,
    cuAABB* mMaabbs,
    unsigned int numberOfPoints,
    unsigned int numberOfFaces);

__global__ void Kernel_DeviceHalfEdgeMesh_GetMortonCodes(
    const float3* positions,
    const uint3* faces,
    uint64_t* mortonCodes,
    unsigned int numberOfPoints,
    unsigned int numberOfFaces,
    float3 min_corner,
    float voxel_size);

__global__ void Kernel_DeviceHalfEdgeMesh_RecalcAABB(
    float3* positions,
    float3* min,
    float3* max,
    unsigned int numberOfPoints);

__global__ void Kernel_DeviceHalfEdgeMesh_GetFaceCurvatures(
    const float3* positions,
    const uint3* faces,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    float* outCurvatures);
#pragma endregion

#pragma region DeviceHalfEdgeMesh
DeviceHalfEdgeMesh::DeviceHalfEdgeMesh() {}

DeviceHalfEdgeMesh::DeviceHalfEdgeMesh(const HostHalfEdgeMesh& other)
{
    *this = other;
}

DeviceHalfEdgeMesh& DeviceHalfEdgeMesh::operator=(const HostHalfEdgeMesh& other)
{
    Terminate();
    Initialize(other.numberOfPoints, other.numberOfFaces);
    CopyFromHost(other);
    return *this;
}

DeviceHalfEdgeMesh::DeviceHalfEdgeMesh(const DeviceHalfEdgeMesh& other)
{
    *this = other;
}

DeviceHalfEdgeMesh& DeviceHalfEdgeMesh::operator=(const DeviceHalfEdgeMesh& other)
{
    if (this == &other) return *this;
    Terminate();
    Initialize(other.numberOfPoints, other.numberOfFaces);

    CUDA_COPY_D2D(positions, other.positions, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2D(normals, other.normals, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2D(colors, other.colors, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2D(faces, other.faces, sizeof(uint3) * numberOfFaces);
    CUDA_COPY_D2D(halfEdges, other.halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
    CUDA_COPY_D2D(halfEdgeFaces, other.halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
    CUDA_COPY_D2D(vertexToHalfEdge, other.vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);

    CUDA_SYNC();

    min = other.min;
    max = other.max;
    return *this;
}

void DeviceHalfEdgeMesh::Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
    this->numberOfPoints = numberOfPoints;
    this->numberOfFaces = numberOfFaces;

    if (numberOfPoints > 0)
    {
        CUDA_MALLOC(&positions, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&normals, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&colors, sizeof(float3) * numberOfPoints);
    }
    else
    {
        positions = nullptr;
        normals = nullptr;
        colors = nullptr;
    }
    if (numberOfFaces > 0)
    {
        CUDA_MALLOC(&faces, sizeof(uint3) * numberOfFaces);
        CUDA_MALLOC(&halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
        CUDA_MALLOC(&halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
        CUDA_MALLOC(&vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);
        CUDA_MEMSET(vertexToHalfEdge, 0xFF, sizeof(unsigned int) * numberOfPoints);
    }
    else
    {
        faces = nullptr;
        halfEdges = nullptr;
        halfEdgeFaces = nullptr;
    }

    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    CUDA_MALLOC(&mortonKeys, sizeof(MortonKey) * numberOfFaces);
}

void DeviceHalfEdgeMesh::Terminate()
{
    CUDA_SAFE_FREE(positions);
    CUDA_SAFE_FREE(normals);
    CUDA_SAFE_FREE(colors);
    CUDA_SAFE_FREE(faces);
    CUDA_SAFE_FREE(halfEdges);
    CUDA_SAFE_FREE(halfEdgeFaces);
    CUDA_SAFE_FREE(vertexToHalfEdge);
    CUDA_SAFE_FREE(mortonKeys);

    CUDA_SAFE_FREE(faceNodes);

    faceNodeHashMap.Terminate();

    numberOfPoints = 0;
    numberOfFaces = 0;

    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
}

void DeviceHalfEdgeMesh::CopyFromHost(const HostHalfEdgeMesh& hostMesh)
{
    CUDA_COPY_H2D(positions, hostMesh.positions, sizeof(float3) * hostMesh.numberOfPoints);
    CUDA_COPY_H2D(normals, hostMesh.normals, sizeof(float3) * hostMesh.numberOfPoints);
    CUDA_COPY_H2D(colors, hostMesh.colors, sizeof(float3) * hostMesh.numberOfPoints);
    CUDA_COPY_H2D(faces, hostMesh.faces, sizeof(uint3) * hostMesh.numberOfFaces);
    CUDA_COPY_H2D(halfEdges, hostMesh.halfEdges, sizeof(HalfEdge) * hostMesh.numberOfFaces * 3);
    CUDA_COPY_H2D(halfEdgeFaces, hostMesh.halfEdgeFaces, sizeof(HalfEdgeFace) * hostMesh.numberOfFaces);
    CUDA_COPY_H2D(vertexToHalfEdge, hostMesh.vertexToHalfEdge, sizeof(unsigned int) * hostMesh.numberOfPoints);
    CUDA_SYNC();

    min = hostMesh.min;
    max = hostMesh.max;
}

void DeviceHalfEdgeMesh::CopyToHost(HostHalfEdgeMesh& hostMesh) const
{
    hostMesh.Terminate();
    hostMesh.Initialize(numberOfPoints, numberOfFaces);

    CUDA_COPY_D2H(hostMesh.positions, positions, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(hostMesh.normals, normals, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(hostMesh.colors, colors, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(hostMesh.faces, faces, sizeof(uint3) * numberOfFaces);
    CUDA_COPY_D2H(hostMesh.halfEdges, halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
    CUDA_COPY_D2H(hostMesh.halfEdgeFaces, halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
    CUDA_COPY_D2H(hostMesh.vertexToHalfEdge, vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);
    CUDA_SYNC();

    hostMesh.min = min;
    hostMesh.max = max;
}

void DeviceHalfEdgeMesh::RecalcAABB()
{
    float3* d_min = nullptr;
    float3* d_max = nullptr;

    CUDA_MALLOC(&d_min, sizeof(float3));
    CUDA_MALLOC(&d_max, sizeof(float3));

    float3 initial_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 initial_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    CUDA_COPY_H2D(d_min, &initial_min, sizeof(float3));
    CUDA_COPY_H2D(d_max, &initial_max, sizeof(float3));

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_RecalcAABB, numberOfPoints,
        positions, d_min, d_max, numberOfPoints);

    CUDA_COPY_D2H(&min, d_min, sizeof(float3));
    CUDA_COPY_D2H(&max, d_max, sizeof(float3));

    CUDA_SYNC();

	CUDA_SAFE_FREE(d_min);
    CUDA_SAFE_FREE(d_max);
}

void DeviceHalfEdgeMesh::UpdateBVH()
{
    bvh.Terminate();

    CUDA_TS(InitializeBVH);
    bvh.Initialize(positions, faces, min, max, numberOfFaces);
    CUDA_TE(InitializeBVH);
}

void DeviceHalfEdgeMesh::BuildHalfEdges()
{
    if (numberOfFaces == 0 || faces == nullptr) return;

    CUDA_TS(BuildHalfEdges);

    unsigned int numHalfEdges = numberOfFaces * 3;
    HashMap<uint64_t, unsigned int> edgeMap;
    edgeMap.Initialize(numHalfEdges * 2);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_BuildHalfEdges, numberOfFaces,
        faces, numberOfFaces, halfEdges, halfEdgeFaces, edgeMap.info);
    CUDA_SYNC();

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_LinkOpposites, numberOfFaces,
        faces, numberOfFaces, halfEdges, edgeMap.info);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_ValidateOppositeEdges, numHalfEdges, halfEdges, numHalfEdges);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_BuildVertexToHalfEdgeMapping, numHalfEdges,
        halfEdges, vertexToHalfEdge, numHalfEdges);
    CUDA_SYNC();

    CUDA_TE(BuildHalfEdges);

    edgeMap.Terminate();
}

void DeviceHalfEdgeMesh::RemoveIsolatedVertices()
{
    unsigned int* vertexIndexMapping = nullptr;
    CUDA_MALLOC(&vertexIndexMapping, sizeof(unsigned int) * numberOfPoints);
    CUDA_MEMSET(vertexIndexMapping, 0xFF, sizeof(unsigned int) * numberOfPoints);
    unsigned int* vertexIndexMappingIndex = nullptr;
    CUDA_MALLOC(&vertexIndexMappingIndex, sizeof(unsigned int));
    CUDA_MEMSET(vertexIndexMappingIndex, 0, sizeof(unsigned int));

    float3* newPositions = nullptr;
    CUDA_MALLOC(&newPositions, sizeof(float3) * numberOfPoints);
    float3* newNormals = nullptr;
    CUDA_MALLOC(&newNormals, sizeof(float3) * numberOfPoints);
    float3* newColors = nullptr;
    CUDA_MALLOC(&newColors, sizeof(float3) * numberOfPoints);

    unsigned int* newVertexToHalfEdge = nullptr;
    CUDA_MALLOC(&newVertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);

    unsigned int numberOfNewPoints = 0;
    LaunchKernel(Kernel_DeviceHalfEdgeMesh_CompactVertices, numberOfPoints,
        vertexToHalfEdge, vertexIndexMapping, vertexIndexMappingIndex,
        positions, normals, colors, newPositions, newNormals, newColors, numberOfPoints);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_RemapVertexToHalfEdge, numberOfPoints,
        vertexToHalfEdge, newVertexToHalfEdge, vertexIndexMapping, numberOfPoints);

    CUDA_COPY_D2H(&numberOfNewPoints, vertexIndexMappingIndex, sizeof(unsigned int));
    CUDA_SYNC();

    numberOfPoints = numberOfNewPoints;
    float3* oldPositions = positions;
    float3* oldNormals = normals;
    float3* oldColors = colors;

    positions = newPositions;
    normals = newNormals;
    colors = newColors;

    unsigned int* oldVertexToHalfEdge = vertexToHalfEdge;
    vertexToHalfEdge = newVertexToHalfEdge;

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_RemapVertexIndexOfFacesAndHalfEdges, numberOfFaces,
        faces, halfEdges, numberOfFaces, vertexIndexMapping);
    CUDA_SYNC();

    CUDA_FREE(oldPositions);
    CUDA_FREE(oldNormals);
    CUDA_FREE(oldColors);

    CUDA_FREE(oldVertexToHalfEdge);

    CUDA_FREE(vertexIndexMapping);
    CUDA_FREE(vertexIndexMappingIndex);

    RecalcAABB();
    UpdateBVH();
}

void DeviceHalfEdgeMesh::BuildFaceNodeHashMap()
{
    faceNodeHashMap.Terminate();
    faceNodeHashMap.Initialize(numberOfFaces * 64);
    faceNodeHashMap.Clear(0xFFFFFFFF);

    CUDA_MALLOC(&faceNodes, sizeof(FaceNode) * numberOfFaces);
    CUDA_MEMSET(faceNodes, 0xFF, sizeof(FaceNode) * numberOfFaces);

    float voxelSize = 0.1f;

    //printf("Face count: %u, HashMap capacity: %zu, maxProbe: %u\n", numberOfFaces, faceNodeHashMap.info.capacity, faceNodeHashMap.info.maxProbe);

    unsigned int* d_numDropped;
    CUDA_MALLOC(&d_numDropped, sizeof(unsigned int));
    CUDA_MEMSET(d_numDropped, 0, sizeof(unsigned int));

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_BuildFaceNodeHashMap, numberOfFaces,
        faceNodeHashMap.info,
        positions,
        faces,
        faceNodes,
        voxelSize,
        numberOfFaces,
        d_numDropped);
    CUDA_SYNC();

    unsigned int h_numDropped = 0;
    CUDA_COPY_D2H(&h_numDropped, d_numDropped, sizeof(unsigned int));
    printf("probe 실패 face 개수: %u\n", h_numDropped);

    //CUDA_FREE(faceNodes);
    CUDA_FREE(d_numDropped)
}

vector<float3> DeviceHalfEdgeMesh::GetFaceNodePositions()
{
    // 1. host로 데이터 복사
    std::vector<HashMapEntry<uint64_t, FaceNodeHashMapEntry>> h_entries(faceNodeHashMap.info.capacity);
    CUDA_COPY_D2H(h_entries.data(), faceNodeHashMap.info.entries,
        sizeof(HashMapEntry<uint64_t, FaceNodeHashMapEntry>) * faceNodeHashMap.info.capacity);

    std::vector<FaceNode> h_faceNodes(numberOfFaces);
    CUDA_COPY_D2H(h_faceNodes.data(), faceNodes, sizeof(FaceNode) * numberOfFaces);

    std::vector<float3> h_positions(numberOfPoints);
    std::vector<uint3> h_faces(numberOfFaces);
    CUDA_COPY_D2H(h_positions.data(), positions, sizeof(float3) * numberOfPoints);
    CUDA_COPY_D2H(h_faces.data(), faces, sizeof(uint3) * numberOfFaces);

    std::vector<float3> result;

    for (size_t i = 0; i < h_entries.size(); ++i)
    {
        const auto& entry = h_entries[i];
        if (entry.key == EMPTY_KEY)
            continue;

        unsigned int idx = entry.value.headIndex;
        size_t count = 0;
        while (idx != UINT32_MAX && idx < numberOfFaces && count < entry.value.length)
        {
            unsigned int faceIdx = h_faceNodes[idx].faceIndex;
            if (faceIdx >= numberOfFaces)
                break; // 잘못된 인덱스면 중단

            const uint3& f = h_faces[faceIdx];
            const float3& p0 = h_positions[f.x];
            const float3& p1 = h_positions[f.y];
            const float3& p2 = h_positions[f.z];
            float3 centroid = (p0 + p1 + p2) / 3.0f;
            result.push_back(centroid);

            idx = h_faceNodes[idx].nextNodeIndex;
            ++count;
        }
    }
    return result;
}

void DeviceHalfEdgeMesh::DebugPrintFaceNodeHashMap()
{
    std::vector<HashMapEntry<uint64_t, FaceNodeHashMapEntry>> h_entries(faceNodeHashMap.info.capacity);
    CUDA_COPY_D2H(h_entries.data(), faceNodeHashMap.info.entries,
        sizeof(HashMapEntry<uint64_t, FaceNodeHashMapEntry>) * faceNodeHashMap.info.capacity);
    std::vector<FaceNode> h_faceNodes(numberOfFaces);
    CUDA_COPY_D2H(h_faceNodes.data(), faceNodes, sizeof(FaceNode) * numberOfFaces);

    size_t totalCount = 0;
    //for (size_t slot = 0; slot < h_entries.size(); ++slot)
    //{
    //    const auto& entry = h_entries[slot];
    //    if (entry.key == EMPTY_KEY)
    //        continue;

    //    printf("[Slot %zu] key=0x%llx head=%u length=%u tail=%u\n",
    //        slot, entry.key, entry.value.headIndex, entry.value.length, entry.value.tailIndex);

    //    unsigned int idx = entry.value.headIndex;
    //    size_t chainCount = 0;
    //    std::set<unsigned int> visited;

    //    while (idx != UINT32_MAX && idx < numberOfFaces && chainCount < entry.value.length)
    //    {
    //        if (visited.count(idx))
    //        {
    //            printf("  [Loop Detected] at idx=%u\n", idx);
    //            break;
    //        }
    //        visited.insert(idx);

    //        unsigned int faceIdx = h_faceNodes[idx].faceIndex;
    //        printf("    [Node %zu] idx=%u faceIdx=%u next=%u\n", chainCount, idx, faceIdx, h_faceNodes[idx].nextNodeIndex);

    //        idx = h_faceNodes[idx].nextNodeIndex;
    //        ++chainCount;
    //    }

    //    if (chainCount != entry.value.length)
    //    {
    //        printf("  [Warning] chain count(%zu) != length(%u)\n", chainCount, entry.value.length);
    //    }
    //    totalCount += chainCount;
    //}

    for (size_t slot = 0; slot < h_entries.size(); ++slot)
    {
        const auto& entry = h_entries[slot];
        if (entry.key == EMPTY_KEY)
            continue;

        unsigned int idx = entry.value.headIndex;
        size_t chainCount = 0;
        std::set<unsigned int> visited;

        while (idx != UINT32_MAX && idx < numberOfFaces && chainCount < entry.value.length)
        {
            if (visited.count(idx)) break;
            visited.insert(idx);
            idx = h_faceNodes[idx].nextNodeIndex;
            ++chainCount;
        }

        printf("[Slot %zu] key=0x%llx length=%u chain=%zu %s\n",
            slot, entry.key, entry.value.length, chainCount,
            (chainCount != entry.value.length) ? "[불일치]" : "");

        totalCount += chainCount;
    }

    printf("==== 전체 연결된 face node 개수: %zu\n", totalCount);
}

std::vector<unsigned int> DeviceHalfEdgeMesh::FindUnlinkedFaceNodes()
{
    // 1. device에서 host로 faceNodes, 해시맵 엔트리 복사
    std::vector<FaceNode> h_faceNodes(numberOfFaces);
    CUDA_COPY_D2H(h_faceNodes.data(), faceNodes, sizeof(FaceNode) * numberOfFaces);

    std::vector<HashMapEntry<uint64_t, FaceNodeHashMapEntry>> h_entries(faceNodeHashMap.info.capacity);
    CUDA_COPY_D2H(h_entries.data(), faceNodeHashMap.info.entries,
        sizeof(HashMapEntry<uint64_t, FaceNodeHashMapEntry>) * faceNodeHashMap.info.capacity);

    // 2. 연결리스트를 따라가며 방문한 faceNode 인덱스 체크
    std::vector<bool> visited(numberOfFaces, false);

    for (size_t slot = 0; slot < h_entries.size(); ++slot)
    {
        const auto& entry = h_entries[slot];
        if (entry.key == EMPTY_KEY)
            continue;

        unsigned int idx = entry.value.headIndex;
        size_t count = 0;
        while (idx != UINT32_MAX && idx < numberOfFaces && count < entry.value.length)
        {
            if (visited[idx])
                break; // loop 보호
            visited[idx] = true;
            idx = h_faceNodes[idx].nextNodeIndex;
            ++count;
        }
    }

    // 3. 연결되지 않은 faceNode 인덱스 추출
    std::vector<unsigned int> unlinked;
    for (unsigned int i = 0; i < numberOfFaces; ++i)
    {
        if (!visited[i])
            unlinked.push_back(i);
    }

    return unlinked;
}

std::vector<unsigned int> DeviceHalfEdgeMesh::FindNearestTriangleIndices(float3* d_positions, unsigned int numberOfInputPoints)
{
    unsigned int* d_indices = nullptr;
    CUDA_MALLOC(&d_indices, sizeof(unsigned int) * numberOfInputPoints);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap, numberOfInputPoints,
        d_positions, numberOfInputPoints,
        positions, faces,
        faceNodeHashMap.info,
        faceNodes,
        /*voxelSize=*/0.1f,
        d_indices);

    CUDA_SYNC();

    std::vector<unsigned int> h_indices(numberOfInputPoints);
    CUDA_COPY_D2H(h_indices.data(), d_indices, sizeof(unsigned int) * numberOfInputPoints);
    CUDA_FREE(d_indices);

    return h_indices;
}

std::vector<unsigned int> DeviceHalfEdgeMesh::FindNearestTriangleIndicesAndClosestPoints(
    float3* d_positions, unsigned int numberOfInputPoints, int offset, std::vector<float3>& outClosestPoints)
{
    unsigned int* d_indices = nullptr;
    float3* d_closest = nullptr;
    CUDA_MALLOC(&d_indices, sizeof(unsigned int) * numberOfInputPoints);
    CUDA_MALLOC(&d_closest, sizeof(float3) * numberOfInputPoints);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap_ClosestPoint, numberOfInputPoints,
        d_positions, numberOfInputPoints,
        positions, faces,
        faceNodeHashMap.info,
        faceNodes,
        /*voxelSize=*/0.1f,
        offset,
        d_indices, d_closest);

    CUDA_SYNC();

    std::vector<unsigned int> h_indices(numberOfInputPoints);
    outClosestPoints.resize(numberOfInputPoints);
    CUDA_COPY_D2H(h_indices.data(), d_indices, sizeof(unsigned int) * numberOfInputPoints);
    CUDA_COPY_D2H(outClosestPoints.data(), d_closest, sizeof(float3) * numberOfInputPoints);
    CUDA_FREE(d_indices);
    CUDA_FREE(d_closest);

    return h_indices;
}

bool DeviceHalfEdgeMesh::PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const
{
    int* d_hitIdx;
    float* d_hitT;
    CUDA_MALLOC(&d_hitIdx, sizeof(int));
    CUDA_MALLOC(&d_hitT, sizeof(float));
    int initIdx = -1;
    float initT = 1e30f;
    CUDA_COPY_H2D(d_hitIdx, &initIdx, sizeof(int));
    CUDA_COPY_H2D(d_hitT, &initT, sizeof(float));

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_PickFace, numberOfFaces,
        rayOrigin, rayDir, positions, faces, numberOfFaces, d_hitIdx, d_hitT);
    CUDA_SYNC();

    CUDA_COPY_D2H(&outHitIndex, d_hitIdx, sizeof(int));
    CUDA_COPY_D2H(&outHitT, d_hitT, sizeof(float));

    CUDA_FREE(d_hitIdx);
    CUDA_FREE(d_hitT);

    return (outHitIndex >= 0);
}

std::vector<unsigned int> DeviceHalfEdgeMesh::GetOneRingVertices(unsigned int v, bool fixBorderVertices) const
{
    assert(v < numberOfPoints);

    unsigned int* d_neighbors = nullptr;
    unsigned int* d_count = nullptr;
    CUDA_MALLOC(&d_neighbors, sizeof(unsigned int) * MAX_NEIGHBORS);
    CUDA_MALLOC(&d_count, sizeof(unsigned int));
    CUDA_MEMSET(d_neighbors, 0xFF, sizeof(unsigned int) * MAX_NEIGHBORS);
    CUDA_MEMSET(d_count, 0, sizeof(unsigned int));

    Kernel_DeviceHalfEdgeMesh_OneRing << <1, 1 >> > (
        halfEdges,
        vertexToHalfEdge,
        numberOfPoints,
        v,
        fixBorderVertices,
        d_neighbors,
        d_count
        );
    CUDA_SYNC();

    unsigned int h_count = 0;
    unsigned int h_neighbors[MAX_NEIGHBORS];
    CUDA_COPY_D2H(&h_count, d_count, sizeof(unsigned int));
    CUDA_COPY_D2H(h_neighbors, d_neighbors, sizeof(unsigned int) * MAX_NEIGHBORS);

    CUDA_FREE(d_neighbors);
    CUDA_FREE(d_count);

    std::vector<unsigned int> result;
    for (unsigned int i = 0; i < h_count; ++i)
    {
        if (h_neighbors[i] != UINT32_MAX)
            result.push_back(h_neighbors[i]);
    }
    return result;
}

std::vector<unsigned int> DeviceHalfEdgeMesh::GetVerticesInRadius(unsigned int startVertex, float radius)
{
    if (startVertex >= numberOfPoints)
    {
        printf("[ERROR] startVertex(%u) >= numberOfPoints(%u)\n", startVertex, numberOfPoints);
        return {};
    }
    if (!positions || !halfEdges || !vertexToHalfEdge)
    {
        printf("[ERROR] DeviceHalfEdgeMesh not initialized.\n");
        return {};
    }

    unsigned int* d_visited = nullptr;
    unsigned int* d_frontier[2] = { nullptr, nullptr };
    unsigned int* d_frontierSize = nullptr;
    unsigned int* d_nextFrontierSize = nullptr;
    unsigned int* d_result = nullptr;
    unsigned int* d_resultSize = nullptr;

    CUDA_MALLOC(&d_visited, sizeof(unsigned int) * numberOfPoints);
    CUDA_MEMSET(d_visited, 0, sizeof(unsigned int) * numberOfPoints);
    CUDA_MALLOC(&d_frontier[0], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontier[1], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_nextFrontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_result, sizeof(unsigned int) * MAX_RESULT);
    CUDA_MALLOC(&d_resultSize, sizeof(unsigned int));
    CUDA_MEMSET(d_resultSize, 0, sizeof(unsigned int));

    unsigned int one = 1;
    CUDA_COPY_H2D(&d_visited[startVertex], &one, sizeof(unsigned int));
    CUDA_COPY_H2D(d_frontier[0], &startVertex, sizeof(unsigned int));
    unsigned int h_frontierSize = 1;
    CUDA_COPY_H2D(d_frontierSize, &h_frontierSize, sizeof(unsigned int));
    CUDA_COPY_H2D(d_result, &startVertex, sizeof(unsigned int));
    unsigned int resultInit = 1;
    CUDA_COPY_H2D(d_resultSize, &resultInit, sizeof(unsigned int));

    int iter = 0;
    while (true)
    {
        unsigned int currFrontierSize = 0;
        CUDA_COPY_D2H(&currFrontierSize, d_frontierSize, sizeof(unsigned int));
        if (currFrontierSize == 0) break;

        if (currFrontierSize >= MAX_FRONTIER)
        {
            printf("[BFS][ERROR] frontier buffer overflow. Increase MAX_FRONTIER or reduce radius.\n");
            break;
        }

        CUDA_MEMSET(d_nextFrontierSize, 0, sizeof(unsigned int));

        LaunchKernel(Kernel_DeviceHalfEdgeMesh_GetVerticesInRadius, currFrontierSize,
            positions, numberOfPoints, halfEdges, vertexToHalfEdge,
            d_frontier[iter % 2], currFrontierSize,
            d_visited,
            d_frontier[(iter + 1) % 2], d_nextFrontierSize,
            d_result, d_resultSize,
            startVertex, radius);
        CUDA_SYNC();

        CUDA_COPY_D2D(d_frontierSize, d_nextFrontierSize, sizeof(unsigned int));
        iter++;

        unsigned int h_resultSize = 0;
        CUDA_COPY_D2H(&h_resultSize, d_resultSize, sizeof(unsigned int));
        if (h_resultSize >= MAX_RESULT)
        {
            printf("[BFS][ERROR] result buffer overflow. Increase MAX_RESULT or reduce radius.\n");
            break;
        }
    }

    unsigned int h_resultSize = 0;
    CUDA_COPY_D2H(&h_resultSize, d_resultSize, sizeof(unsigned int));
    std::vector<unsigned int> h_result(h_resultSize);
    CUDA_COPY_D2H(h_result.data(), d_result, sizeof(unsigned int) * h_resultSize);

    CUDA_FREE(d_visited);
    CUDA_FREE(d_frontier[0]);
    CUDA_FREE(d_frontier[1]);
    CUDA_FREE(d_frontierSize);
    CUDA_FREE(d_nextFrontierSize);
    CUDA_FREE(d_result);
    CUDA_FREE(d_resultSize);

    return h_result;
}

std::vector<unsigned int> DeviceHalfEdgeMesh::GetOneRingFaces(unsigned int f) const
{
    unsigned int* d_faces = nullptr;
    unsigned int* d_count = nullptr;
    CUDA_MALLOC(&d_faces, sizeof(unsigned int) * 3); // 한 face의 1-ring face는 최대 3개
    CUDA_MALLOC(&d_count, sizeof(unsigned int));
    CUDA_MEMSET(d_count, 0, sizeof(unsigned int));

    Kernel_DeviceHalfEdgeMesh_OneRingFaces << <1, 1 >> > (
        f, halfEdges, numberOfFaces, d_faces, d_count
        );
    CUDA_SYNC();

    unsigned int h_count = 0;
    unsigned int h_faces[3];
    CUDA_COPY_D2H(&h_count, d_count, sizeof(unsigned int));
    CUDA_COPY_D2H(h_faces, d_faces, sizeof(unsigned int) * 3);

    CUDA_FREE(d_faces);
    CUDA_FREE(d_count);

    std::vector<unsigned int> result;
    for (unsigned int i = 0; i < h_count; ++i)
        result.push_back(h_faces[i]);
    return result;
}

std::vector<unsigned int> DeviceHalfEdgeMesh::GetFacesInRadius(unsigned int startFace, float radius)
{
    if (startFace >= numberOfFaces)
    {
        printf("[ERROR] startFace(%u) >= numberOfFaces(%u)\n", startFace, numberOfFaces);
        return {};
    }
    if (!positions || !halfEdges || !faces)
    {
        printf("[ERROR] DeviceHalfEdgeMesh not initialized.\n");
        return {};
    }

    unsigned int* d_visited = nullptr;
    unsigned int* d_frontier[2] = { nullptr, nullptr };
    unsigned int* d_frontierSize = nullptr;
    unsigned int* d_nextFrontierSize = nullptr;
    unsigned int* d_result = nullptr;
    unsigned int* d_resultSize = nullptr;

    CUDA_MALLOC(&d_visited, sizeof(unsigned int) * numberOfFaces);
    CUDA_MEMSET(d_visited, 0, sizeof(unsigned int) * numberOfFaces);
    CUDA_MALLOC(&d_frontier[0], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontier[1], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_nextFrontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_result, sizeof(unsigned int) * MAX_RESULT);
    CUDA_MALLOC(&d_resultSize, sizeof(unsigned int));
    CUDA_MEMSET(d_resultSize, 0, sizeof(unsigned int));

    unsigned int one = 1;
    CUDA_COPY_H2D(&d_visited[startFace], &one, sizeof(unsigned int));
    CUDA_COPY_H2D(d_frontier[0], &startFace, sizeof(unsigned int));
    unsigned int h_frontierSize = 1;
    CUDA_COPY_H2D(d_frontierSize, &h_frontierSize, sizeof(unsigned int));
    CUDA_COPY_H2D(d_result, &startFace, sizeof(unsigned int));
    unsigned int resultInit = 1;
    CUDA_COPY_H2D(d_resultSize, &resultInit, sizeof(unsigned int));

    int iter = 0;
    while (true)
    {
        unsigned int currFrontierSize = 0;
        CUDA_COPY_D2H(&currFrontierSize, d_frontierSize, sizeof(unsigned int));
        if (currFrontierSize == 0) break;

        if (currFrontierSize >= MAX_FRONTIER)
        {
            printf("[BFS][ERROR] frontier buffer overflow. Increase MAX_FRONTIER or reduce radius.\n");
            break;
        }

        CUDA_MEMSET(d_nextFrontierSize, 0, sizeof(unsigned int));

        LaunchKernel(Kernel_DeviceHalfEdgeMesh_GetFacesInRadius, currFrontierSize,
            positions, faces, halfEdges, numberOfFaces,
            d_frontier[iter % 2], currFrontierSize,
            d_visited,
            d_frontier[(iter + 1) % 2], d_nextFrontierSize,
            d_result, d_resultSize,
            startFace, radius);
        CUDA_SYNC();

        CUDA_COPY_D2D(d_frontierSize, d_nextFrontierSize, sizeof(unsigned int));
        iter++;

        unsigned int h_resultSize = 0;
        CUDA_COPY_D2H(&h_resultSize, d_resultSize, sizeof(unsigned int));
        if (h_resultSize >= MAX_RESULT)
        {
            printf("[BFS][ERROR] result buffer overflow. Increase MAX_RESULT or reduce radius.\n");
            break;
        }
    }

    unsigned int h_resultSize = 0;
    CUDA_COPY_D2H(&h_resultSize, d_resultSize, sizeof(unsigned int));
    std::vector<unsigned int> h_result(h_resultSize);
    CUDA_COPY_D2H(h_result.data(), d_result, sizeof(unsigned int) * h_resultSize);

    CUDA_FREE(d_visited);
    CUDA_FREE(d_frontier[0]);
    CUDA_FREE(d_frontier[1]);
    CUDA_FREE(d_frontierSize);
    CUDA_FREE(d_nextFrontierSize);
    CUDA_FREE(d_result);
    CUDA_FREE(d_resultSize);

    return h_result;
}

void DeviceHalfEdgeMesh::LaplacianSmoothing(unsigned int iterations, float lambda, bool fixBorderVertices)
{
    if (numberOfPoints == 0 || numberOfFaces == 0) return;

    float3* positionsA = positions;
    float3* positionsB = nullptr;
    CUDA_MALLOC(&positionsB, sizeof(float3) * numberOfPoints);

    float3* toFree = positionsB;

    auto numberOfHalfEdges = numberOfFaces * 3;

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_DeviceHalfEdgeMesh_LaplacianSmooth, numberOfPoints,
            positionsA, positionsB, numberOfPoints, fixBorderVertices, halfEdges, numberOfHalfEdges, vertexToHalfEdge, lambda);
        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
    {
        CUDA_COPY_D2D(positions, positionsA, sizeof(float3) * numberOfPoints);
        CUDA_SYNC();
    }

    CUDA_FREE(toFree);

    RecalcAABB();
    UpdateBVH();
}

void DeviceHalfEdgeMesh::RadiusLaplacianSmoothing(float radius, unsigned int iterations, float lambda)
{
    if (numberOfPoints == 0 || numberOfFaces == 0) return;

    float3* positionsA = positions;
    float3* positionsB = nullptr;
    CUDA_MALLOC(&positionsB, sizeof(float3) * numberOfPoints);
    float3* toFree = positionsB;

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_DeviceHalfEdgeMesh_RadiusLaplacianSmooth, numberOfPoints,
            positionsA, positionsB, numberOfPoints,
            halfEdges, vertexToHalfEdge, MAX_NEIGHBORS, radius, lambda);
        CUDA_SYNC();

        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
    {
        CUDA_COPY_D2D(positions, positionsA, sizeof(float3) * numberOfPoints);
        CUDA_SYNC();
    }
    CUDA_FREE(toFree);

    RecalcAABB();
    UpdateBVH();
}

void DeviceHalfEdgeMesh::GetAABBs(vector<cuAABB>& result, cuAABB& mMaabbs)
{
    result.resize(numberOfFaces);

    cuAABB* aabbs = nullptr;
    CUDA_MALLOC(&aabbs, sizeof(cuAABB) * numberOfFaces);

    cuAABB* d_mMaabbs = nullptr;
    CUDA_MALLOC(&d_mMaabbs, sizeof(cuAABB));
    CUDA_COPY_H2D(d_mMaabbs, &mMaabbs, sizeof(cuAABB));

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_GetAABB, numberOfFaces,
        positions, faces, aabbs, d_mMaabbs, numberOfPoints, numberOfFaces);

    CUDA_COPY_D2H(result.data(), aabbs, sizeof(cuAABB) * numberOfFaces);
    CUDA_COPY_D2H(&mMaabbs, d_mMaabbs, sizeof(cuAABB));
    CUDA_SYNC();

    CUDA_FREE(aabbs);
    CUDA_FREE(d_mMaabbs);
}

vector<uint64_t> DeviceHalfEdgeMesh::GetMortonCodes()
{
    float3 aabb_extent = max - min;
    float max_extent = fmaxf(aabb_extent.x, fmaxf(aabb_extent.y, aabb_extent.z));
    float voxelSize = max_extent / ((1 << 21) - 2); // safety margin

    uint64_t* d_mortonCodes = nullptr;
    CUDA_MALLOC(&d_mortonCodes, sizeof(uint64_t) * numberOfFaces);

    LaunchKernel(
        Kernel_DeviceHalfEdgeMesh_GetMortonCodes,
        numberOfFaces,
        positions, faces, d_mortonCodes, numberOfPoints, numberOfFaces,
        min, voxelSize
    );

    std::vector<uint64_t> result(numberOfFaces);
    CUDA_COPY_D2H(result.data(), d_mortonCodes, sizeof(uint64_t) * numberOfFaces);

    CUDA_FREE(d_mortonCodes);
    CUDA_SYNC();
    return result;
}

vector<float> DeviceHalfEdgeMesh::GetFaceCurvatures()
{
    float* d_curvatures = nullptr;
    CUDA_MALLOC(&d_curvatures, sizeof(float) * numberOfFaces);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_GetFaceCurvatures, numberOfFaces,
        positions, faces, halfEdges, numberOfFaces, d_curvatures);

    std::vector<float> h_curvatures(numberOfFaces);
    CUDA_COPY_D2H(h_curvatures.data(), d_curvatures, sizeof(float) * numberOfFaces);
    CUDA_FREE(d_curvatures);

    return h_curvatures;
}
#pragma endregion

#pragma region DeviceHalfEdgeMesh Device Functions
__host__ __device__ uint64_t DeviceHalfEdgeMesh::PackEdge(unsigned int v0, unsigned int v1)
{
    return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
}
__device__ bool DeviceHalfEdgeMesh::HashMapInsert(HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value)
{
    size_t idx = key % info.capacity;
    for (int i = 0; i < info.maxProbe; ++i)
    {
        size_t slot = (idx + i) % info.capacity;
        uint64_t* slot_key = reinterpret_cast<uint64_t*>(&info.entries[slot].key);
        uint64_t prev_key = atomicCAS(slot_key, empty_key<uint64_t>(), key);
        if (prev_key == empty_key<uint64_t>() || prev_key == key)
        {
            info.entries[slot].value = value;
            return true;
        }
    }
    return false;
}
__device__ bool DeviceHalfEdgeMesh::HashMapFind(const HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue)
{
    size_t idx = key % info.capacity;
    for (int i = 0; i < info.maxProbe; ++i)
    {
        size_t slot = (idx + i) % info.capacity;
        uint64_t k = info.entries[slot].key;
        if (k == key)
        {
            *outValue = info.entries[slot].value;
            return true;
        }
        if (k == empty_key<uint64_t>())
        {
            return false;
        }
    }
    return false;
}

__device__ void DeviceHalfEdgeMesh::GetOneRingVertices_Device(
    unsigned int v,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int numberOfPoints,
    bool fixBorderVertices,
    unsigned int* outNeighbors,
    unsigned int& outCount,
    unsigned int maxNeighbors)
{
    outCount = 0;
    if (!vertexToHalfEdge || v >= numberOfPoints)
        return;

    unsigned int ishe = vertexToHalfEdge[v];
    if (ishe == UINT32_MAX)
        return;
    unsigned int ihe = ishe;
    unsigned int lihe = ihe;
    bool borderFound = false;

    // cw
    do
    {
        const HalfEdge& he = halfEdges[ihe];
        if (he.nextIndex == UINT32_MAX) break;

        unsigned int ioe = he.oppositeIndex;
        if (ioe == UINT32_MAX)
        {
            lihe = ihe;
            borderFound = true;
            break;
        }
        const HalfEdge& oe = halfEdges[ioe];
        if (oe.vertexIndex == UINT32_MAX)
            break;

        if (outCount < maxNeighbors)
            outNeighbors[outCount++] = oe.vertexIndex;

        if (oe.nextIndex == UINT32_MAX)
            break;

        ihe = oe.nextIndex;
    } while (ihe != ishe && ihe != UINT32_MAX);

    // ccw
    if (borderFound)
    {
        outCount = 0;
        ishe = lihe;
        ihe = lihe;
        if (fixBorderVertices) return;

        do
        {
            const HalfEdge& he = halfEdges[ihe];
            if (he.nextIndex == UINT32_MAX)
                break;

            const HalfEdge& ne = halfEdges[he.nextIndex];
            if (ne.nextIndex == UINT32_MAX)
                break;

            if (ishe == ihe && outCount < maxNeighbors)
                outNeighbors[outCount++] = ne.vertexIndex;

            const HalfEdge& pe = halfEdges[ne.nextIndex];
            if (outCount < maxNeighbors)
                outNeighbors[outCount++] = pe.vertexIndex;

            ihe = pe.oppositeIndex;
        } while (ihe != ishe && ihe != UINT32_MAX);
    }
}

__device__ void DeviceHalfEdgeMesh::GetAllVerticesInRadius_Device(
    unsigned int vid,
    const float3* positions,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int* neighbors,
    unsigned int& outCount,
    unsigned int maxNeighbors,
    float radius)
{
    outCount = 0;
    if (vid >= numberOfPoints) return;

    unsigned int visited[MAX_NEIGHBORS];
    unsigned int frontier[MAX_NEIGHBORS];
    unsigned int nextFrontier[MAX_NEIGHBORS];

    unsigned int visitedCount = 0;
    unsigned int frontierSize = 0;
    unsigned int nextFrontierSize = 0;

    frontier[0] = vid;
    frontierSize = 1;
    visited[0] = vid;
    visitedCount = 1;

    neighbors[0] = vid;
    outCount = 1;

    float3 startPos = positions[vid];

    while (frontierSize > 0 && outCount < maxNeighbors)
    {
        nextFrontierSize = 0;

        for (unsigned int fi = 0; fi < frontierSize; ++fi)
        {
            unsigned int v = frontier[fi];
            unsigned int ringNeighbors[32];
            unsigned int nCount = 0;
            DeviceHalfEdgeMesh::GetOneRingVertices_Device(
                v, halfEdges, vertexToHalfEdge, numberOfPoints, false, ringNeighbors, nCount, 32);

            for (unsigned int ni = 0; ni < nCount; ++ni)
            {
                unsigned int nb = ringNeighbors[ni];
                if (nb == UINT32_MAX || nb >= numberOfPoints)
                    continue;

                bool alreadyVisited = false;
                for (unsigned int vi = 0; vi < visitedCount; ++vi)
                {
                    if (visited[vi] == nb)
                    {
                        alreadyVisited = true;
                        break;
                    }
                }
                if (alreadyVisited)
                    continue;

                float dist2 = length2(positions[nb] - startPos);
                if (dist2 <= radius * radius)
                {
                    if (visitedCount < maxNeighbors)
                        visited[visitedCount++] = nb;
                    if (outCount < maxNeighbors)
                        neighbors[outCount++] = nb;
                    if (nextFrontierSize < maxNeighbors)
                        nextFrontier[nextFrontierSize++] = nb;
                }
            }
        }

        frontierSize = nextFrontierSize;
        for (unsigned int i = 0; i < nextFrontierSize; ++i)
            frontier[i] = nextFrontier[i];
    }
}

__device__ void DeviceHalfEdgeMesh::GetOneRingFaces_Device(
    unsigned int f,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    unsigned int* outFaces,
    unsigned int& outCount)
{
    outCount = 0;
    if (f >= numberOfFaces) return;

    for (int e = 0; e < 3; ++e)
    {
        unsigned int heIdx = f * 3 + e;
        const HalfEdge& he = halfEdges[heIdx];
        if (he.oppositeIndex == UINT32_MAX)
            continue;

        const HalfEdge& oppHe = halfEdges[he.oppositeIndex];
        unsigned int neighborFace = oppHe.faceIndex;
        if (neighborFace != f)
            outFaces[outCount++] = neighborFace;
    }
}
#pragma endregion

#pragma region Kernel_DeviceHalfEdgeMesh
__global__ void Kernel_DeviceHalfEdgeMesh_BuildHalfEdges(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HalfEdgeFace* halfEdgeFaces,
    HashMapInfo<uint64_t, unsigned int> info)
{
    unsigned int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceIdx >= numberOfFaces) return;

    const uint3 face = faces[faceIdx];

    unsigned int v0 = face.x, v1 = face.y, v2 = face.z;

    unsigned int he0 = faceIdx * 3 + 0;
    unsigned int he1 = faceIdx * 3 + 1;
    unsigned int he2 = faceIdx * 3 + 2;

    halfEdges[he0].faceIndex = faceIdx;
    halfEdges[he1].faceIndex = faceIdx;
    halfEdges[he2].faceIndex = faceIdx;

    halfEdges[he0].nextIndex = he1;
    halfEdges[he1].nextIndex = he2;
    halfEdges[he2].nextIndex = he0;

    halfEdges[he0].vertexIndex = v0;
    halfEdges[he1].vertexIndex = v1;
    halfEdges[he2].vertexIndex = v2;

    halfEdges[he0].oppositeIndex = UINT32_MAX;
    halfEdges[he1].oppositeIndex = UINT32_MAX;
    halfEdges[he2].oppositeIndex = UINT32_MAX;

    halfEdgeFaces[faceIdx].halfEdgeIndex = he0;

    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v0, v1), he0);
    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v1, v2), he1);
    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v2, v0), he2);
}

__global__ void Kernel_DeviceHalfEdgeMesh_LinkOpposites(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HashMapInfo<uint64_t, unsigned int> info)
{
    unsigned int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceIdx >= numberOfFaces) return;

    const uint3 face = faces[faceIdx];
    unsigned int he0 = faceIdx * 3 + 0;
    unsigned int he1 = faceIdx * 3 + 1;
    unsigned int he2 = faceIdx * 3 + 2;

    {
        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(face.y, face.x);
        unsigned int oppIdx = UINT32_MAX;
        if (DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx))
            halfEdges[he0].oppositeIndex = oppIdx;
    }
    {
        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(face.z, face.y);
        unsigned int oppIdx = UINT32_MAX;
        if (DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx))
            halfEdges[he1].oppositeIndex = oppIdx;
    }
    {
        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(face.x, face.z);
        unsigned int oppIdx = UINT32_MAX;
        if (DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx))
            halfEdges[he2].oppositeIndex = oppIdx;
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_ValidateOppositeEdges(const HalfEdge* halfEdges, size_t numHalfEdges)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numHalfEdges)
        return;

    const HalfEdge& he = halfEdges[i];

    if (i < 5)
    {
        printf("[Entry] Checking halfEdge[%zu]: vertex = %u, next = %u, opp = %u\n",
            i, he.vertexIndex, he.nextIndex, he.oppositeIndex);
    }

    if (he.oppositeIndex == UINT32_MAX)
    {
        if (i < 5)
            printf("[Boundary] halfEdge[%zu] has no opposite\n", i);
        return;
    }

    const HalfEdge& ohe = halfEdges[he.oppositeIndex];

    if (ohe.oppositeIndex != i)
    {
        printf("[Invalid Opposite] he[%zu] → opp[%u] → opp = %u (expected %zu)\n",
            i, he.oppositeIndex, ohe.oppositeIndex, i);
    }

    uint32_t he_v0 = he.vertexIndex;
    uint32_t he_v1 = halfEdges[he.nextIndex].vertexIndex;

    uint32_t ohe_v0 = ohe.vertexIndex;
    uint32_t ohe_v1 = halfEdges[ohe.nextIndex].vertexIndex;

    if (!(he_v0 == ohe_v1 && he_v1 == ohe_v0))
    {
        printf("[Mismatch] Edge direction mismatch at he[%zu]: %u→%u vs opp[%u]: %u→%u\n",
            i, he_v0, he_v1, he.oppositeIndex, ohe_v0, ohe_v1);
    }

    const HalfEdge& e1 = halfEdges[he.nextIndex];
    const HalfEdge& e2 = halfEdges[e1.nextIndex];
    if (e2.nextIndex != i)
    {
        printf("[Loop Error] halfEdge[%zu] has broken loop\n", i);
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_BuildVertexToHalfEdgeMapping(
    const HalfEdge* halfEdges,
    unsigned int* vertexToHalfEdge,
    unsigned int numberOfHalfEdges)
{
    unsigned int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (heIdx >= numberOfHalfEdges) return;

    unsigned int v = halfEdges[heIdx].vertexIndex;

    atomicMin(&vertexToHalfEdge[v], heIdx);
}

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
    unsigned int numberOfPoints)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfPoints) return;

    if (UINT32_MAX == vertexToHalfEdge[threadid]) return;

    auto index = atomicAdd(vertexIndexMappingIndex, 1);
    vertexIndexMapping[threadid] = index;
    newPositions[index] = oldPositions[threadid];
    newNormals[index] = oldNormals[threadid];
    newColors[index] = oldColors[threadid];
}

__global__ void Kernel_DeviceHalfEdgeMesh_RemapVertexToHalfEdge(
    unsigned int* vertexToHalfEdge,
    unsigned int* newVertexToHalfEdge,
    unsigned int* vertexIndexMapping,
    unsigned int numberOfPoints)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfPoints) return;

    if (vertexIndexMapping[threadid] == UINT32_MAX) return;

    unsigned int newIndex = vertexIndexMapping[threadid];
    newVertexToHalfEdge[newIndex] = vertexToHalfEdge[threadid];
}

__global__ void Kernel_DeviceHalfEdgeMesh_RemapVertexIndexOfFacesAndHalfEdges(uint3* faces, HalfEdge* halfEdges, unsigned int numberOfFaces, unsigned int* vertexIndexMapping)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfFaces) return;

    auto& f = faces[threadid];
    f.x = vertexIndexMapping[f.x];
    f.y = vertexIndexMapping[f.y];
    f.z = vertexIndexMapping[f.z];

    halfEdges[threadid * 3].vertexIndex = vertexIndexMapping[halfEdges[threadid * 3].vertexIndex];
    halfEdges[threadid * 3 + 1].vertexIndex = vertexIndexMapping[halfEdges[threadid * 3 + 1].vertexIndex];
    halfEdges[threadid * 3 + 2].vertexIndex = vertexIndexMapping[halfEdges[threadid * 3 + 2].vertexIndex];
}

__global__ void Kernel_DeviceHalfEdgeMesh_BuildFaceNodeHashMap(
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> info,
    float3* positions,
    uint3* faces,
    FaceNode* faceNodes,
    float voxelSize,
    unsigned int numberOfFaces,
    unsigned int* d_numDropped)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfFaces) return;

    auto f = faces[threadid];
    auto p0 = positions[f.x];
    auto p1 = positions[f.y];
    auto p2 = positions[f.z];
    auto centroid = (p0 + p1 + p2) / 3.0f;

    auto voxelIndex = PositionToIndex(centroid, voxelSize);
    auto voxelKey = IndexToVoxelKey(voxelIndex);
    auto hashIdx = VoxelKeyHash(voxelKey, info.capacity);

    bool inserted = false;
    for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
    {
        size_t slot = (hashIdx + probe) % info.capacity;
        HashMapEntry<uint64_t, FaceNodeHashMapEntry>* entry = &info.entries[slot];

        uint64_t old = atomicCAS(
            reinterpret_cast<unsigned long long*>(&entry->key),
            EMPTY_KEY, voxelKey);

        if (old == EMPTY_KEY || old == voxelKey)
        {
            unsigned int prevHead;
            do {
                prevHead = entry->value.headIndex;
            } while (atomicCAS(&(entry->value.headIndex), prevHead, threadid) != prevHead);

            faceNodes[threadid].faceIndex = threadid;
            faceNodes[threadid].nextNodeIndex = prevHead;

            // length: atomicAdd로 안전하게 누적
            unsigned int prevLen = atomicAdd(&(entry->value.length), 1);

            // tailIndex: atomicCAS로 "최초 tail"만 등록
            // prevHead == UINT32_MAX이면 내가 최초 등록자
            if (prevHead == UINT32_MAX)
            {
                // 최초 등록자는 tailIndex도 자신으로 세팅
                atomicExch(&(entry->value.tailIndex), threadid);
            }
            // (tailIndex는 최초 등록자만 자신의 index로 set됨)

            return;
        }
    }

    if (!inserted)
        atomicAdd(d_numDropped, 1);
}

__global__ void Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap(
    const float3* points,
    unsigned int numPoints,
    const float3* positions,
    const uint3* faces,
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> faceNodeHashMap,
    const FaceNode* faceNodes,
    float voxelSize,
    unsigned int* outIndices)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float3 p = points[idx];

    int3 centerVoxel = PositionToIndex(p, voxelSize);
    unsigned int minFace = UINT32_MAX;
    float minDist2 = 1e30f;

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 voxelIndex = make_int3(centerVoxel.x + dx, centerVoxel.y + dy, centerVoxel.z + dz);
                uint64_t voxelKey = IndexToVoxelKey(voxelIndex);
                size_t hashIdx = VoxelKeyHash(voxelKey, faceNodeHashMap.capacity);

                for (unsigned int probe = 0; probe < faceNodeHashMap.maxProbe; ++probe)
                {
                    size_t slot = (hashIdx + probe) % faceNodeHashMap.capacity;
                    const auto& entry = faceNodeHashMap.entries[slot];
                    if (entry.key == voxelKey)
                    {
                        unsigned int nodeIdx = entry.value.headIndex;
                        while (nodeIdx != UINT32_MAX)
                        {
                            const FaceNode& fn = faceNodes[nodeIdx];
                            unsigned int faceIdx = fn.faceIndex;
                            uint3 tri = faces[faceIdx];
                            float3 a = positions[tri.x];
                            float3 b = positions[tri.y];
                            float3 c = positions[tri.z];
                            float dist2 = PointTriangleDistance2(p, a, b, c);
                            if (dist2 < minDist2)
                            {
                                minDist2 = dist2;
                                minFace = faceIdx;
                            }
                            nodeIdx = fn.nextNodeIndex;
                        }
                        break;
                    }
                    if (entry.key == EMPTY_KEY)
                        break;
                }
            }

    outIndices[idx] = minFace;
}

__global__ void Kernel_DeviceHalfEdgeMesh_FindNearestTriangles_HashMap_ClosestPoint(
    const float3* points,
    unsigned int numPoints,
    const float3* positions,
    const uint3* faces,
    HashMapInfo<uint64_t, FaceNodeHashMapEntry> faceNodeHashMap,
    const FaceNode* faceNodes,
    float voxelSize,
    int offset,
    unsigned int* outIndices,
    float3* outClosestPoints)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float3 p = points[idx];
    int3 centerVoxel = PositionToIndex(p, voxelSize);

    unsigned int minFace = UINT32_MAX;
    float minDist2 = 1e30f;
    float3 closest = make_float3(0, 0, 0);

    for (int dz = -offset; dz <= offset; ++dz)
        for (int dy = -offset; dy <= offset; ++dy)
            for (int dx = -offset; dx <= offset; ++dx)
            {
                int3 voxelIndex = make_int3(centerVoxel.x + dx, centerVoxel.y + dy, centerVoxel.z + dz);
                uint64_t voxelKey = IndexToVoxelKey(voxelIndex);
                size_t hashIdx = VoxelKeyHash(voxelKey, faceNodeHashMap.capacity);

                for (unsigned int probe = 0; probe < faceNodeHashMap.maxProbe; ++probe)
                {
                    size_t slot = (hashIdx + probe) % faceNodeHashMap.capacity;
                    const auto& entry = faceNodeHashMap.entries[slot];
                    if (entry.key == voxelKey)
                    {
                        unsigned int nodeIdx = entry.value.headIndex;
                        while (nodeIdx != UINT32_MAX)
                        {
                            const FaceNode& fn = faceNodes[nodeIdx];
                            unsigned int faceIdx = fn.faceIndex;
                            uint3 tri = faces[faceIdx];
                            float3 a = positions[tri.x];
                            float3 b = positions[tri.y];
                            float3 c = positions[tri.z];
                            float3 proj = ClosestPointOnTriangle(p, a, b, c);
                            float dist2 = length2(p - proj);

                            if (dist2 < minDist2)
                            {
                                minDist2 = dist2;
                                minFace = faceIdx;
                                closest = proj;
                            }
                            nodeIdx = fn.nextNodeIndex;
                        }
                        break;
                    }
                    if (entry.key == EMPTY_KEY)
                        break;
                }
            }
    outIndices[idx] = minFace;
    outClosestPoints[idx] = closest;
}

__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    bool fixborderVertices,
    const HalfEdge* halfEdges,
    unsigned int numberOfHalfEdges,
    unsigned int* vertexToHalfEdge,
    float lambda)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    unsigned int neighbors[MAX_NEIGHBORS];
    unsigned int nCount = 0;

    DeviceHalfEdgeMesh::GetOneRingVertices_Device(
        vid,
        halfEdges,
        vertexToHalfEdge,
        numberOfPoints,
        fixborderVertices,
        neighbors,
        nCount,
        MAX_NEIGHBORS);

    float3 center = positions_in[vid];
    if (nCount == 0)
    {
        positions_out[vid] = center;
        return;
    }

    float3 sum = make_float3(0, 0, 0);
    for (unsigned int i = 0; i < nCount; ++i)
    {
        unsigned int nb = neighbors[i];
        if (nb < numberOfPoints)
            sum += positions_in[nb];
    }

    float3 avg = sum / (float)nCount;
    positions_out[vid] = center + lambda * (avg - center);
}

__global__ void Kernel_DeviceHalfEdgeMesh_PickFace(
    const float3 rayOrigin,
    const float3 rayDir,
    const float3* positions,
    const uint3* faces,
    unsigned int numberOfFaces,
    int* outHitIndex,
    float* outHitT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberOfFaces) return;

    const uint3 tri = faces[i];
    const float3 v0 = positions[tri.x];
    const float3 v1 = positions[tri.y];
    const float3 v2 = positions[tri.z];
    float t, u, v;
    if (RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
    {
        atomicMinF(outHitT, t, outHitIndex, i);
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_OneRing(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int numberOfPoints,
    unsigned int v,
    bool fixBorderVertices,
    unsigned int* outBuffer,
    unsigned int* outCount)
{
    if (threadIdx.x == 0)
    {
        unsigned int neighbors[32];
        unsigned int nCount = 0;
        DeviceHalfEdgeMesh::GetOneRingVertices_Device(
            v, halfEdges, vertexToHalfEdge, numberOfPoints, fixBorderVertices, neighbors, nCount, 32);
        for (unsigned int i = 0; i < nCount; ++i)
        {
            outBuffer[i] = neighbors[i];
        }
        *outCount = nCount;
    }
}

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
    float radius)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontierSize) return;

    unsigned int v = frontier[idx];

    float3 startPos = positions[startVertex];

    unsigned int neighbors[32];
    unsigned int nCount = 0;
    DeviceHalfEdgeMesh::GetOneRingVertices_Device(
        v, halfEdges, vertexToHalfEdge, numberOfPoints, false, neighbors, nCount, 32);

    for (unsigned int i = 0; i < nCount; ++i)
    {
        unsigned int nb = neighbors[i];
        if (nb == UINT32_MAX || nb >= numberOfPoints)
            continue;
        unsigned int old = atomicExch(&visited[nb], 1);
        if (old == 0)
        {
            float dist2 = length2(positions[nb] - startPos);
            if (dist2 <= radius * radius)
            {
                unsigned int nfIdx = atomicAdd(nextFrontierSize, 1);
                if (nfIdx < MAX_FRONTIER)
                    nextFrontier[nfIdx] = nb;

                unsigned int resIdx = atomicAdd(resultSize, 1);
                if (resIdx < MAX_RESULT)
                    result[resIdx] = nb;
            }
        }
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_OneRingFaces(
    unsigned int f,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    unsigned int* outFaces,
    unsigned int* outCount)
{
    if (threadIdx.x == 0)
    {
        DeviceHalfEdgeMesh::GetOneRingFaces_Device(f, halfEdges, numberOfFaces, outFaces, *outCount);
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_GetFacesInRadius(
    const float3* positions,
    const uint3* faces,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    const unsigned int* frontier,
    unsigned int frontierSize,
    unsigned int* visited,
    unsigned int* nextFrontier,
    unsigned int* nextFrontierSize,
    unsigned int* result,
    unsigned int* resultSize,
    unsigned int startFace,
    float radius)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontierSize) return;

    unsigned int f = frontier[idx];

    // centroid 계산
    uint3 face = faces[f];
    float3 center = (positions[face.x] + positions[face.y] + positions[face.z]) / 3.0f;

    // 각 halfedge로 연결된 face 순회
    for (int e = 0; e < 3; ++e)
    {
        unsigned int heIdx = f * 3 + e;
        const HalfEdge& he = halfEdges[heIdx];
        if (he.oppositeIndex == UINT32_MAX) continue;
        unsigned int nbFace = halfEdges[he.oppositeIndex].faceIndex;
        if (visited[nbFace]) continue;

        uint3 nbF = faces[nbFace];
        float3 nbCenter = (positions[nbF.x] + positions[nbF.y] + positions[nbF.z]) / 3.0f;
        float dist2 = length2(nbCenter - center);
        if (dist2 <= radius * radius)
        {
            unsigned int old = atomicExch(&visited[nbFace], 1);
            if (old == 0)
            {
                unsigned int nfIdx = atomicAdd(nextFrontierSize, 1);
                if (nfIdx < MAX_FRONTIER)
                    nextFrontier[nfIdx] = nbFace;

                unsigned int resIdx = atomicAdd(resultSize, 1);
                if (resIdx < MAX_RESULT)
                    result[resIdx] = nbFace;
            }
        }
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_RadiusLaplacianSmooth(
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int maxNeighbors,
    float radius,
    float lambda)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    float3 center = positions_in[vid];

    unsigned int neighbors[64];
    unsigned int nCount = 0;

    DeviceHalfEdgeMesh::GetAllVerticesInRadius_Device(
        vid, positions_in, numberOfPoints, halfEdges, vertexToHalfEdge,
        neighbors, nCount, maxNeighbors, radius);

    float3 sum = make_float3(0, 0, 0);
    int count = 0;
    for (unsigned int i = 0; i < nCount; ++i)
    {
        unsigned int nb = neighbors[i];
        if (nb == vid || nb == UINT32_MAX) continue;
        sum += positions_in[nb];
        ++count;
    }
    if (count > 0)
    {
        float3 avg = sum / (float)count;
        positions_out[vid] = center + lambda * (avg - center);
    }
    else
    {
        positions_out[vid] = center;
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_GetAABB(
    float3* positions,
    uint3* faces,
    cuAABB* aabbs,
    cuAABB* mMaabbs,
    unsigned int numberOfPoints,
    unsigned int numberOfFaces)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfFaces) return;

    auto& f = faces[threadid];

    if (f.x >= numberOfPoints) { printf("f.x >= nop"); return; }
    if (f.y >= numberOfPoints) { printf("f.y >= nop"); return; }
    if (f.z >= numberOfPoints) { printf("f.z >= nop"); return; }

    auto& p0 = positions[f.x];
    auto& p1 = positions[f.y];
    auto& p2 = positions[f.z];

    cuAABB aabb;
    aabb.min = make_float3(
        fminf(p0.x, fminf(p1.x, p2.x)),
        fminf(p0.y, fminf(p1.y, p2.y)),
        fminf(p0.z, fminf(p1.z, p2.z))
    );
    aabb.max = make_float3(
        fmaxf(p0.x, fmaxf(p1.x, p2.x)),
        fmaxf(p0.y, fmaxf(p1.y, p2.y)),
        fmaxf(p0.z, fmaxf(p1.z, p2.z))
    );
    aabbs[threadid] = aabb;

    auto delta = aabb.max - aabb.min;
    mMaabbs->min = make_float3(
        fminf(mMaabbs->min.x, delta.x),
        fminf(mMaabbs->min.y, delta.y),
        fminf(mMaabbs->min.z, delta.z));
    mMaabbs->max = make_float3(
        fmaxf(mMaabbs->max.x, delta.x),
        fmaxf(mMaabbs->max.y, delta.y),
        fmaxf(mMaabbs->max.z, delta.z));
}

__global__ void Kernel_DeviceHalfEdgeMesh_GetMortonCodes(
    const float3* positions,
    const uint3* faces,
    uint64_t* mortonCodes,
    unsigned int numberOfPoints,
    unsigned int numberOfFaces,
    float3 min_corner,
    float voxel_size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfFaces) return;

    uint3 f = faces[tid];
    if (f.x >= numberOfPoints || f.y >= numberOfPoints || f.z >= numberOfPoints)
        return;

    float3 p0 = positions[f.x];
    float3 p1 = positions[f.y];
    float3 p2 = positions[f.z];
    float3 centroid = (p0 + p1 + p2) / 3.0f;

    mortonCodes[tid] = Float3ToMorton64(centroid, min_corner, voxel_size);
}

__global__ void Kernel_DeviceHalfEdgeMesh_RecalcAABB(float3* positions, float3* min, float3* max, unsigned int numberOfPoints)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= numberOfPoints) return;

    auto& p = positions[threadid];
    atomicMinF(&min->x, p.x);
    atomicMinF(&min->y, p.y);
    atomicMinF(&min->z, p.z);
    atomicMaxF(&max->x, p.x);
    atomicMaxF(&max->y, p.y);
    atomicMaxF(&max->z, p.z);
}

__global__ void Kernel_DeviceHalfEdgeMesh_GetFaceCurvatures(
    const float3* positions,
    const uint3* faces,
    const HalfEdge* halfEdges,
    unsigned int numberOfFaces,
    float* outCurvatures)
{
    unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numberOfFaces) return;

    // 1. Face normal
    const uint3 tri = faces[f];
    float3 v0 = positions[tri.x];
    float3 v1 = positions[tri.y];
    float3 v2 = positions[tri.z];
    float3 n = normalize(cross(v1 - v0, v2 - v0));

    float sumAngles = 0.0f;
    int count = 0;
    for (int e = 0; e < 3; ++e)
    {
        const HalfEdge& he = halfEdges[f * 3 + e];
        if (he.oppositeIndex == UINT32_MAX)
            continue;
        const HalfEdge& opp = halfEdges[he.oppositeIndex];
        unsigned int neighborFace = opp.faceIndex;
        if (neighborFace == f) continue;

        // neighbor normal
        const uint3 tri_nb = faces[neighborFace];
        float3 nv0 = positions[tri_nb.x];
        float3 nv1 = positions[tri_nb.y];
        float3 nv2 = positions[tri_nb.z];
        float3 n_nb = normalize(cross(nv1 - nv0, nv2 - nv0));

        float dotval = dot(n, n_nb);
        // clamp(-1,1)
        dotval = fmaxf(-1.0f, fminf(1.0f, dotval));
        float angle = acosf(dotval);

        sumAngles += angle;
        ++count;
    }
    outCurvatures[f] = (count > 0) ? (sumAngles / count) : 0.0f;
}
#pragma endregion
