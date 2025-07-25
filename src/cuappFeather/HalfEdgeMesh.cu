#include <HalfEdgeMesh.cuh>
#include <Mesh.cuh>

#include <set>

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
}

uint64_t HostHalfEdgeMesh::PackEdge(unsigned int v0, unsigned int v1)
{
    return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
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

        edgeMap[PackEdge(face.x, face.y)] = i * 3 + 0; // v0->v1
        edgeMap[PackEdge(face.y, face.z)] = i * 3 + 1; // v1->v2
        edgeMap[PackEdge(face.z, face.x)] = i * 3 + 2; // v2->v0
    }

    for (unsigned int i = 0; i < numberOfFaces; ++i)
    {
        auto& face = faces[i];
        // 0: v0->v1, 1: v1->v2, 2: v2->v0
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

    // 1. Points
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

    // 2. Faces using halfEdgeFaces (robust even for non-contiguous memory or non-trivial face topologies)
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

    // Points
    const auto& pts = ply.GetPoints();
    for (unsigned int i = 0; i < n; ++i)
    {
        positions[i].x = pts[i * 3 + 0];
        positions[i].y = pts[i * 3 + 1];
        positions[i].z = pts[i * 3 + 2];
    }

    // Normals
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

    // Colors
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

    // Faces
    const auto& idx = ply.GetTriangleIndices();
    for (unsigned int i = 0; i < nf; ++i)
    {
        faces[i].x = idx[i * 3 + 0];
        faces[i].y = idx[i * 3 + 1];
        faces[i].z = idx[i * 3 + 2];
    }

    // HalfEdge 구조 생성
    BuildHalfEdges();

    return true;
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
        if (DeviceHalfEdgeMesh::RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
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
    return *this;
}

void DeviceHalfEdgeMesh::Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
    this->numberOfPoints = numberOfPoints;
    this->numberOfFaces = numberOfFaces;

    if (numberOfPoints > 0)
    {
        cudaMalloc(&positions, sizeof(float3) * numberOfPoints);
        cudaMalloc(&normals, sizeof(float3) * numberOfPoints);
        cudaMalloc(&colors, sizeof(float3) * numberOfPoints);
    }
    else
    {
        positions = nullptr;
        normals = nullptr;
        colors = nullptr;
    }
    if (numberOfFaces > 0)
    {
        cudaMalloc(&faces, sizeof(uint3) * numberOfFaces);
        cudaMalloc(&halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
        cudaMalloc(&halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
        cudaMalloc(&vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);
        cudaMemset(vertexToHalfEdge, 0xFF, sizeof(unsigned int) * numberOfPoints);
    }
    else
    {
        faces = nullptr;
        halfEdges = nullptr;
        halfEdgeFaces = nullptr;
    }
}

void DeviceHalfEdgeMesh::Terminate()
{
    if (positions) cudaFree(positions);
    if (normals) cudaFree(normals);
    if (colors) cudaFree(colors);
    if (faces) cudaFree(faces);
    if (halfEdges) cudaFree(halfEdges);
    if (halfEdgeFaces) cudaFree(halfEdgeFaces);
    if (vertexToHalfEdge) cudaFree(vertexToHalfEdge);

    positions = nullptr;
    normals = nullptr;
    colors = nullptr;
    faces = nullptr;
    halfEdges = nullptr;
    halfEdgeFaces = nullptr;
    vertexToHalfEdge = nullptr;
    numberOfPoints = 0;
    numberOfFaces = 0;
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
}

void DeviceHalfEdgeMesh::BuildHalfEdges()
{
    if (numberOfFaces == 0 || faces == nullptr) return;

    size_t numHalfEdges = numberOfFaces * 3;
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

    edgeMap.Terminate();
}

bool DeviceHalfEdgeMesh::PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const
{
    int* d_hitIdx;
    float* d_hitT;
    cudaMalloc(&d_hitIdx, sizeof(int));
    cudaMalloc(&d_hitT, sizeof(float));
    int initIdx = -1;
    float initT = 1e30f;
    cudaMemcpy(d_hitIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hitT, &initT, sizeof(float), cudaMemcpyHostToDevice);

    LaunchKernel(Kernel_DeviceHalfEdgeMesh_PickFace, numberOfFaces,
        rayOrigin, rayDir, positions, faces, numberOfFaces, d_hitIdx, d_hitT);
    CUDA_SYNC();

    cudaMemcpy(&outHitIndex, d_hitIdx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&outHitT, d_hitT, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_hitIdx);
    cudaFree(d_hitT);

    return (outHitIndex >= 0);
}

void DeviceHalfEdgeMesh::LaplacianSmoothing(unsigned int iterations, float lambda)
{
    if (numberOfPoints == 0 || numberOfFaces == 0) return;

    float3* positionsA = positions;
    float3* positionsB = nullptr;
    cudaMalloc(&positionsB, sizeof(float3) * numberOfPoints);

    float3* toFree = positionsB;

    auto numberOfHalfEdges = numberOfFaces * 3;

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_DeviceHalfEdgeMesh_LaplacianSmooth, numberOfPoints,
            positionsA, positionsB, numberOfPoints, halfEdges, numberOfHalfEdges, lambda);
        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
    {
        CUDA_COPY_D2D(positions, positionsA, sizeof(float3) * numberOfPoints);
        CUDA_SYNC();
    }

    cudaFree(toFree);
}

void DeviceHalfEdgeMesh::LaplacianSmoothingNRing(unsigned int iterations, float lambda, int nRing)
{
 /*   if (numberOfPoints == 0 || numberOfFaces == 0) return;

    float3* positionsA = positions;
    float3* positionsB;
    cudaMalloc(&positionsB, sizeof(float3) * numberOfPoints);

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_DeviceHalfEdgeMesh_LaplacianSmoothNRing, numberOfPoints,
            halfEdges, vertexToHalfEdge, positionsA, positionsB, numberOfPoints, lambda, nRing);
        CUDA_SYNC();
        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
    {
        CUDA_COPY_D2D(positions, positionsA, sizeof(float3) * numberOfPoints);
    }
    cudaFree(positionsB);*/
}

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

__host__ __device__
bool DeviceHalfEdgeMesh::RayTriangleIntersect(const float3& orig, const float3& dir,
    const float3& v0, const float3& v1, const float3& v2,
    float& t, float& u, float& v)
{
    const float EPSILON = 1e-6f;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (fabs(a) < EPSILON) return false; // 평행
    float f = 1.0f / a;
    float3 s = orig - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross(s, edge1);
    v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * dot(edge2, q);
    return t > EPSILON; // ray 방향상의 hit만
}

__device__ void DeviceHalfEdgeMesh::atomicMinF(float* addr, float val, int* idx, int myIdx)
{
    int* intAddr = (int*)addr;
    int old = *intAddr, assumed;
    do
    {
        assumed = old;
        float oldVal = __int_as_float(assumed);
        if (val >= oldVal) break;
        old = atomicCAS(intAddr, assumed, __float_as_int(val));
        if (old == assumed)
            *idx = myIdx;
    } while (old != assumed);
}

__device__ void DeviceHalfEdgeMesh::atomicMinWithIndex(float* address, float val, int* idxAddress, int idx)
{
    int* intAddress = reinterpret_cast<int*>(address);
    int old = *intAddress, assumed;
    do
    {
        assumed = old;
        float oldVal = __int_as_float(assumed);
        if (val >= oldVal)
            return;
        old = atomicCAS(intAddress, assumed, __float_as_int(val));
        if (old == assumed)
            *idxAddress = idx;
    } while (old != assumed);
}

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

    // 세 꼭짓점 인덱스
    unsigned int v0 = face.x, v1 = face.y, v2 = face.z;

    // HalfEdgeVertex 및 HalfEdge 인덱스 (faceIdx 기준)
    unsigned int he0 = faceIdx * 3 + 0;
    unsigned int he1 = faceIdx * 3 + 1;
    unsigned int he2 = faceIdx * 3 + 2;

    // HalfEdge 설정
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

    // HalfEdgeFace 설정
    halfEdgeFaces[faceIdx].halfEdgeIndex = he0;

    // Edge map 구축 (항상 "vertex index 쌍"으로!)
    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v0, v1), he0); // v0->v1
    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v1, v2), he1); // v1->v2
    DeviceHalfEdgeMesh::HashMapInsert(info, DeviceHalfEdgeMesh::PackEdge(v2, v0), he2); // v2->v0
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

    // (v0,v1), (v1,v2), (v2,v0) 각각의 반대 edge 찾기
    {
        // he0: v0->v1의 반대는 v1->v0
        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(face.y, face.x);
        unsigned int oppIdx = UINT32_MAX;
        if (DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx))
            halfEdges[he0].oppositeIndex = oppIdx;
    }
    {
        // he1: v1->v2의 반대는 v2->v1
        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(face.z, face.y);
        unsigned int oppIdx = UINT32_MAX;
        if (DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx))
            halfEdges[he1].oppositeIndex = oppIdx;
    }
    {
        // he2: v2->v0의 반대는 v0->v2
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

    // 진입 확인 로그 (일단 출력되는지 체크)
    if (i < 5)
    {
        printf("[Entry] Checking halfEdge[%zu]: vertex = %u, next = %u, opp = %u\n",
            i, he.vertexIndex, he.nextIndex, he.oppositeIndex);
    }

    if (he.oppositeIndex == UINT32_MAX)
    {
        // boundary edge
        if (i < 5)
            printf("[Boundary] halfEdge[%zu] has no opposite\n", i);
        return;
    }

    const HalfEdge& ohe = halfEdges[he.oppositeIndex];

    // 1. Opposite가 맞게 연결되어 있는가?
    if (ohe.oppositeIndex != i)
    {
        printf("[Invalid Opposite] he[%zu] → opp[%u] → opp = %u (expected %zu)\n",
            i, he.oppositeIndex, ohe.oppositeIndex, i);
    }

    // 2. 방향이 맞게 연결되어 있는가?
    uint32_t he_v0 = he.vertexIndex;
    uint32_t he_v1 = halfEdges[he.nextIndex].vertexIndex;

    uint32_t ohe_v0 = ohe.vertexIndex;
    uint32_t ohe_v1 = halfEdges[ohe.nextIndex].vertexIndex;

    if (!(he_v0 == ohe_v1 && he_v1 == ohe_v0))
    {
        printf("[Mismatch] Edge direction mismatch at he[%zu]: %u→%u vs opp[%u]: %u→%u\n",
            i, he_v0, he_v1, he.oppositeIndex, ohe_v0, ohe_v1);
    }

    // 3. Face loop closure (optional)
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

    // atomicMin 방식으로 병렬 충돌 방지
    atomicMin(&vertexToHalfEdge[v], heIdx);
}

__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    unsigned int numberOfHalfEdges,
    float lambda)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    float3 center = positions_in[vid];
    float3 sum = make_float3(0, 0, 0);
    unsigned int count = 0;

    for (unsigned int i = 0; i < numberOfHalfEdges; ++i)
    {
        if (halfEdges[i].vertexIndex != vid) continue;

        int opp = halfEdges[i].oppositeIndex;
        if (opp == UINT32_MAX) continue;

        int neighbor = halfEdges[opp].vertexIndex;
        if (neighbor == vid || neighbor == UINT32_MAX) continue;

        sum += positions_in[neighbor];
        count++;
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


__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmoothNRing(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int nRing,
    float lambda)
{
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= numberOfPoints) return;

    int startHE = vertexToHalfEdge[threadId];
    if (startHE == UINT32_MAX) {
        positions_out[threadId] = positions_in[threadId];
        return;
    }

    // Ring traversal
    constexpr int MAX_RING = 64;
    unsigned int neighborIndices[MAX_RING];
    unsigned int currentFront[MAX_RING];
    unsigned int nextFront[MAX_RING];
    unsigned int currentCount = 0, nextCount = 0;
    bool visited[MAX_RING] = {};

    // Initialize front with 1-ring neighbors
    int he = startHE;
    do {
        int oppHE = halfEdges[he].oppositeIndex;
        if (oppHE == UINT32_MAX) break;
        int neighborVertex = halfEdges[oppHE].vertexIndex;

        if (currentCount < MAX_RING && !visited[neighborVertex]) {
            neighborIndices[currentCount] = neighborVertex;
            currentFront[currentCount] = neighborVertex;
            visited[neighborVertex] = true;
            ++currentCount;
        }

        he = halfEdges[halfEdges[oppHE].nextIndex].nextIndex;
    } while (he != startHE);

    // n-ring propagation
    for (unsigned int ring = 1; ring < nRing; ++ring) {
        nextCount = 0;
        for (unsigned int i = 0; i < currentCount; ++i) {
            int currentVertex = currentFront[i];
            int he2 = vertexToHalfEdge[currentVertex];
            if (he2 == UINT32_MAX) continue;

            int heLoop = he2;
            do {
                int oppHE = halfEdges[heLoop].oppositeIndex;
                if (oppHE == UINT32_MAX) break;
                int neighborVertex = halfEdges[oppHE].vertexIndex;

                if (nextCount < MAX_RING && !visited[neighborVertex]) {
                    neighborIndices[currentCount + nextCount] = neighborVertex;
                    nextFront[nextCount] = neighborVertex;
                    visited[neighborVertex] = true;
                    ++nextCount;
                }

                heLoop = halfEdges[halfEdges[oppHE].nextIndex].nextIndex;
            } while (heLoop != he2);
        }

        for (unsigned int i = 0; i < nextCount; ++i)
            currentFront[i] = nextFront[i];
        currentCount += nextCount;
    }

    float3 sum = make_float3(0, 0, 0);
    int valid = 0;
    for (unsigned int i = 0; i < currentCount; ++i)
    {
        if (neighborIndices[i] < numberOfPoints)
        {
            sum += positions_in[neighborIndices[i]];
            ++valid;
        }
    }

    float3 self = positions_in[threadId];
    if (valid > 0)
    {
        float3 avg = sum / (float)valid;
        float3 displacement = avg - self;
        positions_out[threadId] = self + lambda * displacement;
    }
    else
    {
        positions_out[threadId] = self;
    }
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
    if (DeviceHalfEdgeMesh::RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
    {
        // 원자적 최소 t 갱신 (여러 thread가 동시에 hit할 수 있음)
        DeviceHalfEdgeMesh::atomicMinF(outHitT, t, outHitIndex, i);
    }
}
