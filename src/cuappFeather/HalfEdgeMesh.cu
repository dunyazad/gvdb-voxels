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
    }
    else
    {
        faces = nullptr;
        halfEdges = nullptr;
        halfEdgeFaces = nullptr;
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

    positions = nullptr;
    normals = nullptr;
    colors = nullptr;
    faces = nullptr;
    halfEdges = nullptr;
    halfEdgeFaces = nullptr;
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

    CUDA_SYNC();
}

void HostHalfEdgeMesh::BuildHalfEdges()
{
    if (numberOfFaces == 0 || faces == nullptr)
    {
        return;
    }

    // halfEdges, halfEdgeFaces 메모리 할당/재할당 (이미 Initialize에서 처리됨)
    // 기존 할당된 것이 있다고 가정

    // edge map: key = PackEdge(v0, v1), value = halfEdge index
    // std::unordered_map<uint64_t, unsigned int>를 사용
    std::unordered_map<uint64_t, unsigned int> edgeMap;

    // 1. 각 face마다 3개의 halfedge 생성 및 edgeMap 채우기
    for (unsigned int f = 0; f < numberOfFaces; ++f)
    {
        const uint3& tri = faces[f];
        unsigned int baseIdx = f * 3;
        unsigned int heIdx[3] = { baseIdx, baseIdx + 1, baseIdx + 2 };

        for (int i = 0; i < 3; ++i)
        {
            unsigned int v0, v1;
            if (i == 0) { v0 = tri.x; v1 = tri.y; }
            else if (i == 1) { v0 = tri.y; v1 = tri.z; }
            else { v0 = tri.z; v1 = tri.x; }

            HalfEdge& he = halfEdges[heIdx[i]];
            he.vertexIndex = v1;
            he.faceIndex = f;
            he.nextIndex = heIdx[(i + 1) % 3];
            he.oppositeIndex = UINT32_MAX;

            uint64_t edgeKey = (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
            edgeMap[edgeKey] = heIdx[i];
        }

        halfEdgeFaces[f].halfEdgeIndex = baseIdx;
    }

    // 2. opposite 링크 생성
    for (unsigned int f = 0; f < numberOfFaces; ++f)
    {
        unsigned int baseIdx = f * 3;
        const uint3& tri = faces[f];

        for (int i = 0; i < 3; ++i)
        {
            unsigned int currHeIdx = baseIdx + i;
            unsigned int v0, v1;
            if (i == 0) { v0 = tri.x; v1 = tri.y; }
            else if (i == 1) { v0 = tri.y; v1 = tri.z; }
            else { v0 = tri.z; v1 = tri.x; }

            uint64_t oppKey = (static_cast<uint64_t>(v1) << 32) | static_cast<uint64_t>(v0);

            auto it = edgeMap.find(oppKey);
            if (it != edgeMap.end())
            {
                halfEdges[currHeIdx].oppositeIndex = it->second;
            }
        }
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
        cudaMalloc(&vertexToHalfEdge, sizeof(unsigned int) * numberOfPoints);
    }
    else
    {
        positions = nullptr;
        normals = nullptr;
        colors = nullptr;
        vertexToHalfEdge = nullptr;
    }
    if (numberOfFaces > 0)
    {
        cudaMalloc(&faces, sizeof(uint3) * numberOfFaces);
        cudaMalloc(&halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
        cudaMalloc(&halfEdgeFaces, sizeof(HalfEdgeFace) * numberOfFaces);
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
    CUDA_SYNC();

    edgeMap.Terminate();

    // ========== vertexToHalfEdge 세팅 ==========
    // (호스트에서 한번만, 커널 이후에 복사)
    std::vector<HalfEdge> h_halfEdges(numHalfEdges);
    CUDA_COPY_D2H(h_halfEdges.data(), halfEdges, sizeof(HalfEdge) * numHalfEdges);

    std::vector<unsigned int> h_vertexToHalfEdge(numberOfPoints, UINT32_MAX);
    for (unsigned int i = 0; i < numHalfEdges; ++i)
    {
        unsigned int v = h_halfEdges[i].vertexIndex;
        if (v < numberOfPoints && h_vertexToHalfEdge[v] == UINT32_MAX)
        {
            h_vertexToHalfEdge[v] = i;
        }
    }
    CUDA_COPY_H2D(vertexToHalfEdge, h_vertexToHalfEdge.data(), sizeof(unsigned int) * numberOfPoints);
    CUDA_SYNC();
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

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_LaplacianSmooth, numberOfPoints,
            halfEdges, vertexToHalfEdge, positionsA, positionsB, numberOfPoints, lambda);
        CUDA_SYNC();
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
    if (numberOfPoints == 0 || numberOfFaces == 0) return;

    float3* positionsA = positions;
    float3* positionsB;
    cudaMalloc(&positionsB, sizeof(float3) * numberOfPoints);

    for (unsigned int it = 0; it < iterations; ++it)
    {
        LaunchKernel(Kernel_LaplacianSmoothNRing, numberOfPoints,
            halfEdges, vertexToHalfEdge, positionsA, positionsB, numberOfPoints, lambda, nRing);
        CUDA_SYNC();
        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
    {
        CUDA_COPY_D2D(positions, positionsA, sizeof(float3) * numberOfPoints);
    }
    cudaFree(positionsB);
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
    HalfEdgeFace* outFaces,
    HashMapInfo<uint64_t, unsigned int> info)
{
    unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numberOfFaces) return;

    uint3 tri = faces[f];
    unsigned int baseIdx = f * 3;
    unsigned int heIdx[3] = { baseIdx, baseIdx + 1, baseIdx + 2 };

    for (int i = 0; i < 3; ++i)
    {
        unsigned int v0, v1;
        if (i == 0) { v0 = tri.x; v1 = tri.y; }
        else if (i == 1) { v0 = tri.y; v1 = tri.z; }
        else { v0 = tri.z; v1 = tri.x; }

        HalfEdge he;
        he.vertexIndex = v1;
        he.faceIndex = f;
        he.nextIndex = heIdx[(i + 1) % 3];
        he.oppositeIndex = UINT32_MAX;

        halfEdges[heIdx[i]] = he;

        uint64_t edgeKey = DeviceHalfEdgeMesh::PackEdge(v0, v1);
        DeviceHalfEdgeMesh::HashMapInsert(info, edgeKey, heIdx[i]);
    }

    outFaces[f].halfEdgeIndex = baseIdx;
}

__global__ void Kernel_DeviceHalfEdgeMesh_LinkOpposites(
    const uint3* faces,
    unsigned int numberOfFaces,
    HalfEdge* halfEdges,
    HashMapInfo<uint64_t, unsigned int> info)
{
    unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numberOfFaces) return;

    uint3 tri = faces[f];
    unsigned int baseIdx = f * 3;

    for (int i = 0; i < 3; ++i)
    {
        unsigned int currHeIdx = baseIdx + i;
        unsigned int v0, v1;
        if (i == 0) { v0 = tri.x; v1 = tri.y; }
        else if (i == 1) { v0 = tri.y; v1 = tri.z; }
        else { v0 = tri.z; v1 = tri.x; }

        uint64_t oppKey = DeviceHalfEdgeMesh::PackEdge(v1, v0);
        unsigned int oppIdx = UINT32_MAX;
        bool found = DeviceHalfEdgeMesh::HashMapFind(info, oppKey, &oppIdx);
        if (found && oppIdx != UINT32_MAX)
        {
            halfEdges[currHeIdx].oppositeIndex = oppIdx;
        }
    }
}

__global__ void Kernel_LaplacianSmooth(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    float lambda)
{
    unsigned int vtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vtx >= numberOfPoints) return;

    unsigned int startHe = vertexToHalfEdge[vtx];
    if (startHe == UINT32_MAX)
    {
        positions_out[vtx] = positions_in[vtx];
        return;
    }

    float3 sum = make_float3(0, 0, 0);
    int n = 0;
    unsigned int curr = startHe;
    do
    {
        unsigned int neighbor = halfEdges[curr].vertexIndex;
        if (neighbor < numberOfPoints)
        {
            sum += positions_in[neighbor];
            n++;
        }
        unsigned int opp = halfEdges[curr].oppositeIndex;
        if (opp == UINT32_MAX)
            break;
        curr = halfEdges[opp].nextIndex;
    } while (curr != startHe);

    if (n > 0)
    {
        float3 center = sum / float(n);
        float3 orig = positions_in[vtx];
        positions_out[vtx] = orig + (center - orig) * lambda;
    }
    else
    {
        positions_out[vtx] = positions_in[vtx];
    }
}

__global__ void Kernel_LaplacianSmoothNRing(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    float lambda,
    int maxRing)
{
    unsigned int vtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vtx >= numberOfPoints) return;

    // BFS를 위한 큐 (고정 크기)
    const int MAX_NEIGHBORS = 1024;
    unsigned int queue[MAX_NEIGHBORS];
    unsigned int visited[MAX_NEIGHBORS];
    unsigned int ring_level[MAX_NEIGHBORS];

    int front = 0, rear = 0;
    int neighbor_count = 0;

    // 방문체크 및 자기 자신 방문표시
    for (int i = 0; i < MAX_NEIGHBORS; ++i) visited[i] = UINT32_MAX;
    visited[0] = vtx;
    queue[0] = vtx;
    ring_level[0] = 0;
    rear = 1;

    while (front < rear && neighbor_count < MAX_NEIGHBORS)
    {
        unsigned int curr_vtx = queue[front];
        int curr_ring = ring_level[front];
        front++;

        if (curr_ring >= maxRing) continue;
        unsigned int startHe = vertexToHalfEdge[curr_vtx];
        if (startHe == UINT32_MAX) continue;

        unsigned int currHe = startHe;
        do
        {
            unsigned int nextHe = halfEdges[currHe].nextIndex;
            unsigned int neighbor = halfEdges[nextHe].vertexIndex;
            // 아직 방문 안 했으면 큐에 추가
            bool is_visited = false;
            for (int k = 0; k < rear; ++k)
            {
                if (visited[k] == neighbor) { is_visited = true; break; }
            }
            if (!is_visited && neighbor < numberOfPoints && neighbor != vtx && rear < MAX_NEIGHBORS)
            {
                visited[rear] = neighbor;
                queue[rear] = neighbor;
                ring_level[rear] = curr_ring + 1;
                rear++;
                neighbor_count++;
            }
            unsigned int opp = halfEdges[currHe].oppositeIndex;
            if (opp == UINT32_MAX) break;
            currHe = halfEdges[opp].nextIndex;
        } while (currHe != startHe && neighbor_count < MAX_NEIGHBORS);
    }

    // neighbor(자기 자신 제외) 평균
    float3 accum = make_float3(0, 0, 0);
    int n = 0;
    for (int i = 1; i < rear; ++i) // i=0은 자기 자신
    {
        accum += positions_in[visited[i]];
        n++;
    }

    if (n > 0)
    {
        float3 center = accum / float(n);
        float3 orig = positions_in[vtx];
        positions_out[vtx] = orig + (center - orig) * lambda;
    }
    else
    {
        positions_out[vtx] = positions_in[vtx];
    }
}

__global__ void Kernel_PickFace(
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

__global__ void Kernel_DeviceHalfEdgeMesh_PickFace(
    const float3 rayOrigin,
    const float3 rayDir,
    const float3* positions,
    const uint3* faces,
    unsigned int numberOfFaces,
    int* outHitIndex,
    float* outHitT)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberOfFaces) return;

    const uint3 tri = faces[i];
    const float3 v0 = positions[tri.x];
    const float3 v1 = positions[tri.y];
    const float3 v2 = positions[tri.z];

    float t, u, v;
    if (DeviceHalfEdgeMesh::RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
    {
        DeviceHalfEdgeMesh::atomicMinWithIndex(outHitT, t, outHitIndex, i);
        // barycentric 등 추가 저장은 필요시
    }
}
