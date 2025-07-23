#include <HalfEdgeMesh.cuh>
#include <Mesh.cuh>

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

    positions = nullptr;
    normals = nullptr;
    colors = nullptr;
    faces = nullptr;
    halfEdges = nullptr;
    halfEdgeFaces = nullptr;
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

    // HalfEdge, HalfEdgeFace 메모리는 이미 Initialize에서 할당됨

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
}

void DeviceHalfEdgeMesh::LaplacianSmoothing(int iterations, float lambda)
{
    if (numberOfPoints == 0 || positions == nullptr)
        return;

    float3* d_positions_src = positions;
    float3* d_positions_tmp = nullptr;
    cudaMalloc(&d_positions_tmp, sizeof(float3) * numberOfPoints);

    for (int iter = 0; iter < iterations; ++iter)
    {
        Kernel_LaplacianSmoothing << <(numberOfPoints + 255) / 256, 256 >> > (
            d_positions_src,
            halfEdges,
            numberOfPoints,
            numberOfFaces,
            d_positions_tmp,
            halfEdgeFaces
            );
        CUDA_SYNC();

        if (lambda != 1.0f)
        {
            // pos_out = (1 - lambda) * pos_orig + lambda * pos_smoothed
            Kernel_LaplacianLerp << <(numberOfPoints + 255) / 256, 256 >> > (
                d_positions_tmp, d_positions_src, d_positions_tmp, lambda, numberOfPoints
                );
            CUDA_SYNC();
        }

        // Swap buffers for next iteration (unless last iter)
        if (iter < iterations - 1)
            std::swap(d_positions_src, d_positions_tmp);
    }

    // 마지막 결과를 positions에 복사
    if (d_positions_src != positions)
    {
        CUDA_COPY_D2D(positions, d_positions_src, sizeof(float3) * numberOfPoints);
    }
    cudaFree(d_positions_tmp);
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

__global__ void Kernel_LaplacianLerp(
    float3* dst,
    const float3* src,
    const float3* smoothed,
    float lambda,
    unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = (1.0f - lambda) * src[i] + lambda * smoothed[i];
}

__global__ void Kernel_LaplacianSmoothing(
    const float3* positions,
    const HalfEdge* halfEdges,
    const unsigned int numberOfPoints,
    const unsigned int numberOfFaces,
    float3* positions_out,
    const HalfEdgeFace* halfEdgeFaces
)
{
    unsigned int vtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vtx >= numberOfPoints)
        return;

    float3 center = positions[vtx];
    float3 sum = make_float3(0, 0, 0);
    int neighborCount = 0;

    // 1. Find a halfedge starting from this vertex
    unsigned int startHE = UINT32_MAX;
    for (unsigned int f = 0; f < numberOfFaces; ++f)
    {
        unsigned int base = f * 3;
        for (int i = 0; i < 3; ++i)
        {
            const HalfEdge& he = halfEdges[base + i];
            if (he.vertexIndex == vtx)
            {
                startHE = base + i;
                break;
            }
        }
        if (startHE != UINT32_MAX)
            break;
    }

    if (startHE == UINT32_MAX)
    {
        // Isolated vertex (no incident halfedge)
        positions_out[vtx] = center;
        return;
    }

    unsigned int heIdx = startHE;
    unsigned int curr = heIdx;
    bool first = true;

    // 2. 1-ring traversal (CCW order)
    do
    {
        const HalfEdge& he = halfEdges[curr];
        unsigned int opp = he.oppositeIndex;
        if (opp == UINT32_MAX)
            break; // Boundary

        unsigned int next = halfEdges[opp].nextIndex;
        unsigned int nei_vtx = halfEdges[next].vertexIndex;
        if (nei_vtx != vtx) // avoid self-loop
        {
            sum += positions[nei_vtx];
            ++neighborCount;
        }

        curr = next;
        first = false;
    } while (curr != heIdx && neighborCount < numberOfPoints);

    if (neighborCount > 0)
        positions_out[vtx] = sum / neighborCount;
    else
        positions_out[vtx] = center;
}
