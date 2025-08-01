#include <HalfEdgeMesh.cuh>
#include <Mesh.cuh>

#include <set>

#define MAX_FRONTIER (1 << 16)
#define MAX_RESULT   (1 << 20)

#define MAX_NEIGHBORS 64


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

void HostHalfEdgeMesh::ComputeFeatureWeights(std::vector<float>& featureWeights, float sharpAngleDeg)
{
    featureWeights.resize(numberOfPoints, 1.0f);
    float cosThr = std::cos(sharpAngleDeg * PI / 180.0f);

    for (unsigned int vid = 0; vid < numberOfPoints; ++vid)
    {
        float3 n0 = normals[vid];
        std::vector<unsigned int> neighbors = GetOneRingVertices(vid);

        for (unsigned int nb : neighbors)
        {
            float3 n1 = normals[nb];
            float dotn = n0.x * n1.x + n0.y * n1.y + n0.z * n1.z;
            if (dotn < cosThr)
            {
                featureWeights[vid] = 0.1f; // 샤프엣지, feature
                break;
            }
        }
    }
}

void HostHalfEdgeMesh::BilateralNormalSmoothing(
    std::vector<float3>& outNormals,
    const std::vector<float>& featureWeights,
    float sigma_s,
    float sigma_n)
{
    outNormals.resize(numberOfPoints);

    for (unsigned int vid = 0; vid < numberOfPoints; ++vid)
    {
        float3 vpos = positions[vid];
        float3 vnor = normals[vid];
        float w_sum = 0.0f;
        float3 n_sum = make_float3(0, 0, 0);

        std::vector<unsigned int> neighbors = GetOneRingVertices(vid);
        for (unsigned int nb : neighbors)
        {
            float3 npos = positions[nb];
            float3 nnor = normals[nb];

            float dist2 = length2(vpos - npos);
            float angle2 = length2(vnor - nnor); // 혹은 1 - dot(vnor, nnor)

            float w_spatial = std::exp(-dist2 / (2.0f * sigma_s * sigma_s));
            float w_normal = std::exp(-angle2 / (2.0f * sigma_n * sigma_n));
            float w_feature = std::min(featureWeights[vid], featureWeights[nb]);

            float w = w_spatial * w_normal * w_feature;
            n_sum += nnor * w;
            w_sum += w;
        }

        if (w_sum > 0)
        {
            float3 nout = n_sum / w_sum;
            outNormals[vid] = normalize(nout);
        }
        else
        {
            outNormals[vid] = vnor;
        }
    }
}

void HostHalfEdgeMesh::TangentPlaneProjection(
    std::vector<float3>& outPositions,
    const std::vector<float3>& smoothNormals,
    const std::vector<float>& featureWeights,
    float sigma_proj)
{
    outPositions.resize(numberOfPoints);

    for (unsigned int vid = 0; vid < numberOfPoints; ++vid)
    {
        float3 vpos = positions[vid];
        float3 vnor = smoothNormals[vid];

        float3 p_sum = make_float3(0, 0, 0);
        float w_sum = 0.0f;

        std::vector<unsigned int> neighbors = GetOneRingVertices(vid);
        for (unsigned int nb : neighbors)
        {
            float3 npos = positions[nb];

            // tangent plane projection
            float3 delta = npos - vpos;
            float d = delta.x * vnor.x + delta.y * vnor.y + delta.z * vnor.z;
            float3 proj = npos - vnor * d;

            float w_spatial = std::exp(-length2(delta) / (2.0f * sigma_proj * sigma_proj));
            float w_feature = std::min(featureWeights[vid], featureWeights[nb]);
            float w = w_spatial * w_feature;
            p_sum += proj * w;
            w_sum += w;
        }

        // 본인 좌표 포함(안정성, optional)
        p_sum += vpos;
        w_sum += 1.0f;

        if (w_sum > 0)
            outPositions[vid] = p_sum / w_sum;
        else
            outPositions[vid] = vpos;
    }
}

void HostHalfEdgeMesh::RobustSmooth(int iterations, float sigma_s, float sigma_n, float sigma_proj, float sharpAngleDeg)
{
    std::vector<float> featureWeights(numberOfPoints, 1.0f);
    std::vector<float3> smoothNormals(numberOfPoints);
    std::vector<float3> smoothPositions(numberOfPoints);

    for (int it = 0; it < iterations; ++it)
    {
        ComputeFeatureWeights(featureWeights, sharpAngleDeg);
        BilateralNormalSmoothing(smoothNormals, featureWeights, sigma_s, sigma_n);
        TangentPlaneProjection(smoothPositions, smoothNormals, featureWeights, sigma_proj);

        // apply results
        for (unsigned int vid = 0; vid < numberOfPoints; ++vid)
        {
            normals[vid] = smoothNormals[vid];
            positions[vid] = smoothPositions[vid];
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

std::vector<unsigned int> DeviceHalfEdgeMesh::GetOneRingVertices(unsigned int v, bool fixBorderVertices) const
{
    // 파라미터 체크
    assert(v < numberOfPoints);

    // 디바이스 결과 버퍼 할당
    unsigned int* d_neighbors = nullptr;
    unsigned int* d_count = nullptr;
    cudaMalloc(&d_neighbors, sizeof(unsigned int) * MAX_NEIGHBORS);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_neighbors, 0xFF, sizeof(unsigned int) * MAX_NEIGHBORS);
    cudaMemset(d_count, 0, sizeof(unsigned int));

    // 커널 호출 (단일 스레드, 하나의 vertex만 추출)
    Kernel_DeviceHalfEdgeMesh_OneRing << <1, 1 >> > (
        halfEdges,
        vertexToHalfEdge,
        numberOfPoints,
        v,
        fixBorderVertices,
        d_neighbors,
        d_count
        );
    cudaDeviceSynchronize();

    // 결과 복사
    unsigned int h_count = 0;
    unsigned int h_neighbors[MAX_NEIGHBORS];
    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbors, d_neighbors, sizeof(unsigned int) * MAX_NEIGHBORS, cudaMemcpyDeviceToHost);

    // 정리
    cudaFree(d_neighbors);
    cudaFree(d_count);

    // 결과 생성
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

    // 1. 할당 및 초기화
    CUDA_MALLOC(&d_visited, sizeof(unsigned int) * numberOfPoints);
    CUDA_MEMSET(d_visited, 0, sizeof(unsigned int) * numberOfPoints);
    CUDA_MALLOC(&d_frontier[0], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontier[1], sizeof(unsigned int) * MAX_FRONTIER);
    CUDA_MALLOC(&d_frontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_nextFrontierSize, sizeof(unsigned int));
    CUDA_MALLOC(&d_result, sizeof(unsigned int) * MAX_RESULT);
    CUDA_MALLOC(&d_resultSize, sizeof(unsigned int));
    CUDA_MEMSET(d_resultSize, 0, sizeof(unsigned int));

    // 2. 시작점 초기화
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
            startVertex, radius);   // ← startVertex만 넘김
        CUDA_SYNC();

        CUDA_COPY_D2D(d_frontierSize, d_nextFrontierSize, sizeof(unsigned int));
        iter++;

        // result size overflow 체크
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
    cudaMalloc(&positionsB, sizeof(float3) * numberOfPoints);

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

    cudaFree(toFree);
}

void DeviceHalfEdgeMesh::RadiusLaplacianSmoothing(float radius, unsigned int iterations)
{
    unsigned int* d_neighbors = nullptr;
    unsigned int* d_neighborSizes = nullptr;
    cudaMalloc(&d_neighbors, sizeof(unsigned int) * numberOfPoints * MAX_NEIGHBORS);
    cudaMalloc(&d_neighborSizes, sizeof(unsigned int) * numberOfPoints);

    float3* positionsA = positions;
    float3* positionsB = nullptr;
    cudaMalloc(&positionsB, sizeof(float3) * numberOfPoints);

    for (unsigned int it = 0; it < iterations; ++it)
    {
        // 1. 전체 vertex에 대해 neighbor set 구성
        unsigned int blockSize = 128;
        unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;
        Kernel_GetAllVerticesInRadius << <gridSize, blockSize >> > (
            positionsA,
            numberOfPoints,
            halfEdges,
            vertexToHalfEdge,
            d_neighbors,
            d_neighborSizes,
            MAX_NEIGHBORS,
            radius
            );
        cudaDeviceSynchronize();

        // 2. smoothing (neighbor set 활용)
        Kernel_RadiusLaplacianSmooth_WithNeighbors << <gridSize, blockSize >> > (
            positionsA,
            positionsB,
            numberOfPoints,
            d_neighbors,
            d_neighborSizes,
            MAX_NEIGHBORS
            );
        cudaDeviceSynchronize();
        std::swap(positionsA, positionsB);
    }

    if (positionsA != positions)
        cudaMemcpy(positions, positionsA, sizeof(float3) * numberOfPoints, cudaMemcpyDeviceToDevice);

    cudaFree(positionsB);
    cudaFree(d_neighbors);
    cudaFree(d_neighborSizes);
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
bool DeviceHalfEdgeMesh::RayTriangleIntersect(
    const float3& orig, const float3& dir,
    const float3& v0, const float3& v1, const float3& v2,
    float& t, float& u, float& v)
{
    const float EPSILON = 1e-6f;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (fabs(a) < EPSILON) return false;
    float f = 1.0f / a;
    float3 s = orig - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross(s, edge1);
    v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * dot(edge2, q);
    return t > EPSILON;
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

    // CW traversal
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

    // CCW traversal (border case)
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
        // boundary edge
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

    // atomicMin 방식으로 병렬 충돌 방지
    atomicMin(&vertexToHalfEdge[v], heIdx);
}
//
//__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
//    float3* positions_in,
//    float3* positions_out,
//    unsigned int numberOfPoints,
//    const HalfEdge* halfEdges,
//    unsigned int numberOfHalfEdges,
//    unsigned int* vertexToHalfEdge,
//    float lambda)
//{
//    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (vid >= numberOfPoints) return;
//
//    /*
//    float3 center = positions_in[vid];
//    float3 sum = make_float3(0, 0, 0);
//    unsigned int count = 0;
//
//    for (unsigned int i = 0; i < numberOfHalfEdges; ++i)
//    {
//        if (halfEdges[i].vertexIndex != vid) continue;
//
//        int opp = halfEdges[i].oppositeIndex;
//        if (opp == UINT32_MAX) continue;
//
//        int neighbor = halfEdges[opp].vertexIndex;
//        if (neighbor == vid || neighbor == UINT32_MAX) continue;
//
//        sum += positions_in[neighbor];
//        count++;
//    }
//
//    if (count > 0)
//    {
//        float3 avg = sum / (float)count;
//        positions_out[vid] = center + lambda * (avg - center);
//    }
//    else
//    {
//        positions_out[vid] = center;
//    }
//    */
//
//    unsigned int heStart = vertexToHalfEdge[vid];
//    if (UINT32_MAX == heStart) return;
//
//    float3 center = positions_in[vid];
//    
//    if (UINT32_MAX == halfEdges[heStart].oppositeIndex)
//    {
//        positions_out[vid] = center;
//        return;
//    }
//
//    float3 sum = make_float3(0, 0, 0);
//    unsigned int count = 0;
//
//    unsigned int he = heStart;
//    do
//    {
//        int opp = halfEdges[he].oppositeIndex;
//        if (opp == UINT32_MAX)
//        {
//            positions_out[vid] = center;
//            return;
//        }
//
//        unsigned int neighbor = halfEdges[opp].vertexIndex;
//        if (neighbor >= numberOfPoints)
//        {
//            positions_out[vid] = center;
//            return;
//        }
//
//        sum += positions_in[neighbor];
//        count++;
//
//        he = halfEdges[opp].nextIndex;
//
//    } while (he != heStart);
//
//    if (count > 0)
//    {
//        float3 avg = sum / (float)count;
//        positions_out[vid] = center + lambda * (avg - center);
//    }
//    else
//    {
//        positions_out[vid] = center;
//    }
//}


__global__ void Kernel_DeviceHalfEdgeMesh_LaplacianSmooth(
    float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    bool fixeborderVertices,
    const HalfEdge* halfEdges,
    unsigned int numberOfHalfEdges,
    unsigned int* vertexToHalfEdge,
    float lambda)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    // 1-ring 이웃 인덱스 추출
    unsigned int neighbors[MAX_NEIGHBORS];
    unsigned int nCount = 0;

    DeviceHalfEdgeMesh::GetOneRingVertices_Device(
        vid,
        halfEdges,
        vertexToHalfEdge,
        numberOfPoints,
        fixeborderVertices,
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
    if (DeviceHalfEdgeMesh::RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, u, v))
    {
        DeviceHalfEdgeMesh::atomicMinF(outHitT, t, outHitIndex, i);
    }
}

__global__ void Kernel_DeviceHalfEdgeMesh_OneRing(
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int numberOfPoints,
    unsigned int v, // 조사할 vertex index
    bool fixBorderVertices, 
    unsigned int* outBuffer, // outBuffer[0:count-1]에 결과 저장
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

    // 1-ring neighbors (최대 32개까지)
    unsigned int neighbors[32];
    unsigned int nCount = 0;
    DeviceHalfEdgeMesh::GetOneRingVertices_Device(
        v, halfEdges, vertexToHalfEdge, numberOfPoints, false, neighbors, nCount, 32);

    for (unsigned int i = 0; i < nCount; ++i)
    {
        unsigned int nb = neighbors[i];
        if (nb == UINT32_MAX || nb >= numberOfPoints)
            continue;
        // 방문 체크 (visited[nb] == 0이면 방문X)
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

__global__ void Kernel_GetAllVerticesInRadius(
    const float3* positions,
    unsigned int numberOfPoints,
    const HalfEdge* halfEdges,
    const unsigned int* vertexToHalfEdge,
    unsigned int* allNeighbors,   // [numberOfPoints * MAX_NEIGHBORS]
    unsigned int* allNeighborSizes, // [numberOfPoints]
    unsigned int maxNeighbors,
    float radius
)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    // 각 vertex마다 별도의 output 버퍼/카운터
    unsigned int* neighbors = allNeighbors + vid * maxNeighbors;
    unsigned int* neighborSize = allNeighborSizes + vid;

    // 초기화
    for (unsigned int i = 0; i < maxNeighbors; ++i)
        neighbors[i] = UINT32_MAX;
    *neighborSize = 0;

    // Frontier, visited, nextFrontier 등 임시 버퍼 (로컬 변수, 크기 한정)
    unsigned int frontier[MAX_FRONTIER];
    unsigned int visited[MAX_FRONTIER];
    unsigned int nextFrontier[MAX_FRONTIER];
    unsigned int nextFrontierSize = 0;
    unsigned int result[MAX_NEIGHBORS];
    unsigned int resultSize = 0;

    // 시작점 넣기
    unsigned int startVertex = vid;
    unsigned int h_frontierSize = 1;
    frontier[0] = startVertex;
    visited[0] = startVertex;
    unsigned int visitedCount = 1;
    result[0] = startVertex;
    resultSize = 1;

    float3 startPos = positions[startVertex];

    for (unsigned int iter = 0; h_frontierSize > 0 && resultSize < maxNeighbors; ++iter)
    {
        nextFrontierSize = 0;
        for (unsigned int fi = 0; fi < h_frontierSize; ++fi)
        {
            unsigned int v = frontier[fi];
            // 1-ring neighbor 찾기
            unsigned int neighbors1ring[32];
            unsigned int nCount = 0;
            DeviceHalfEdgeMesh::GetOneRingVertices_Device(
                v, halfEdges, vertexToHalfEdge, numberOfPoints, false, neighbors1ring, nCount, 32);

            for (unsigned int ni = 0; ni < nCount; ++ni)
            {
                unsigned int nb = neighbors1ring[ni];
                if (nb == UINT32_MAX || nb >= numberOfPoints) continue;
                // 방문 여부
                bool alreadyVisited = false;
                for (unsigned int vi = 0; vi < visitedCount; ++vi)
                    if (visited[vi] == nb) { alreadyVisited = true; break; }
                if (alreadyVisited) continue;
                // 거리 제한
                float dist2 = length2(positions[nb] - startPos);
                if (dist2 <= radius * radius)
                {
                    if (visitedCount < MAX_FRONTIER)
                    {
                        visited[visitedCount++] = nb;
                        if (resultSize < maxNeighbors)
                            result[resultSize++] = nb;
                        if (nextFrontierSize < MAX_FRONTIER)
                            nextFrontier[nextFrontierSize++] = nb;
                    }
                }
            }
        }
        if (nextFrontierSize == 0) break;
        for (unsigned int i = 0; i < nextFrontierSize; ++i)
            frontier[i] = nextFrontier[i];
        h_frontierSize = nextFrontierSize;
    }

    // 결과 저장
    unsigned int outCount = min(resultSize, maxNeighbors);
    for (unsigned int i = 0; i < outCount; ++i)
        neighbors[i] = result[i];
    *neighborSize = outCount;
}

__global__ void Kernel_RadiusLaplacianSmooth_WithNeighbors(
    const float3* positions_in,
    float3* positions_out,
    unsigned int numberOfPoints,
    const unsigned int* allNeighbors,   // [numberOfPoints * MAX_NEIGHBORS]
    const unsigned int* allNeighborSizes, // [numberOfPoints]
    unsigned int maxNeighbors)
{
    unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numberOfPoints) return;

    const unsigned int* neighbors = allNeighbors + vid * maxNeighbors;
    unsigned int nCount = allNeighborSizes[vid];

    float3 center = positions_in[vid];
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
        positions_out[vid] = (sum / (float)count);
    else
        positions_out[vid] = center;
}
