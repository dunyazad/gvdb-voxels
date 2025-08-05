#include <ThrustHalfEdgeMesh.cuh>
#include <HashMap.hpp>

ThrustHostHalfEdgeMesh::ThrustHostHalfEdgeMesh()
{
}

ThrustHostHalfEdgeMesh::ThrustHostHalfEdgeMesh(const ThrustHostHalfEdgeMesh& other)
{
    this->operator=(other);
}

ThrustHostHalfEdgeMesh& ThrustHostHalfEdgeMesh::operator=(const ThrustHostHalfEdgeMesh& other)
{
    numberOfPoints = other.numberOfPoints;
    positions = other.positions;
    normals = other.normals;
    colors = other.colors;

    numberOfFaces = other.numberOfFaces;
    faces = other.faces;

    halfEdges = other.halfEdges;
    halfEdgeFaces = other.halfEdgeFaces;
    vertexToHalfEdge = other.vertexToHalfEdge;
    return *this;
}

ThrustHostHalfEdgeMesh::ThrustHostHalfEdgeMesh(const ThrustDeviceHalfEdgeMesh& other)
{
    this->operator=(other);
}

ThrustHostHalfEdgeMesh& ThrustHostHalfEdgeMesh::operator=(const ThrustDeviceHalfEdgeMesh& other)
{
    numberOfPoints = other.numberOfPoints;
    positions = other.positions;
    normals = other.normals;
    colors = other.colors;

    numberOfFaces = other.numberOfFaces;
    faces = other.faces;

    halfEdges = other.halfEdges;
    halfEdgeFaces = other.halfEdgeFaces;
    vertexToHalfEdge = other.vertexToHalfEdge;
    return *this;
}

void ThrustHostHalfEdgeMesh::Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
    this->numberOfPoints = numberOfPoints;
    positions.resize(numberOfPoints);
    normals.resize(numberOfPoints);
    colors.resize(numberOfPoints);

    this->numberOfFaces = numberOfFaces;
    faces.resize(numberOfFaces);

    halfEdges.resize(numberOfFaces * 3);
    halfEdgeFaces.resize(numberOfFaces);
    vertexToHalfEdge.resize(numberOfPoints, UINT32_MAX);
}

void ThrustHostHalfEdgeMesh::Terminate()
{
    numberOfPoints = 0;
    positions.clear();
    normals.clear();
    colors.clear();

    numberOfFaces = 0;
    faces.clear();

    halfEdges.clear();
    halfEdgeFaces.clear();
    vertexToHalfEdge.clear();
}

void ThrustHostHalfEdgeMesh::CopyFromDevice(const ThrustDeviceHalfEdgeMesh& deviceMesh)
{
    numberOfPoints = deviceMesh.numberOfPoints;
    positions = deviceMesh.positions;
    normals = deviceMesh.normals;
    colors = deviceMesh.colors;

    numberOfFaces = deviceMesh.numberOfFaces;
    faces = deviceMesh.faces;

    //halfEdges = deviceMesh.halfEdges;
    //halfEdgeFaces = deviceMesh.halfEdgeFaces;
    //vertexToHalfEdge = deviceMesh.vertexToHalfEdge;
}

void ThrustHostHalfEdgeMesh::CopyToDevice(ThrustDeviceHalfEdgeMesh& deviceMesh) const
{
    deviceMesh.numberOfPoints = numberOfPoints;
    deviceMesh.positions = positions;
    deviceMesh.normals = normals;
    deviceMesh.colors = colors;

    deviceMesh.numberOfFaces = numberOfFaces;
    deviceMesh.faces = faces;

    deviceMesh.halfEdges = halfEdges;
    deviceMesh.halfEdgeFaces = halfEdgeFaces;
    deviceMesh.vertexToHalfEdge = vertexToHalfEdge;
}

uint64_t ThrustHostHalfEdgeMesh::PackEdge(unsigned int v0, unsigned int v1)
{
    return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
}

bool ThrustHostHalfEdgeMesh::PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const
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

void ThrustHostHalfEdgeMesh::BuildHalfEdges()
{
    if (numberOfFaces == 0)
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

    vertexToHalfEdge.clear();
    if (numberOfPoints != vertexToHalfEdge.size()) vertexToHalfEdge.resize(numberOfPoints);
    for (unsigned int i = 0; i < numberOfPoints; ++i)
        vertexToHalfEdge[i] = UINT32_MAX;

    for (unsigned int i = 0; i < numberOfFaces * 3; ++i)
    {
        unsigned int v = halfEdges[i].vertexIndex;
        if (v < numberOfPoints && vertexToHalfEdge[v] == UINT32_MAX)
            vertexToHalfEdge[v] = i;
    }
}

bool ThrustHostHalfEdgeMesh::SerializePLY(const std::string& filename, bool useAlpha)
{
    PLYFormat ply;

    for (unsigned int i = 0; i < numberOfPoints; ++i)
    {
        ply.AddPoint(positions[i].x, positions[i].y, positions[i].z);
        if (0 != normals.size()) ply.AddNormal(normals[i].x, normals[i].y, normals[i].z);
        if (0 != colors.size())
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
        const ThrustHalfEdge& e0 = halfEdges[he0];
        const ThrustHalfEdge& e1 = halfEdges[e0.nextIndex];
        const ThrustHalfEdge& e2 = halfEdges[e1.nextIndex];
        // e0.faceIndex == e1.faceIndex == e2.faceIndex == f

        ply.AddFace(e0.vertexIndex, e1.vertexIndex, e2.vertexIndex);
    }

    return ply.Serialize(filename);
}

bool ThrustHostHalfEdgeMesh::DeserializePLY(const std::string& filename)
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

    if (0 != normals.size() && !ply.GetNormals().empty())
    {
        const auto& ns = ply.GetNormals();
        for (unsigned int i = 0; i < n; ++i)
        {
            normals[i].x = ns[i * 3 + 0];
            normals[i].y = ns[i * 3 + 1];
            normals[i].z = ns[i * 3 + 2];
        }
    }

    if (0 != colors.size() && !ply.GetColors().empty())
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

std::vector<unsigned int> ThrustHostHalfEdgeMesh::GetOneRingVertices(unsigned int v) const
{
    std::vector<unsigned int> neighbors;
    
    return neighbors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

ThrustDeviceHalfEdgeMesh::ThrustDeviceHalfEdgeMesh()
{
}

ThrustDeviceHalfEdgeMesh::ThrustDeviceHalfEdgeMesh(const ThrustHostHalfEdgeMesh& other)
{
    this->operator=(other);
}

ThrustDeviceHalfEdgeMesh& ThrustDeviceHalfEdgeMesh::operator=(const ThrustHostHalfEdgeMesh& other)
{
    numberOfPoints = other.numberOfPoints;
    positions = other.positions;
    normals = other.normals;
    colors = other.colors;

    numberOfFaces = other.numberOfFaces;
    faces = other.faces;

    halfEdges = other.halfEdges;
    halfEdgeFaces = other.halfEdgeFaces;
    vertexToHalfEdge = other.vertexToHalfEdge;
    return *this;
}

ThrustDeviceHalfEdgeMesh::ThrustDeviceHalfEdgeMesh(const ThrustDeviceHalfEdgeMesh& other)
{
    this->operator=(other);
}

ThrustDeviceHalfEdgeMesh& ThrustDeviceHalfEdgeMesh::operator=(const ThrustDeviceHalfEdgeMesh& other)
{
    numberOfPoints = other.numberOfPoints;
    positions = other.positions;
    normals = other.normals;
    colors = other.colors;

    numberOfFaces = other.numberOfFaces;
    faces = other.faces;

    halfEdges = other.halfEdges;
    halfEdgeFaces = other.halfEdgeFaces;
    vertexToHalfEdge = other.vertexToHalfEdge;
    return *this;
}

void ThrustDeviceHalfEdgeMesh::Initialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
    this->numberOfPoints = numberOfPoints;
    positions.resize(numberOfPoints);
    normals.resize(numberOfPoints);
    colors.resize(numberOfPoints);
    
    this->numberOfFaces = numberOfFaces;
    faces.resize(numberOfFaces);
    
    halfEdges.resize(numberOfFaces * 3);
    halfEdgeFaces.resize(numberOfFaces);
    vertexToHalfEdge.resize(numberOfPoints, UINT32_MAX);
}

void ThrustDeviceHalfEdgeMesh::Terminate()
{
    numberOfPoints = 0;
    positions.clear();
    normals.clear();
    colors.clear();

    numberOfFaces = 0;
    faces.clear();

    halfEdges.clear();
    halfEdgeFaces.clear();
    vertexToHalfEdge.clear();
}

void ThrustDeviceHalfEdgeMesh::CopyFromHost(const ThrustHostHalfEdgeMesh& hostMesh)
{
    numberOfPoints = hostMesh.numberOfPoints;
    positions = hostMesh.positions;
    normals = hostMesh.normals;
    colors = hostMesh.colors;

    numberOfFaces = hostMesh.numberOfFaces;
    faces = hostMesh.faces;

    halfEdges = hostMesh.halfEdges;
    halfEdgeFaces = hostMesh.halfEdgeFaces;
    vertexToHalfEdge = hostMesh.vertexToHalfEdge;
}

void ThrustDeviceHalfEdgeMesh::CopyToHost(ThrustHostHalfEdgeMesh& hostMesh) const
{
    hostMesh.numberOfPoints = numberOfPoints;
    hostMesh.positions = positions;
    hostMesh.normals = normals;
    hostMesh.colors = colors;

    hostMesh.numberOfFaces = numberOfFaces;
    hostMesh.faces = faces;

    hostMesh.halfEdges = halfEdges;
    hostMesh.halfEdgeFaces = halfEdgeFaces;
    hostMesh.vertexToHalfEdge = vertexToHalfEdge;
}

__device__ uint64_t ThrustDeviceHalfEdgeMesh::PackEdge(unsigned int v0, unsigned int v1)
{
    return (static_cast<uint64_t>(v0) << 32) | static_cast<uint64_t>(v1);
}

bool ThrustDeviceHalfEdgeMesh::PickFace(const float3& rayOrigin, const float3& rayDir, int& outHitIndex, float& outHitT) const
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

void ThrustDeviceHalfEdgeMesh::BuildHalfEdges()
{
    CUDA_TS(BuildHalfEdges);

    using uint2 = thrust::tuple<unsigned int, unsigned int>;

    const unsigned int numFaces = numberOfFaces;
    const unsigned int numHalfEdges = numFaces * 3;

    thrust::device_vector<uint2> edge_keys(numHalfEdges);
    thrust::device_vector<unsigned int> he_indices(numHalfEdges);

    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numFaces),
        [faces_ptr = faces.data().get(),
        edge_keys_ptr = edge_keys.data().get(),
        he_indices_ptr = he_indices.data().get()] __device__(unsigned int f)
    {
        uint3 tri = faces_ptr[f];
        edge_keys_ptr[f * 3 + 0] = thrust::make_tuple(tri.x, tri.y);
        he_indices_ptr[f * 3 + 0] = f * 3 + 0;
        edge_keys_ptr[f * 3 + 1] = thrust::make_tuple(tri.y, tri.z);
        he_indices_ptr[f * 3 + 1] = f * 3 + 1;
        edge_keys_ptr[f * 3 + 2] = thrust::make_tuple(tri.z, tri.x);
        he_indices_ptr[f * 3 + 2] = f * 3 + 2;
    });

    thrust::device_vector<uint2> edge_keys_sorted = edge_keys;
    thrust::device_vector<unsigned int> he_indices_sorted = he_indices;
    thrust::sort_by_key(edge_keys_sorted.begin(), edge_keys_sorted.end(), he_indices_sorted.begin());

    thrust::device_vector<uint2> reversed_keys(numHalfEdges);
    thrust::transform(
        edge_keys.begin(), edge_keys.end(),
        reversed_keys.begin(),
        [] __device__(const uint2 & e)
    {
        return thrust::make_tuple(thrust::get<1>(e), thrust::get<0>(e));
    });

    thrust::device_vector<unsigned int> opp_sorted_idx(numHalfEdges);
    thrust::lower_bound(
        edge_keys_sorted.begin(), edge_keys_sorted.end(),
        reversed_keys.begin(), reversed_keys.end(),
        opp_sorted_idx.begin());

    auto faces_ptr = faces.data().get();
    auto halfEdges_ptr = halfEdges.data().get();
    auto halfEdgeFaces_ptr = halfEdgeFaces.data().get();
    auto edge_keys_ptr = edge_keys.data().get();
    auto he_indices_ptr = he_indices.data().get();
    auto edge_keys_sorted_ptr = edge_keys_sorted.data().get();
    auto he_indices_sorted_ptr = he_indices_sorted.data().get();
    auto opp_sorted_idx_ptr = opp_sorted_idx.data().get();

    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numFaces),
        [faces_ptr, halfEdges_ptr, halfEdgeFaces_ptr] __device__(unsigned int f)
    {
        uint3 tri = faces_ptr[f];
        unsigned int base = f * 3;

        halfEdges_ptr[base + 0].faceIndex = f;
        halfEdges_ptr[base + 1].faceIndex = f;
        halfEdges_ptr[base + 2].faceIndex = f;

        halfEdges_ptr[base + 0].vertexIndex = tri.x;
        halfEdges_ptr[base + 1].vertexIndex = tri.y;
        halfEdges_ptr[base + 2].vertexIndex = tri.z;

        halfEdges_ptr[base + 0].nextIndex = base + 1;
        halfEdges_ptr[base + 1].nextIndex = base + 2;
        halfEdges_ptr[base + 2].nextIndex = base + 0;

        halfEdges_ptr[base + 0].oppositeIndex = UINT32_MAX;
        halfEdges_ptr[base + 1].oppositeIndex = UINT32_MAX;
        halfEdges_ptr[base + 2].oppositeIndex = UINT32_MAX;

        halfEdgeFaces_ptr[f].halfEdgeIndex = base;
    });

    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numHalfEdges),
        [=] __device__(unsigned int i)
    {
        unsigned int opp_idx = opp_sorted_idx_ptr[i];
        unsigned int my_he = he_indices_ptr[i];

        if (opp_idx < numHalfEdges)
        {
            uint2 query = thrust::make_tuple(
                thrust::get<1>(edge_keys_ptr[my_he]),
                thrust::get<0>(edge_keys_ptr[my_he])
            );
            uint2 found = edge_keys_sorted_ptr[opp_idx];
            if (found == query)
            {
                halfEdges_ptr[my_he].oppositeIndex = he_indices_sorted_ptr[opp_idx];
            }
        }
    });

    thrust::fill(vertexToHalfEdge.begin(), vertexToHalfEdge.end(), UINT32_MAX);

    auto vertexToHalfEdge_ptr = vertexToHalfEdge.data().get();

    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numHalfEdges),
        [halfEdges_ptr, vertexToHalfEdge_ptr, numberOfPoints = this->numberOfPoints] __device__(unsigned int h)
    {
        unsigned int v = halfEdges_ptr[h].vertexIndex;
        if (v < numberOfPoints)
        {
            atomicMin(&vertexToHalfEdge_ptr[v], h);
        }
    }
    );

    CUDA_TE(BuildHalfEdges);
}
//
//std::vector<unsigned int> ThrustDeviceHalfEdgeMesh::GetOneRingVertices(unsigned int v)
//{
//    std::vector<unsigned int> result;
//    if (vertexToHalfEdge.size() == 0 || v >= numberOfPoints)
//        return result;
//
//    // const 절대 금지!!
//    auto d_halfEdges = thrust::raw_pointer_cast(halfEdges.data());
//    auto d_vertexToHalfEdge = thrust::raw_pointer_cast(vertexToHalfEdge.data());
//
//    const size_t maxRing = 128;
//    thrust::device_vector<unsigned int> d_neighbors(maxRing, UINT32_MAX);
//    thrust::device_vector<unsigned int> d_count(1, 0);
//
//    // 반드시 [=]로 캡처해도 상관없음 (포인터는 non-const로만 뽑으면 됨)
//    thrust::for_each_n(
//        thrust::device, thrust::counting_iterator<int>(0), 1,
//        [=] __device__(int)
//    {
//        unsigned int ishe = d_vertexToHalfEdge[v];
//        if (ishe == UINT32_MAX)
//            return;
//
//        unsigned int ihe = ishe;
//        unsigned int lihe = ihe;
//        bool borderFound = false;
//        unsigned int outCount = 0;
//
//        // CW traversal
//        do
//        {
//            const ThrustHalfEdge& he = d_halfEdges[ihe];
//            if (he.nextIndex == UINT32_MAX) break;
//
//            unsigned int ioe = he.oppositeIndex;
//            if (ioe == UINT32_MAX)
//            {
//                lihe = ihe;
//                borderFound = true;
//                break;
//            }
//            const ThrustHalfEdge& oe = d_halfEdges[ioe];
//            if (oe.vertexIndex == UINT32_MAX)
//                break;
//
//            if (outCount < maxRing)
//                d_neighbors[outCount++] = oe.vertexIndex;
//
//            if (oe.nextIndex == UINT32_MAX)
//                break;
//
//            ihe = oe.nextIndex;
//        } while (ihe != ishe && ihe != UINT32_MAX);
//
//        // CCW (border case)
//        if (borderFound)
//        {
//            outCount = 0;
//            ihe = lihe;
//            do
//            {
//                const ThrustHalfEdge& he = d_halfEdges[ihe];
//                if (he.nextIndex == UINT32_MAX)
//                    break;
//
//                const ThrustHalfEdge& ne = d_halfEdges[he.nextIndex];
//                if (ne.nextIndex == UINT32_MAX)
//                    break;
//
//                if (ihe == lihe && outCount < maxRing)
//                    d_neighbors[outCount++] = ne.vertexIndex;
//
//                const ThrustHalfEdge& pe = d_halfEdges[ne.nextIndex];
//                if (outCount < maxRing)
//                    d_neighbors[outCount++] = pe.vertexIndex;
//
//                ihe = pe.oppositeIndex;
//            } while (ihe != lihe && ihe != UINT32_MAX);
//        }
//        d_count[0] = outCount;
//    }
//    );
//
//    unsigned int neighborCount = 0;
//    thrust::copy_n(d_count.begin(), 1, &neighborCount);
//
//    if (neighborCount > maxRing) neighborCount = maxRing;
//    result.resize(neighborCount);
//    if (neighborCount > 0)
//        thrust::copy_n(d_neighbors.begin(), neighborCount, result.begin());
//
//    return result;
//}

__device__ bool ThrustDeviceHalfEdgeMesh::HashMapInsert(HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int value)
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
__device__ bool ThrustDeviceHalfEdgeMesh::HashMapFind(const HashMapInfo<uint64_t, unsigned int>& info, uint64_t key, unsigned int* outValue)
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
