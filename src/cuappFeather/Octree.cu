#include <Octree.cuh>

#pragma region Octree
__host__ __device__ uint64_t Octree::GetDepth(uint64_t key)
{
    return (key >> 58) & 0x3Full;
}

__host__ __device__ uint64_t Octree::SetDepth(uint64_t key, uint64_t depth)
{
    return (key & ~(0x3Full << 58)) | ((depth & 0x3Full) << 58);
}

__host__ __device__ uint64_t Octree::GetCode(uint64_t key, uint64_t level)
{
    auto shift = level * 3ull;
    return (shift >= 58ull) ? 0ull : ((key >> shift) & 0x7ull);
}

__host__ __device__ uint64_t Octree::SetCode(uint64_t key, uint64_t level, uint64_t x, uint64_t y, uint64_t z)
{
    auto code3 = (x & 1ull) << 2 | (y & 1ull) << 1 | (z & 1ull) << 0;

    auto shift = level * 3ull;
    if (shift >= 58ull) return key;

    auto mask = 0x7ull << shift;
    auto cleared = key & ~mask;
    return cleared | ((code3 & 0x7ull) << shift);
}

__host__ __device__ uint64_t Octree::GetParentKey(uint64_t key)
{
    auto depth = GetDepth(key);
    if (depth == 0) return key;

    auto newDepth = depth - 1;
    auto shift = (newDepth * 3ull);

    auto cleared = key & ~(0x7ull << shift);
    return SetDepth(cleared, newDepth);
}

__host__ __device__ uint64_t Octree::GetChildKey(uint64_t key, uint64_t bx, uint64_t by, uint64_t bz)
{
    auto depth = GetDepth(key);

    auto child = ((bx & 1ull) << 2) | ((by & 1ull) << 1) | (bz & 1ull);
    auto shift = depth * 3ull;

    auto extended = key | (child << shift);
    return SetDepth(extended, depth + 1);
}

__host__ __device__ uint64_t Octree::ToKey(float3 position, float3 aabbMin, float3 aabbMax, uint64_t depth)
{
    uint64_t payload = 0ull;
    float3 bbMin = aabbMin;
    float3 bbMax = aabbMax;

    for (uint64_t i = 0; i < depth; ++i)
    {
        float3 center
        {
            0.5f * (bbMin.x + bbMax.x),
            0.5f * (bbMin.y + bbMax.y),
            0.5f * (bbMin.z + bbMax.z)
        };

        auto bx = (position.x >= center.x) ? 1ull : 0ull;
        auto by = (position.y >= center.y) ? 1ull : 0ull;
        auto bz = (position.z >= center.z) ? 1ull : 0ull;

        auto child = (bx << 2) | (by << 1) | bz;

        payload |= (child << (i * 3ull));

        bbMin.x = bx ? center.x : bbMin.x;
        bbMax.x = bx ? bbMax.x : center.x;

        bbMin.y = by ? center.y : bbMin.y;
        bbMax.y = by ? bbMax.y : center.y;

        bbMin.z = bz ? center.z : bbMin.z;
        bbMax.z = bz ? bbMax.z : center.z;
    }

    return SetDepth(payload, depth);
}

__host__ __device__ float3 Octree::ToPosition(uint64_t key, float3 aabbMin, float3 aabbMax)
{
    unsigned depth = GetDepth(key);

    float3 bbMin = aabbMin;
    float3 bbMax = aabbMax;

    for (unsigned i = 0; i < depth; ++i)
    {
        float3 center
        {
            0.5f * (bbMin.x + bbMax.x),
            0.5f * (bbMin.y + bbMax.y),
            0.5f * (bbMin.z + bbMax.z)
        };

        unsigned code = GetCode(key, i);
        unsigned bx = (code >> 2) & 1ull;
        unsigned by = (code >> 1) & 1ull;
        unsigned bz = (code >> 0) & 1ull;

        bbMin.x = bx ? center.x : bbMin.x;
        bbMax.x = bx ? bbMax.x : center.x;

        bbMin.y = by ? center.y : bbMin.y;
        bbMax.y = by ? bbMax.y : center.y;

        bbMin.z = bz ? center.z : bbMin.z;
        bbMax.z = bz ? bbMax.z : center.z;
    }

    return float3
    {
        0.5f * (bbMin.x + bbMax.x),
        0.5f * (bbMin.y + bbMax.y),
        0.5f * (bbMin.z + bbMax.z)
    };
}

__host__ __device__ float Octree::GetScale(OctreeKey key, uint64_t maxDepth, float unitLength)
{
    auto level = Octree::GetDepth(key);
    const unsigned k = (maxDepth >= level) ? (maxDepth - level) : 0u;

#if defined(__CUDA_ARCH__)
    // device path
    return ldexpf(unitLength, static_cast<int>(k));
#else
    // host path
    return std::ldexp(unitLength, static_cast<int>(k));
#endif
}

__host__ __device__ void Octree::SplitChildAABB(
    const float3& pmin,
    const float3& pmax,
    unsigned childCode3, // 0..7, (bx<<2)|(by<<1)|bz
    float3* cmin,
    float3* cmax)
{
    float3 center
    {
        0.5f * (pmin.x + pmax.x),
        0.5f * (pmin.y + pmax.y),
        0.5f * (pmin.z + pmax.z)
    };

    const unsigned bx = (childCode3 >> 2) & 1u;
    const unsigned by = (childCode3 >> 1) & 1u;
    const unsigned bz = (childCode3 >> 0) & 1u;

    cmin->x = bx ? center.x : pmin.x;  cmax->x = bx ? pmax.x : center.x;
    cmin->y = by ? center.y : pmin.y;  cmax->y = by ? pmax.y : center.y;
    cmin->z = bz ? center.z : pmin.z;  cmax->z = bz ? pmax.z : center.z;
}

__host__ __device__ float Octree::Dist2PointAABB(const float3& q, const float3& bmin, const float3& bmax)
{
    const float dx = fmaxf(fmaxf(bmin.x - q.x, 0.0f), q.x - bmax.x);
    const float dy = fmaxf(fmaxf(bmin.y - q.y, 0.0f), q.y - bmax.y);
    const float dz = fmaxf(fmaxf(bmin.z - q.z, 0.0f), q.z - bmax.z);
    return dx * dx + dy * dy + dz * dz;
}

__host__ __device__ float3 Octree::ClosestPointOnAABB(const float3& q, const float3& bmin, const float3& bmax)
{
    float3 c;
    c.x = fminf(fmaxf(q.x, bmin.x), bmax.x);
    c.y = fminf(fmaxf(q.y, bmin.y), bmax.y);
    c.z = fminf(fmaxf(q.z, bmin.z), bmax.z);
    return c;
}
#pragma endregion


void HostOctree::Initialize(
    float3* positions,
    unsigned int numberOfPositions,
    float3 aabbMin,
    float3 aabbMax,
    uint64_t leafDepth,
    float unitLength)
{
    this->leafDepth = leafDepth;

    auto dimension = aabbMax - aabbMin;
    auto maxLength = fmaxf(dimension.x, fmaxf(dimension.y, dimension.z));
    auto center = (aabbMin + aabbMax) * 0.5f;

    auto bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
    auto bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

    this->aabbMin = bbMin;
    this->aabbMax = bbMax;
    this->unitLength = unitLength;

    std::map<uint64_t, uint64_t> codes;

    for (size_t i = 0; i < numberOfPositions; ++i)
    {
        uint64_t leafKey = Octree::ToKey(positions[i], bbMin, bbMax, leafDepth);
        codes[leafKey] = leafDepth;

        uint64_t code = leafKey;
        if (leafDepth > 0)
        {
            for (int j = static_cast<int>(leafDepth) - 1; j >= 0; --j)
            {
                code = Octree::GetParentKey(code);
                auto [it, inserted] = codes.emplace(code, static_cast<uint64_t>(j));
                if (!inserted) break;
            }
        }
    }

    nodes.reserve(codes.size());
    for (auto& kvp : codes)
    {
        nodes.push_back({
            kvp.first,
            UINT32_MAX,
            { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
              UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX }
            });

        kvp.second = nodes.size() - 1;

        if (0 == Octree::GetDepth(kvp.first))
        {
            rootIndex = kvp.second;
        }
    }


    if (rootIndex != UINT32_MAX)
    {
        nodes[rootIndex].bmin = this->aabbMin;
        nodes[rootIndex].bmax = this->aabbMax;
    }

    for (unsigned int i = 0; i < nodes.size(); i++)
    {
        auto& node = nodes[i];

        if (Octree::GetDepth(node.key) == 0u)
        {
            node.parent = UINT32_MAX;
            continue;
        }

        auto parentKey = Octree::GetParentKey(node.key);
        auto it = codes.find(parentKey);
        if (it == codes.end()) continue;
        auto parentIndex = static_cast<unsigned>(it->second);

        node.parent = parentIndex;
        auto& parentNode = nodes[parentIndex];

        const unsigned childCode =
            static_cast<unsigned>(Octree::GetCode(node.key, Octree::GetDepth(node.key) - 1ull));
        parentNode.children[childCode] = i;

        float3 cmn, cmx;
        Octree::SplitChildAABB(parentNode.bmin, parentNode.bmax, childCode, &cmn, &cmx);
        node.bmin = cmn;
        node.bmax = cmx;
    }
}

void HostOctree::Terminate()
{
    nodes.clear();
    nodes.shrink_to_fit();
}

OctreeNode* HostOctree::NN(float3 query)
{
    if (rootIndex == UINT32_MAX || nodes.empty())
    {
        return nullptr;
    }

    struct Item
    {
        float d2;
        unsigned idx;
    };

    struct Cmp
    {
        bool operator()(const Item& a, const Item& b) const
        {
            return a.d2 > b.d2; // min-heap
        }
    };

    auto nodeDist2 = [&](unsigned idx) -> float
    {
        const auto& n = nodes[idx];
        return Octree::Dist2PointAABB(query, n.bmin, n.bmax);
    };

    std::priority_queue<Item, std::vector<Item>, Cmp> pq;
    pq.push({ nodeDist2(rootIndex), rootIndex });

    float bestD2 = FLT_MAX;
    OctreeNode* bestNode = nullptr;

    while (!pq.empty())
    {
        Item it = pq.top();
        pq.pop();

        // 하한이 현재 best 이상이면 이 서브트리 전체 prune
        if (it.d2 >= bestD2)
        {
            continue;
        }

        const OctreeNode& n = nodes[it.idx];

        bool isLeaf = true;
        for (int c = 0; c < 8; ++c)
        {
            unsigned ci = n.children[c];
            if (ci == UINT32_MAX) continue;
            isLeaf = false;

            float cd2 = nodeDist2(ci);
            if (cd2 < bestD2)
            {
                pq.push({ cd2, ci });
            }
        }

        if (isLeaf)
        {
            // leaf 자체의 AABB 하한이 it.d2 이므로, 여기서 best 갱신
            if (it.d2 < bestD2)
            {
                bestD2 = it.d2;
                bestNode = const_cast<OctreeNode*>(&nodes[it.idx]);
            }
        }
    }

    return bestNode;
}

__global__ void Kernel_DeviceOctree_BuildOctreeNodeKeys(
    const float3* __restrict__ positions,
    unsigned int numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    uint64_t depth,
    HashMap<uint64_t, unsigned int> octreeKeys)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    const float3 p = positions[tid];
    const uint64_t code = Octree::ToKey(p, aabbMin, aabbMax, depth);

    HashMap<uint64_t, unsigned int>::insert(octreeKeys.info, code, 1);
    
    if (depth == 0u) return;

    auto parentCode = Octree::GetParentKey(code);
    for (long i = (long)depth - 1; i >= 0; i--)
    {
        HashMap<uint64_t, unsigned int>::insert(octreeKeys.info, parentCode, 1);
        parentCode = Octree::GetParentKey(parentCode);
    }
}

__global__ void Kernel_DeviceOctree_BuildOctreeNodes(
    HashMap<uint64_t, unsigned int> octreeKeys,
    OctreeNode* nodes,
    unsigned int* numberOfNodes)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= octreeKeys.info.capacity) return;
    
    auto& entry = octreeKeys.info.entries[tid];
    if (UINT64_MAX == entry.key) return;
 
    auto index = atomicAdd(numberOfNodes, 1);
    auto& n = nodes[index];
    n.key = entry.key;
    n.parent = UINT32_MAX;
#pragma unroll
    for (int i = 0; i < 8; ++i) n.children[i] = UINT32_MAX;

    n.bmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    n.bmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    entry.value = index;
}

// 패스1: 링크(자식 AABB 계산 금지)
__global__ void Kernel_DeviceOctree_LinkOctreeNodes(
    HashMap<uint64_t, unsigned int> hm,
    OctreeNode* __restrict__ nodes,
    unsigned int numberOfNodes,
    unsigned int* __restrict__ rootIndex,
    float3 rootMin, float3 rootMax)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfNodes) return;
    OctreeNode& node = nodes[tid];
    unsigned depth = (unsigned)Octree::GetDepth(node.key);
    if (depth == 0u) { *rootIndex = tid; node.parent = UINT32_MAX; node.bmin = rootMin; node.bmax = rootMax; return; }
    uint64_t parentKey = Octree::GetParentKey(node.key);
    unsigned parentIdx = UINT32_MAX;
    hm.find(hm.info, parentKey, &parentIdx);
    if (parentIdx == UINT32_MAX) return;
    node.parent = parentIdx;
    OctreeNode& p = nodes[parentIdx];
    unsigned childCode = (unsigned)Octree::GetCode(node.key, depth - 1u);
    p.children[childCode] = tid;
}

__global__ void Kernel_DeviceOctree_PropagateAABB_Level(
    OctreeNode* __restrict__ nodes,
    unsigned numberOfNodes,
    unsigned level)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfNodes) return;
    OctreeNode& n = nodes[tid];
    unsigned depth = (unsigned)Octree::GetDepth(n.key);
    if (depth != level) return;
    unsigned pidx = n.parent;
    if (pidx == UINT32_MAX) return;
    const OctreeNode& p = nodes[pidx];
    unsigned code = (unsigned)Octree::GetCode(n.key, depth - 1u);
    float3 cmin, cmax;
    Octree::SplitChildAABB(p.bmin, p.bmax, code, &cmin, &cmax);
    n.bmin = cmin; n.bmax = cmax;
}

void DeviceOctree::Initialize(
    float3* positions,
    unsigned int numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    uint64_t leafDepth,
    float unitLength)
{
    this->leafDepth = leafDepth;

    const float3 dimension = aabbMax - aabbMin;
    const float  maxLength = fmaxf(dimension.x, fmaxf(dimension.y, dimension.z));
    const float3 center = (aabbMin + aabbMax) * 0.5f;

    const float3 bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
    const float3 bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

    this->aabbMin = bbMin;
    this->aabbMax = bbMax;
    this->unitLength = unitLength;

    // Build key set first (unique keys go into the hashmap)
    octreeKeys.Initialize(static_cast<size_t>(numberOfPoints) * 64u, 64u);

    LaunchKernel(Kernel_DeviceOctree_BuildOctreeNodeKeys, numberOfPoints,
        positions,
        numberOfPoints,
        bbMin,
        bbMax,
        leafDepth,
        octreeKeys);
    CUDA_SYNC();

    // Determine number of unique entries and size node buffer accordingly
    unsigned int numberOfEntries = 0;
    CUDA_COPY_D2H(&numberOfEntries, octreeKeys.info.numberOfEntries, sizeof(unsigned int));
    CUDA_SYNC();

    const size_t capacity = static_cast<size_t>(numberOfEntries) + 64; // small slack
    CUDA_MALLOC(&nodes, sizeof(OctreeNode) * capacity);
    allocatedNodes = static_cast<unsigned int>(capacity);

    CUDA_MALLOC(&d_numberOfNodes, sizeof(unsigned int));
    CUDA_MEMSET(d_numberOfNodes, 0, sizeof(unsigned int));
    CUDA_SYNC();

    // Build nodes (scan hashmap slots)
    LaunchKernel(Kernel_DeviceOctree_BuildOctreeNodes, octreeKeys.info.capacity,
        octreeKeys,
        nodes,
        d_numberOfNodes);
    CUDA_SYNC();

    CUDA_COPY_D2H(&numberOfNodes, d_numberOfNodes, sizeof(unsigned int));
    CUDA_SYNC();

    CUDA_MALLOC(&rootIndex, sizeof(unsigned int));
    CUDA_MEMSET(rootIndex, 0xFF, sizeof(unsigned int));

    // Link parents/children (no child AABB)
    LaunchKernel(Kernel_DeviceOctree_LinkOctreeNodes, numberOfNodes,
        octreeKeys,
        nodes,
        numberOfNodes,
        rootIndex,
        bbMin,
        bbMax);
    CUDA_SYNC();

    // AABB propagate by depth (parents are ready from previous level)
    for (unsigned int lvl = 1; lvl <= leafDepth; ++lvl)
    {
        LaunchKernel(Kernel_DeviceOctree_PropagateAABB_Level, numberOfNodes,
            nodes,
            numberOfNodes,
            lvl);
        CUDA_SYNC();
    }

    printf("numberOfPoints : %u, numberOfEntries : %u, numberOfNodes : %u\n",
        numberOfPoints, numberOfEntries, numberOfNodes);
}

void DeviceOctree::Terminate()
{
    CUDA_FREE(nodes);
    nodes = nullptr;

    CUDA_FREE(d_numberOfNodes);
    d_numberOfNodes = nullptr;

    allocatedNodes = 0;
    numberOfNodes = 0;

    octreeKeys.Terminate();
}

void DeviceOctree::NN_H(
    float3* queries,
    unsigned int numberOfQueries,
    unsigned int* outNearestIndex,
    float* outNearestD2)
{
    if (nodes == nullptr || rootIndex == nullptr || numberOfNodes == 0 || numberOfQueries == 0)
    {
        return;
    }

    float3* d_queries = nullptr;
    CUDA_MALLOC(&d_queries, sizeof(float3) * numberOfQueries);
    CUDA_COPY_H2D(d_queries, queries, sizeof(float3) * numberOfQueries);

    unsigned int* d_outNearestIndex = nullptr;
    CUDA_MALLOC(&d_outNearestIndex, sizeof(unsigned int) * numberOfQueries);
    CUDA_MEMSET(d_outNearestIndex, 0, sizeof(unsigned int) * numberOfQueries);

    float* d_outNearestD2 = nullptr;
    CUDA_MALLOC(&d_outNearestD2, sizeof(float) * numberOfQueries);
    CUDA_MEMSET(d_outNearestD2, 0, sizeof(float) * numberOfQueries);
    CUDA_SYNC();

    //unsigned int h_root = UINT32_MAX;
    //CUDA_COPY_D2H(&h_root, rootIndex, sizeof(unsigned int));
    //CUDA_SYNC();

    CUDA_TS(NN_H);

    LaunchKernel(Kernel_DeviceOctree_NN_Batch, numberOfQueries,
        nodes,
        rootIndex,
        d_queries,
        numberOfQueries,
        d_outNearestIndex,
        d_outNearestD2);
    CUDA_SYNC();

    CUDA_TE(NN_H);

    CUDA_COPY_D2H(outNearestIndex, d_outNearestIndex, sizeof(unsigned int) * numberOfQueries);
    CUDA_COPY_D2H(outNearestD2, d_outNearestD2, sizeof(float) * numberOfQueries);

    CUDA_SYNC();

    CUDA_FREE(d_queries);
    CUDA_FREE(d_outNearestIndex);
    CUDA_FREE(d_outNearestD2);
}

void DeviceOctree::NN_D(
    float3* d_queries,
    unsigned int numberOfQueries,
    unsigned int* d_outNearestIndex,
    float* d_outNearestD2)
{
    if (nodes == nullptr || rootIndex == nullptr || nullptr == d_outNearestIndex || numberOfNodes == 0 || numberOfQueries == 0)
    {
        return;
    }

    CUDA_TS(NN_D);

    LaunchKernel(Kernel_DeviceOctree_NN_Batch, numberOfQueries,
        nodes,
        rootIndex,
        d_queries,
        numberOfQueries,
        d_outNearestIndex,
        d_outNearestD2);
    CUDA_SYNC();

    CUDA_TE(NN_D);
}

bool DeviceOctree::NN_Single(
    const float3& query,
    unsigned int* h_index,
    float* h_d2)
{
    if (nodes == nullptr || rootIndex == nullptr || numberOfNodes == 0)
    {
        if (h_index) *h_index = UINT32_MAX;
        if (h_d2)    *h_d2 = FLT_MAX;
        return false;
    }

    // 디바이스 버퍼 임시 할당
    float3* d_q = nullptr;
    unsigned int* d_idx = nullptr;
    float* d_d2 = nullptr;
    CUDA_MALLOC(&d_q, sizeof(float3));
    CUDA_MALLOC(&d_idx, sizeof(unsigned int));
    CUDA_MALLOC(&d_d2, sizeof(float));
    CUDA_COPY_H2D(d_q, &query, sizeof(float3));

    // 호출
    NN_D(d_q, 1u, d_idx, d_d2);

    // 결과 수거
    unsigned int idx;
    float d2;
    CUDA_COPY_D2H(&idx, d_idx, sizeof(unsigned int));
    CUDA_COPY_D2H(&d2, d_d2, sizeof(float));
    CUDA_SYNC();

    if (h_index) *h_index = idx;
    if (h_d2)    *h_d2 = d2;

    CUDA_FREE(d_q);
    CUDA_FREE(d_idx);
    CUDA_FREE(d_d2);

    return idx != UINT32_MAX;
}

__device__ float DeviceOctree::nodeDist2_direct(
    const OctreeNode* nodes,
    unsigned int idx,
    const float3& q)
{
    const OctreeNode& n = nodes[idx];
    return Octree::Dist2PointAABB(q, n.bmin, n.bmax);
}

__device__ bool DeviceOctree::isLeaf(const OctreeNode& n)
{
    return (n.children[0] == UINT32_MAX) &&
        (n.children[1] == UINT32_MAX) &&
        (n.children[2] == UINT32_MAX) &&
        (n.children[3] == UINT32_MAX) &&
        (n.children[4] == UINT32_MAX) &&
        (n.children[5] == UINT32_MAX) &&
        (n.children[6] == UINT32_MAX) &&
        (n.children[7] == UINT32_MAX);
}

// DeviceOctree 내부 (isLeaf 아래에 추가)
__device__ void DeviceOctree::preferred_child_order(
    const OctreeNode& n, const float3& q, int order[8])
{
    // 노드 박스 중심
    float3 c{
        0.5f * (n.bmin.x + n.bmax.x),
        0.5f * (n.bmin.y + n.bmax.y),
        0.5f * (n.bmin.z + n.bmax.z)
    };

    // 쿼리가 속한 옥탄을 우선 방문
    const int prefer =
        ((q.x >= c.x) ? 4 : 0) |
        ((q.y >= c.y) ? 2 : 0) |
        ((q.z >= c.z) ? 1 : 0);

    // 인접 옥탄들까지 XOR 패턴으로 배치 (정렬 오버헤드 없음)
    order[0] = prefer;
    order[1] = prefer ^ 1;
    order[2] = prefer ^ 2;
    order[3] = prefer ^ 4;
    order[4] = prefer ^ 3;
    order[5] = prefer ^ 5;
    order[6] = prefer ^ 6;
    order[7] = prefer ^ 7;
}

__device__ void DeviceOctree::push_children_sorted(
    const OctreeNode& n,
    const float3& q,
    const OctreeNode* __restrict__ nodes,
    float bestD2,
    unsigned int* __restrict__ stackIdx,
    float* __restrict__ stackD2,
    int* __restrict__ top,
    const int STACK_CAP)
{
    // 수집
    unsigned int cidx[8];
    float        cd2[8];
    int cnt = 0;

#pragma unroll
    for (int c = 0; c < 8; ++c)
    {
        unsigned int ci = n.children[c];
        if (ci == UINT32_MAX) continue;

        const OctreeNode& cn = nodes[ci];
        float d2 = Octree::Dist2PointAABB(q, cn.bmin, cn.bmax);
        if (d2 >= bestD2) continue; // 프루닝된 건 skip

        cidx[cnt] = ci;
        cd2[cnt] = d2;
        ++cnt;
    }

    // 삽입정렬 (오름차순)
#pragma unroll
    for (int i = 1; i < cnt; ++i)
    {
        float        keyD = cd2[i];
        unsigned int keyI = cidx[i];
        int j = i - 1;
        while (j >= 0 && cd2[j] > keyD)
        {
            cd2[j + 1] = cd2[j];
            cidx[j + 1] = cidx[j];
            --j;
        }
        cd2[j + 1] = keyD;
        cidx[j + 1] = keyI;
    }

    // 가까운 것부터 push
#pragma unroll
    for (int i = 0; i < cnt; ++i)
    {
        if (*top >= STACK_CAP) break;
        stackIdx[*top] = cidx[i];
        stackD2[*top] = cd2[i];
        ++(*top);
    }
}

__device__ unsigned int DeviceOctree::DeviceOctree_NearestLeaf(
    const float3& query,
    const OctreeNode* __restrict__ nodes,
    const unsigned int* __restrict__ rootIndex,
    float* outBestD2)
{
    if (nodes == nullptr || *rootIndex == UINT32_MAX) {
        if (outBestD2) *outBestD2 = FLT_MAX;
        return UINT32_MAX;
    }

    // 스택: 인덱스만 유지 (거리 배열 제거)
    const int STACK_CAP = 64;     // 필요하면 128로
    unsigned int stackIdx[STACK_CAP];
    int top = 0;

    const unsigned int root = *rootIndex;
    stackIdx[top++] = root;

    float bestD2 = FLT_MAX;
    unsigned int bestIdx = UINT32_MAX;

    // 1) Greedy 하강: 가장 유력한 자식 하나로 계속 내려가며 bestD2 초기로 확 낮춤
    {
        unsigned int cur = root;
        for (;;)
        {
            const OctreeNode& n = nodes[cur];
            const float nd2 = Octree::Dist2PointAABB(query, n.bmin, n.bmax);
            if (nd2 >= bestD2) break;

            if (isLeaf(n)) {
                bestD2 = nd2;
                bestIdx = cur;
                break;
            }

            int ord[8];
            preferred_child_order(n, query, ord);

            bool descended = false;
            // 첫 번째(가장 유력) 자식으로 즉시 하강, 나머지는 스택에 후보로 적재
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const unsigned int ci = n.children[ord[k]];
                if (ci == UINT32_MAX) continue;

                const OctreeNode& cn = nodes[ci];
                const float cd2 = Octree::Dist2PointAABB(query, cn.bmin, cn.bmax);
                if (cd2 >= bestD2) continue;

                if (!descended) { cur = ci; descended = true; }
                else if (top < STACK_CAP) { stackIdx[top++] = ci; }
            }
            if (!descended) break;
        }
    }

    // 2) 남은 후보들에 대해 브랜치-앤-바운드
    while (top > 0)
    {
        const unsigned int idx = stackIdx[--top];

        const OctreeNode& n = nodes[idx];
        const float nd2 = Octree::Dist2PointAABB(query, n.bmin, n.bmax);
        if (nd2 >= bestD2) continue;

        if (isLeaf(n)) {
            if (nd2 < bestD2) { bestD2 = nd2; bestIdx = idx; }
            if (bestD2 == 0.0f) break;
            continue;
        }

        int ord[8];
        preferred_child_order(n, query, ord);

#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const unsigned int ci = n.children[ord[k]];
            if (ci == UINT32_MAX) continue;

            const OctreeNode& cn = nodes[ci];
            const float cd2 = Octree::Dist2PointAABB(query, cn.bmin, cn.bmax);
            if (cd2 >= bestD2) continue;

            if (top < STACK_CAP) stackIdx[top++] = ci;
        }
    }

    if (outBestD2) *outBestD2 = bestD2;
    return bestIdx;
}

__global__ void Kernel_DeviceOctree_NN_Batch(
    const OctreeNode* __restrict__ nodes,
    const unsigned int* __restrict__ rootIndex,
    const float3* __restrict__ queries,
    unsigned int numberOfQueries,
    unsigned int* __restrict__ outIndices,
    float* __restrict__ outD2)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfQueries) return;

    float bestD2;
    unsigned int bestIdx =
        DeviceOctree::DeviceOctree_NearestLeaf(queries[tid], nodes, rootIndex, &bestD2);

    outIndices[tid] = bestIdx;
    if(outD2) outD2[tid] = bestD2;
}
