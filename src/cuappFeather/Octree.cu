#include <Octree.cuh>

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

__global__ void Kernel_DeviceOctree_LinkOctreeNodes(
    HashMap<uint64_t, unsigned int> octreeKeys,
    OctreeNode* __restrict__ nodes,
    unsigned int numberOfNodes,
    unsigned int* __restrict__ rootIndex,
    float3 rootMin,
    float3 rootMax)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfNodes) return;

    OctreeNode& node = nodes[tid];
    const unsigned depth = static_cast<unsigned>(Octree::GetDepth(node.key));

    if (depth == 0u)
    {
        // 루트만 AABB 세팅
        *rootIndex = tid;
        node.parent = UINT32_MAX;
        node.bmin = rootMin;
        node.bmax = rootMax;
        return;
    }

    // 부모 찾고 링크만 설정 (AABB 계산 금지!)
    const uint64_t parentKey = Octree::GetParentKey(node.key);
    unsigned int parentIndex = UINT32_MAX;
    octreeKeys.find(octreeKeys.info, parentKey, &parentIndex);
    if (parentIndex == UINT32_MAX) return;

    node.parent = parentIndex;
    OctreeNode& parentNode = nodes[parentIndex];

    const unsigned childCode =
        static_cast<unsigned>(Octree::GetCode(node.key, depth - 1u));
    parentNode.children[childCode] = tid;
}

__global__ void Kernel_DeviceOctree_PropagateAABB_Level(
    OctreeNode* __restrict__ nodes,
    unsigned int numberOfNodes,
    unsigned int level)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfNodes) return;

    OctreeNode& node = nodes[tid];
    const unsigned depth = static_cast<unsigned>(Octree::GetDepth(node.key));
    if (depth != level) return;

    const unsigned int parentIdx = node.parent;
    if (parentIdx == UINT32_MAX) return; // safety

    const OctreeNode& parent = nodes[parentIdx];
    const unsigned childCode =
        static_cast<unsigned>(Octree::GetCode(node.key, depth - 1u));

    float3 cmn, cmx;
    Octree::SplitChildAABB(parent.bmin, parent.bmax, childCode, &cmn, &cmx);
    node.bmin = cmn;
    node.bmax = cmx;
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

void DeviceOctree::NN(
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

    //unsigned int h_root = UINT32_MAX;
    //CUDA_COPY_D2H(&h_root, rootIndex, sizeof(unsigned int));
    //CUDA_SYNC();

    CUDA_TS(NN);

    LaunchKernel(Kernel_DeviceOctree_NN_Batch, numberOfQueries,
        nodes,
        rootIndex,
        d_queries,
        numberOfQueries,
        d_outNearestIndex,
        d_outNearestD2);
    CUDA_SYNC();

    CUDA_TE(NN);

    CUDA_COPY_D2H(outNearestIndex, d_outNearestIndex, sizeof(unsigned int) * numberOfQueries);
    CUDA_COPY_D2H(outNearestD2, d_outNearestD2, sizeof(float) * numberOfQueries);

    CUDA_SYNC();

    CUDA_FREE(d_queries);
    CUDA_FREE(d_outNearestIndex);
    CUDA_FREE(d_outNearestD2);
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
    NN(d_q, 1u, d_idx, d_d2);

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

__global__ void Kernel_DeviceOctree_NN_Batch(
    OctreeNode* __restrict__ nodes,
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
    outD2[tid] = bestD2;
}
