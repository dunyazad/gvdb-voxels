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
            root = &nodes.back();
        }
    }

    for (unsigned int i = 0; i < nodes.size(); i++)
    {
        auto& node = nodes[i];

        auto parentKey = Octree::GetParentKey(node.key);
        auto it = codes.find(parentKey);
        if (it == codes.end()) continue;
        auto parentIndex = (*it).second;

        node.parent = parentIndex;
        auto& parentNode = nodes[parentIndex];
        auto childIndex = Octree::GetCode(node.key, Octree::GetDepth(node.key) - 1);
        parentNode.children[childIndex] = i;
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
    nodes[index].key = entry.key;
    nodes[index].parent = UINT32_MAX;
    nodes[index].children[0] = UINT32_MAX;
    nodes[index].children[1] = UINT32_MAX;
    nodes[index].children[2] = UINT32_MAX;
    nodes[index].children[3] = UINT32_MAX;
    nodes[index].children[4] = UINT32_MAX;
    nodes[index].children[5] = UINT32_MAX;
    nodes[index].children[6] = UINT32_MAX;
    nodes[index].children[7] = UINT32_MAX;

    entry.value = index;
}

__global__ void Kernel_DeviceOctree_LinkOctreeNodes(
    HashMap<uint64_t, unsigned int> octreeKeys,
    OctreeNode* nodes,
    unsigned int numberOfNodes,
    unsigned int* rootIndex)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfNodes) return;

    auto& node = nodes[tid];
    if (0 == Octree::GetDepth(node.key))
    {
        *rootIndex = tid;
        return;
    }

    auto parentKey = Octree::GetParentKey(node.key);
    unsigned int parentIndex = UINT32_MAX;
    octreeKeys.find(octreeKeys.info, parentKey, &parentIndex);
    if (UINT32_MAX == parentIndex) return;

    node.parent = parentIndex;
    auto& parentNode = nodes[parentIndex];
    auto childIndex = Octree::GetCode(node.key, Octree::GetDepth(node.key) - 1);
    //printf("childIndex = %d\n", childIndex);
    parentNode.children[childIndex] = tid;
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

    // Link parents/children
    LaunchKernel(Kernel_DeviceOctree_LinkOctreeNodes, numberOfNodes,
        octreeKeys,
        nodes,
        numberOfNodes,
        rootIndex);
    CUDA_SYNC();

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
