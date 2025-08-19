#include <Octree.cuh>

__global__ void Kernel_DeviceOctree_BuildOctreeNodeKeys(
    const float3* __restrict__ positions,
    unsigned int numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    uint8_t depth,
    HashMap<uint64_t, unsigned int> mortonCodes)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    const float3 p = positions[tid];
    const uint64_t code = DeviceOctree::ToKey(p, aabbMin, aabbMax, depth);

    HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, code, 1);
    
    //auto depth = DeviceOctree::GetDepth(code);
    auto parentCode = DeviceOctree::GetParentKey(code);
    for (int i = depth - 1; i >= 0; i--)
    {
        HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, parentCode, 1);
        parentCode = DeviceOctree::GetParentKey(parentCode);
    }
}

__global__ void Kernel_DeviceOctree_BuildOctreeNodes(
    HashMap<uint64_t, unsigned int> mortonCodes,
    DeviceOctreeNode* nodes,
    unsigned int* numberOfNodes)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= mortonCodes.info.capacity) return;
    
    auto& entry = mortonCodes.info.entries[tid];
    if (UINT64_MAX == entry.key) return;
 
    auto index = atomicAdd(numberOfNodes, 1);
    nodes[index].mortonCode = entry.key;
    nodes[index].level = (uint32_t)DeviceOctree::GetDepth(entry.key);
    nodes[index].parent = UINT32_MAX;
    nodes[index].children[0] = UINT32_MAX;
    nodes[index].children[1] = UINT32_MAX;
    nodes[index].children[2] = UINT32_MAX;
    nodes[index].children[3] = UINT32_MAX;
    nodes[index].children[4] = UINT32_MAX;
    nodes[index].children[5] = UINT32_MAX;
    nodes[index].children[6] = UINT32_MAX;
    nodes[index].children[7] = UINT32_MAX;
}

void DeviceOctree::Initialize(
    float3* positions,
    unsigned int numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    uint64_t leafDepth)
{
    auto dimension = aabbMax - aabbMin;
    auto maxLength = fmaxf(dimension.x, fmaxf(dimension.y, dimension.z));
    auto center = (aabbMin + aabbMax) * 0.5f;

    auto bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
    auto bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

    const size_t capacity = size_t(numberOfPoints) * 64;
    CUDA_MALLOC(&nodes, sizeof(DeviceOctreeNode) * capacity);
    CUDA_MEMSET(nodes, 0xFF, sizeof(DeviceOctreeNode) * capacity);
    allocatedNodes = static_cast<unsigned int>(capacity);

    CUDA_MALLOC(&d_numberOfNodes, sizeof(unsigned int));
    CUDA_MEMSET(d_numberOfNodes, 0, sizeof(unsigned int));
    CUDA_SYNC();

    mortonCodes.Initialize(size_t(numberOfPoints) * 64u, 64u);

    LaunchKernel(Kernel_DeviceOctree_BuildOctreeNodeKeys, numberOfPoints,
        positions,
        numberOfPoints,
        aabbMin,
        aabbMax,
        leafDepth,
        mortonCodes);

    unsigned int numberOfEntries = 0;
    CUDA_COPY_D2H(&numberOfEntries, mortonCodes.info.numberOfEntries, sizeof(unsigned int));
    CUDA_SYNC();

    LaunchKernel(Kernel_DeviceOctree_BuildOctreeNodes, mortonCodes.info.capacity,
        mortonCodes,
        nodes,
        d_numberOfNodes);
    CUDA_SYNC();

    CUDA_COPY_D2H(&numberOfNodes, d_numberOfNodes, sizeof(unsigned int));

    CUDA_SYNC();

    printf("numberOfPoints : %d, numberOfEntries : %d, numberOfNodes : %d\n", numberOfPoints, numberOfEntries, numberOfNodes);
}

void DeviceOctree::Terminate()
{
    CUDA_FREE(nodes);
    CUDA_FREE(d_numberOfNodes);
    mortonCodes.Terminate();
}
