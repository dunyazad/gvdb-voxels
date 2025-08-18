#include <Octree.cuh>

__global__ void Kernel_DeviceOctree_Initialize(
    const float3* __restrict__ positions,
    unsigned int numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    float voxelSize,
    uint8_t depth,
    DeviceOctreeNode* __restrict__ nodes,
    unsigned int numberOfAllocatedNodes,
    unsigned int* __restrict__ numberOfNodes,
    HashMap<uint64_t, unsigned int> mortonCodes)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    const float3 p = positions[tid];
    const uint64_t code = DeviceOctree::ToKey(p, aabbMin, aabbMax, depth);

    //printf("%.4f, %.4f, %.4f - %.4f, %.4f, %.4f, %d, %llu\n", XYZ(aabbMin), XYZ(aabbMax), depth, code);

    DeviceOctreeNode n;
    n.mortonCode = code;
    n.level = DeviceOctree::GetDepth(code);
    n.parent = DeviceOctree::GetParentKey(code);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        n.children[i] = UINT32_MAX;
    }

    nodes[tid] = n;
    if (tid == 0) atomicMax(numberOfNodes, numberOfPoints);

    HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, code, 1);
    
	auto parentCode = DeviceOctree::GetParentKey(code);
    while (0 != parentCode)
    {
        HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, parentCode, 1);
		parentCode = DeviceOctree::GetParentKey(parentCode);
    }
}

void DeviceOctree::Initialize(
    float3* positions,
    unsigned numberOfPoints,
    float3 aabbMin,
    float3 aabbMax,
    uint64_t leafDepth)
{
    auto dimension = aabbMax - aabbMin;
    auto maxLength = fmaxf(dimension.x, fmaxf(dimension.y, dimension.z));
    auto center = (aabbMin + aabbMax) * 0.5f;

    auto bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
    auto bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

    const size_t cap = size_t(numberOfPoints) * 2u;
    CUDA_MALLOC(&nodes, sizeof(DeviceOctreeNode) * cap);
    CUDA_MEMSET(nodes, 0xFF, sizeof(DeviceOctreeNode) * cap);
    allocatedNodes = static_cast<unsigned int>(cap);

    CUDA_MALLOC(&d_numberOfNodes, sizeof(unsigned int));
    CUDA_MEMSET(d_numberOfNodes, 0, sizeof(unsigned int));
    CUDA_SYNC();

    mortonCodeOctreeNodeMapping.Initialize(size_t(numberOfPoints) * 64u, 64u);
    mortonCodes.Initialize(size_t(numberOfPoints) * 64u, 64u);

    LaunchKernel(Kernel_DeviceOctree_Initialize, numberOfPoints,
        positions,
        numberOfPoints,
        aabbMin,
        aabbMax,
        voxelSize,
        leafDepth,
        nodes,
        allocatedNodes,
        d_numberOfNodes,
        mortonCodes);

	unsigned int numberOfEntries = 0;
	CUDA_COPY_D2H(&numberOfEntries, mortonCodes.info.numberOfEntries, sizeof(unsigned int));
    CUDA_SYNC();

    CUDA_COPY_D2H(&numberOfNodes, d_numberOfNodes, sizeof(unsigned int));

    CUDA_SYNC();

    printf("numberOfPoints : %d, numberOfEntries : %d\n", numberOfPoints, numberOfEntries);
}

void DeviceOctree::Terminate()
{
    CUDA_FREE(nodes);
    CUDA_FREE(d_numberOfNodes);
    mortonCodeOctreeNodeMapping.Terminate();
    mortonCodes.Terminate();
}
