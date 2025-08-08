#include <Octree.cuh>

__global__ void Kernel_Octree_Initialize(
    const float3* __restrict__ positions,
    unsigned int numberOfPoints,
    float3 center,
    float voxelSize,
    uint8_t depth,
    int3 gridOffset,
    OctreeNode* __restrict__ nodes,
    unsigned int numberOfAllocatedNodes,
    unsigned int* __restrict__ numberOfNodes)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfPoints) return;

    const float3 p = positions[tid];
    const uint64_t code = Octree::KeyFromPoint_Voxel(p, center, voxelSize, depth, gridOffset);

    OctreeNode n;
    n.mortonCode = code;
    n.level = Octree::UnpackDepth(code);
    n.parent = UINT32_MAX;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        n.children[i] = UINT32_MAX;
    }

    nodes[tid] = n;
    if (tid == 0) atomicMax(numberOfNodes, numberOfPoints);
}

void Octree::Initialize(float3* positions, unsigned numberOfPoints, float3 center, float voxelSize)
{
    const size_t cap = size_t(numberOfPoints) * 2u;
    CUDA_MALLOC(&nodes, sizeof(OctreeNode) * cap);
    CUDA_MEMSET(nodes, 0xFF, sizeof(OctreeNode) * cap);
    allocatedNodes = static_cast<unsigned int>(cap);

    CUDA_MALLOC(&d_numberOfNodes, sizeof(unsigned int));
    CUDA_MEMSET(d_numberOfNodes, 0, sizeof(unsigned int));

    mortonCodeOctreeNodeMapping.Initialize(size_t(numberOfPoints) * 64u, 64u);

    // launch
    const int block = 256;
    const int grid = int((numberOfPoints + block - 1) / block);

    Kernel_Octree_Initialize << <grid, block >> > (
        positions,
        numberOfPoints,
        center,
        voxelSize,
        Octree::kMaxDepth,
        GridOffset(),
        nodes,
        allocatedNodes,
        d_numberOfNodes);

    CUDA_SYNC();

    CUDA_COPY_D2H(&numberOfNodes, d_numberOfNodes, sizeof(unsigned int));

    CUDA_SYNC();
}

void Octree::Terminate()
{
    CUDA_FREE(nodes);
    CUDA_FREE(d_numberOfNodes);
    mortonCodeOctreeNodeMapping.Terminate();
}
