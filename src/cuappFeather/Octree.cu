#include <Octree.cuh>
//
//__global__ void Kernel_Octree_Initialize(
//    const float3* __restrict__ positions,
//    unsigned int numberOfPoints,
//    float3 aabbMin,
//    float3 aabbMax,
//    float voxelSize,
//    uint8_t depth,
//    int3 gridOffset,
//    OctreeNode* __restrict__ nodes,
//    unsigned int numberOfAllocatedNodes,
//    unsigned int* __restrict__ numberOfNodes,
//    HashMap<uint64_t, unsigned int> mortonCodes)
//{
//    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid >= numberOfPoints) return;
//
//    const float3 p = positions[tid];
//    const uint64_t code = Octree::KeyFromPoint_Voxel(p, (aabbMin + aabbMax) * 0.5f, voxelSize, depth, gridOffset);
//
//    OctreeNode n;
//    n.mortonCode = code;
//    n.level = Octree::UnpackDepth(code);
//    n.parent = UINT32_MAX;
//#pragma unroll
//    for (int i = 0; i < 8; ++i)
//    {
//        n.children[i] = UINT32_MAX;
//    }
//
//    nodes[tid] = n;
//    if (tid == 0) atomicMax(numberOfNodes, numberOfPoints);
//
//    HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, code, 1);
//    
//	auto parentCode = Octree::ParentCode(code);
//    while (0 != parentCode)
//    {
//        HashMap<uint64_t, unsigned int>::insert(mortonCodes.info, parentCode, 1);
//		parentCode = Octree::ParentCode(parentCode);
//    }
//}
//
//void Octree::Initialize(
//    float3* positions,
//    unsigned numberOfPoints,
//    float3 aabbMin,
//    float3 aabbMax,
//    float voxelSize)
//{
//	this->numberOfPoints = numberOfPoints;
//	this->aabbMin = aabbMin;
//    this->aabbMax = aabbMax;
//	this->voxelSize = voxelSize;
//
//	auto center = (aabbMin + aabbMax) * 0.5f;
//	auto dimensions = aabbMax - aabbMin;
//	auto maxDimension = fmaxf(dimensions.x, fmaxf(dimensions.y, dimensions.z));
//
//	float unitSize = maxDimension;
//	unsigned int depth = 0;
//    while (unitSize > voxelSize && depth < Octree::kMaxDepth)
//    {
//        unitSize *= 0.5f;
//        ++depth;
//	}
//
//	printf("unitSize : %f, voxelSize : %f, depth : %d\n", unitSize, voxelSize, depth);
//
//    const size_t cap = size_t(numberOfPoints) * 2u;
//    CUDA_MALLOC(&nodes, sizeof(OctreeNode) * cap);
//    CUDA_MEMSET(nodes, 0xFF, sizeof(OctreeNode) * cap);
//    allocatedNodes = static_cast<unsigned int>(cap);
//
//    CUDA_MALLOC(&d_numberOfNodes, sizeof(unsigned int));
//    CUDA_MEMSET(d_numberOfNodes, 0, sizeof(unsigned int));
//    CUDA_SYNC();
//
//    mortonCodeOctreeNodeMapping.Initialize(size_t(numberOfPoints) * 64u, 64u);
//    mortonCodes.Initialize(size_t(numberOfPoints) * 64u, 64u);
//
//    LaunchKernel(Kernel_Octree_Initialize, numberOfPoints,
//        positions,
//        numberOfPoints,
//        aabbMin,
//        aabbMax,
//        voxelSize,
//        //depth, //Octree::kMaxDepth,
//        Octree::kMaxDepth,
//        GridOffset(),
//        nodes,
//        allocatedNodes,
//        d_numberOfNodes,
//        mortonCodes);
//
//	unsigned int numberOfEntries = 0;
//	CUDA_COPY_D2H(&numberOfEntries, mortonCodes.info.numberOfEntries, sizeof(unsigned int));
//    CUDA_SYNC();
//
//    CUDA_COPY_D2H(&numberOfNodes, d_numberOfNodes, sizeof(unsigned int));
//
//    CUDA_SYNC();
//
//    printf("numberOfPoints : %d, numberOfEntries : %d\n", numberOfPoints, numberOfEntries);
//}
//
//void Octree::Terminate()
//{
//    CUDA_FREE(nodes);
//    CUDA_FREE(d_numberOfNodes);
//    mortonCodeOctreeNodeMapping.Terminate();
//    mortonCodes.Terminate();
//}
