#include <LBVH.cuh>

void DeviceLBVH::Initialize(
    float3* positions,
    uint3* faces,
    float3 aabbMin,
    float3 aabbMax,
    unsigned int numberOfFaces)
{
    numberOfMortonKeys = numberOfFaces;

    CUDA_MALLOC(&mortonKeys, sizeof(MortonKey) * numberOfMortonKeys);

    LaunchKernel(Kernel_DeviceLBVH_BuildMortonKeys, numberOfMortonKeys,
        positions, faces, aabbMin, aabbMax, numberOfMortonKeys, mortonKeys);

    allocatedNodes = 2 * numberOfMortonKeys - 1;

    CUDA_MALLOC(&nodes, sizeof(LBVHNode) * allocatedNodes);

    LaunchKernel(Kernel_DeviceLBVH_InitializeLeafNodes, numberOfMortonKeys,
        mortonKeys, numberOfMortonKeys, nodes, aabbMin, aabbMax);
}

__global__ void Kernel_DeviceLBVH_BuildMortonKeys(
    float3* positions,
    uint3* faces,
    float3 aabbMin,
    float3 aabbMax,
    unsigned int numberOfMortonKeys,
    MortonKey* mortonKeys)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numberOfMortonKeys) return;

    auto& face = faces[tid];
    auto& p0 = positions[face.x];
    auto& p1 = positions[face.y];
    auto& p2 = positions[face.z];

    auto center = (p0 + p1 + p2) / 3.0f;
    auto mortonCode = MortonCode::FromPosition(center, aabbMin, aabbMax);

    mortonKeys[tid].code = mortonCode;
    mortonKeys[tid].index = tid;
}

__global__ void Kernel_DeviceLBVH_InitializeLeafNodes(
    MortonKey* mortonKeys,
    unsigned int numberOfMortonKeys,
    LBVHNode* nodes,
    const float3& aabbMin,
    const float3& aabbMax)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numberOfMortonKeys) return;

    auto& node = nodes[tid];
    auto& mortonKey = mortonKeys[tid];

    node.parentNodeIndex = UINT32_MAX;
    node.leftNodeIndex = UINT32_MAX;
    node.rightNodeIndex = UINT32_MAX;

    node.aabb = MortonCode::CodeToAABB(
        mortonKey.code,
        aabbMin,
        aabbMax,
        MortonCode::GetMaxDepth()
    );
}
