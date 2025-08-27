#include <LBVH.cuh>

struct MortonKeyCompare {
    __host__ __device__
        bool operator()(const MortonKey& a, const MortonKey& b) const {
        if (a.code < b.code) return true;
        if (a.code > b.code) return false;
        return a.index < b.index;
    }
};

void DeviceLBVH::Initialize(
    float3* positions,
    uint3* faces,
    const float3& aabbMin,
    const float3& aabbMax,
    unsigned int numberOfFaces)
{
    numberOfMortonKeys = numberOfFaces;

    CUDA_MALLOC(&mortonKeys, sizeof(MortonKey) * numberOfMortonKeys);

    LaunchKernel(Kernel_DeviceLBVH_BuildMortonKeys, numberOfMortonKeys,
        positions, faces, aabbMin, aabbMax, numberOfMortonKeys, mortonKeys);

    allocatedNodes = 2 * numberOfMortonKeys - 1;

    int devCount = 0, curDev = -1;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    CUDA_CHECK(cudaGetDevice(&curDev));
    printf("[LBVH] Device count = %d, current = %d\n", devCount, curDev);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("[LBVH] GPU mem free = %zu / %zu\n", freeMem, totalMem);

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, mortonKeys));
    printf("[LBVH] mortonKeys memoryType = %d\n", attr.type); // cudaMemoryTypeDevice 나와야 정상


    CUDA_SYNC();

    thrust::device_ptr<MortonKey> d_keys(mortonKeys);
    thrust::sort(thrust::device, mortonKeys, mortonKeys + numberOfMortonKeys, MortonKeyCompare());

    CUDA_MALLOC(&nodes, sizeof(LBVHNode) * allocatedNodes);
	CUDA_MEMSET(nodes, 0xFF, sizeof(LBVHNode)* allocatedNodes);

    LaunchKernel(Kernel_DeviceLBVH_InitializeLeafNodes, numberOfMortonKeys,
        mortonKeys, numberOfMortonKeys, nodes, aabbMin, aabbMax);

    LaunchKernel(Kernel_DeviceLBVH_InitializeInternalNodes, numberOfMortonKeys,
        mortonKeys, numberOfMortonKeys, nodes);

    //LaunchKernel(Kernel_DeviceLBVH_RefitAABB, numberOfMortonKeys,
    //    nodes, numberOfMortonKeys);

    {
        CUDA_TS(refit); // 타이머 시작

        // LaunchKernel 매크로 사용 (512 threads per block)
        LaunchKernel(Kernel_DeviceLBVH_RefitAABB, numberOfMortonKeys,
            nodes, numberOfMortonKeys);

        CUDA_CHECK(cudaGetLastError());
        CUDA_SYNC();          // cudaDeviceSynchronize()
        CUDA_TE(refit);       // 타이머 종료 및 출력
    }
}

void DeviceLBVH::Terminate()
{
    CUDA_SAFE_FREE(nodes);
    CUDA_SAFE_FREE(mortonKeys);

    allocatedNodes = 0;
    numberOfMortonKeys = 0;
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
    float3 aabbMin,
    float3 aabbMax)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numberOfMortonKeys) return;

    auto& node = nodes[numberOfMortonKeys - 1 + tid];
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

__global__ void Kernel_DeviceLBVH_InitializeInternalNodes(
    MortonKey* mortonKeys,
    unsigned int numberOfMortonKeys,
    LBVHNode* nodes)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numberOfMortonKeys - 1) return;

    unsigned int first, last;
    DeviceLBVH::FindRange(mortonKeys, numberOfMortonKeys, tid, first, last);

    unsigned int split = DeviceLBVH::FindSplit(mortonKeys, first, last);

    unsigned int leftIndex = split;
    unsigned int rightIndex = split + 1;

    if (std::min(first, last) == split)
    {
        leftIndex += numberOfMortonKeys - 1;
    }
    if (std::max(first, last) == split + 1)
    {
        rightIndex += numberOfMortonKeys - 1;
    }

    nodes[tid].leftNodeIndex = leftIndex;
    nodes[tid].rightNodeIndex = rightIndex;

    if (UINT32_MAX != nodes[leftIndex].parentNodeIndex)
    {
        printf("Error: left child %u already has parent %u (new parent %u)\n",
            leftIndex, nodes[leftIndex].parentNodeIndex, tid);

        return;
    }
    else
    {
        nodes[leftIndex].parentNodeIndex = tid;
    }
    if (UINT32_MAX != nodes[rightIndex].parentNodeIndex)
    {
        printf("Error: right child %u already has parent %u (new parent %u)\n",
            rightIndex, nodes[rightIndex].parentNodeIndex, tid);

        return;
    }
    else
    {
        nodes[rightIndex].parentNodeIndex = tid;
    }

	nodes[tid].aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    nodes[tid].aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    nodes[tid].pending = 2;
}

//__global__ void Kernel_DeviceLBVH_RefitAABB(LBVHNode* nodes, unsigned numberOfMortonKeys)
//{
//    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= numberOfMortonKeys - 1) return;
//
//    // 내부 노드를 뒤에서부터(큰 index → 작은 index) 순회하려면
//    unsigned int nodeIdx = (numberOfMortonKeys - 2) - idx;
//
//    const LBVHNode& L = nodes[nodes[nodeIdx].leftNodeIndex];
//    const LBVHNode& R = nodes[nodes[nodeIdx].rightNodeIndex];
//
//    nodes[nodeIdx].aabb = cuAABB::merge(L.aabb, R.aabb);
//}

__global__ void Kernel_DeviceLBVH_RefitAABB(
    LBVHNode* nodes, unsigned numLeaves)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) return;

    unsigned nodeIdx = (numLeaves - 1) + leafIdx; // leaf index
    unsigned parent = nodes[nodeIdx].parentNodeIndex;

    while (parent != UINT32_MAX)
    {
        int old = atomicSub(&(nodes[parent].pending), 1);
        if (old == 1)
        {
            LBVHNode& L = nodes[nodes[parent].leftNodeIndex ];
            LBVHNode& R = nodes[nodes[parent].rightNodeIndex];
            nodes[parent].aabb = cuAABB::merge(L.aabb, R.aabb);

            parent = nodes[parent].parentNodeIndex;
        }
        else
        {
            break;
        }
    }
}