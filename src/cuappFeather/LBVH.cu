#include <LBVH.cuh>

struct MortonKeyCompare {
    __host__ __device__
        bool operator()(const MortonKey& a, const MortonKey& b) const {
        if (a.code < b.code) return true;
        if (a.code > b.code) return false;
		if (a.index == b.index) printf("Error: duplicate morton key (code=%llu, index=%u)\n", a.code, a.index);
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
    float3 m = aabbMin;
	float3 M = aabbMax;

    CUDA_MALLOC(&mortonKeys, sizeof(MortonKey) * numberOfMortonKeys);
    CUDA_SYNC();

    LaunchKernel(Kernel_DeviceLBVH_BuildMortonKeys, numberOfMortonKeys,
        positions, faces, m, M, numberOfMortonKeys, mortonKeys);

	auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in Kernel_DeviceLBVH_BuildMortonKeys: %s\n", cudaGetErrorString(err));
    }

    allocatedNodes = 2 * numberOfMortonKeys - 1;

    CUDA_SYNC();

    thrust::sort(thrust::device,
        mortonKeys,
        mortonKeys + numberOfMortonKeys,
        MortonKeyCompare());

    CUDA_MALLOC(&nodes, sizeof(LBVHNode) * allocatedNodes);
	CUDA_MEMSET(nodes, 0xFF, sizeof(LBVHNode)* allocatedNodes);

    LaunchKernel(Kernel_DeviceLBVH_InitializeLeafNodes, numberOfMortonKeys,
        mortonKeys, numberOfMortonKeys, nodes, m, M);

    CUDA_SYNC();

    LaunchKernel(Kernel_DeviceLBVH_InitializeInternalNodes, numberOfMortonKeys,
        mortonKeys, numberOfMortonKeys, nodes);

    CUDA_SYNC();

    LaunchKernel(Kernel_DeviceLBVH_SetParentNodes, numberOfMortonKeys - 1, nodes, numberOfMortonKeys);

    CUDA_SYNC();

    {
        CUDA_TS(refit);

        LaunchKernel(Kernel_DeviceLBVH_RefitAABB, numberOfMortonKeys,
            nodes, numberOfMortonKeys);

        CUDA_CHECK(cudaGetLastError());
        CUDA_SYNC();
        CUDA_TE(refit);
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

    if (tid > 0) {
        if (mortonKeys[tid].code == mortonKeys[tid - 1].code &&
            mortonKeys[tid].index == mortonKeys[tid - 1].index) {
            printf("Duplicate detected at tid=%u (face=%u %u %u)\n",
                tid, face.x, face.y, face.z);
        }
    }
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

    node.pending = 0;
}

__global__ void Kernel_DeviceLBVH_InitializeInternalNodes(
    MortonKey* mortonKeys,
    unsigned int numberOfMortonKeys,
    LBVHNode* nodes)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfMortonKeys - 1) return;

    unsigned int first, last;
    DeviceLBVH::FindRange(mortonKeys, numberOfMortonKeys, tid, first, last);
    unsigned int split = DeviceLBVH::FindSplit(mortonKeys, first, last);

    unsigned int leftIndex = (split == first) ? (split + numberOfMortonKeys - 1) : split;
    unsigned int rightIndex = (split + 1 == last) ? (split + 1 + numberOfMortonKeys - 1) : (split + 1);

    nodes[tid].leftNodeIndex = leftIndex;
    nodes[tid].rightNodeIndex = rightIndex;
    nodes[tid].aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    nodes[tid].aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    nodes[tid].pending = 2;
}

__global__ void Kernel_DeviceLBVH_SetParentNodes(
    LBVHNode* nodes,
    unsigned int numberOfMortonKeys)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfMortonKeys - 1) return;

    unsigned int leftIndex = nodes[tid].leftNodeIndex;
    unsigned int rightIndex = nodes[tid].rightNodeIndex;

    if (leftIndex != UINT32_MAX) {
        unsigned int old = atomicCAS(&(nodes[leftIndex].parentNodeIndex),
            UINT32_MAX, tid);
        if (old != UINT32_MAX && old != tid) {
            printf("Conflict: left child %u already has parent %u (new %u)\n",
                leftIndex, old, tid);
        }
    }

    if (rightIndex != UINT32_MAX) {
        unsigned int old = atomicCAS(&(nodes[rightIndex].parentNodeIndex),
            UINT32_MAX, tid);
        if (old != UINT32_MAX && old != tid) {
            printf("Conflict: right child %u already has parent %u (new %u)\n",
                rightIndex, old, tid);
        }
    }
}

__global__ void Kernel_DeviceLBVH_RefitAABB(
    LBVHNode* nodes, unsigned numLeaves)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) return;

    unsigned nodeIdx = (numLeaves - 1) + leafIdx;
    unsigned parent = nodes[nodeIdx].parentNodeIndex;

    while (parent != UINT32_MAX)
    {
        int old = atomicSub(&(nodes[parent].pending), 1);
        if (old == 1)
        {
            LBVHNode& L = nodes[nodes[parent].leftNodeIndex];
            LBVHNode& R = nodes[nodes[parent].rightNodeIndex];
            nodes[parent].aabb = cuAABB::merge(L.aabb, R.aabb);

            parent = nodes[parent].parentNodeIndex;
        }
        else
        {
            return;
        }
    }
}
