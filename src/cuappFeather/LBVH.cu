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

    

    // 1. 정렬
    thrust::sort(thrust::device,
        mortonKeys,
        mortonKeys + numberOfMortonKeys,
        [] __host__ __device__(const MortonKey & a, const MortonKey & b) {
        if (a.code < b.code) return true;
        if (a.code > b.code) return false;
        return a.index < b.index;
    });

    // 2. unique 호출 (중복 제거)
    auto new_end = thrust::unique(thrust::device,
        mortonKeys,
        mortonKeys + numberOfMortonKeys,
        [] __host__ __device__(const MortonKey & a, const MortonKey & b) {
        return (a.code == b.code && a.index == b.index);
    });

    // 3. 중복 개수
    size_t num_unique = new_end - mortonKeys;
    size_t num_total = numberOfMortonKeys;
    size_t num_duplicates = num_total - num_unique;

    if (num_duplicates > 0) {
        printf("LBVH: Found %zu duplicate MortonKeys (removed)\n", num_duplicates);
    }

    // 4. 중복 제거 반영
    numberOfMortonKeys = static_cast<unsigned int>(num_unique);

    // 5. 노드 메모리 다시 할당
    allocatedNodes = 2 * numberOfMortonKeys - 1;
    CUDA_MALLOC(&nodes, sizeof(LBVHNode) * allocatedNodes);
    CUDA_MEMSET(nodes, 0xFF, sizeof(LBVHNode) * allocatedNodes);




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

void DeviceLBVH::NearestNeighbor(const float3* d_queries,
    unsigned int numQueries,
    const float3* d_points,
    unsigned int* d_outIndices,
    float3* d_outPositions)
{
    LaunchKernel(Kernel_DeviceLBVH_NearestNeighbor, numQueries,
        nodes,
        mortonKeys,
        d_points,
        numberOfMortonKeys,
        d_queries,
        numQueries,
        d_outIndices,
        d_outPositions);

    CUDA_CHECK(cudaGetLastError());
    CUDA_SYNC();
}

__global__ void Kernel_DeviceLBVH_ValidateBottomUp(
    const LBVHNode* nodes,
    unsigned int numberOfMortonKeys,
    int* errorFlags)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numberOfMortonKeys) return;

    unsigned int nodeIndex = numberOfMortonKeys - 1 + tid; // leaf
    unsigned int parentNodeIndex = nodes[nodeIndex].parentNodeIndex;
    int steps = 0;

    unsigned int footprints[256];
	footprints[steps] = nodeIndex;
    while (parentNodeIndex != UINT32_MAX && parentNodeIndex != 0) {
        if (++steps > 2 * (32 - __clz(numberOfMortonKeys))) { // ~2*log2(N)
            errorFlags[tid] = 1; // infinite loop / invalid hierarchy
            return;
        }

        footprints[steps] = parentNodeIndex;
        parentNodeIndex = nodes[parentNodeIndex].parentNodeIndex;
    }

	// verify downwards links
    for (int i = steps; i > 0; i--) {
        unsigned int idx = footprints[i];
        unsigned int L = nodes[idx].leftNodeIndex;
        unsigned int R = nodes[idx].rightNodeIndex;
        if (L == UINT32_MAX || R == UINT32_MAX) {
            errorFlags[tid] = 2; // missing child
            return;
        }
        if (nodes[L].parentNodeIndex != idx) {
            errorFlags[tid] = 3; // left child has wrong parent
            return;
        }
        if (nodes[R].parentNodeIndex != idx) {
            errorFlags[tid] = 4; // right child has wrong parent
            return;
        }
	}
}

void DeviceLBVH::ValidateHierarchy()
{
  //  LaunchKernel(Kernel_DeviceLBVH_ValidateBottomUp, numberOfMortonKeys,
		//nodes, numberOfMortonKeys);




    std::vector<int> h_error(numberOfMortonKeys, 0);
    int* d_error;
    cudaMalloc(&d_error, sizeof(int) * numberOfMortonKeys);
    cudaMemset(d_error, 0, sizeof(int) * numberOfMortonKeys);

    LaunchKernel(Kernel_DeviceLBVH_ValidateBottomUp, numberOfMortonKeys,
        nodes, numberOfMortonKeys, d_error);

    cudaMemcpy(h_error.data(), d_error, sizeof(int) * numberOfMortonKeys, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < numberOfMortonKeys; i++) {
        if (h_error[i]) {
            std::cout << "Validation error(" << h_error[i] << ") in leaf : " << i << "\n";
        }
    }
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

__global__ void Kernel_DeviceLBVH_NearestNeighbor(
    const LBVHNode* __restrict__ nodes,
    const MortonKey* __restrict__ mortonKeys,
    const float3* __restrict__ points,   // candidate point positions (device)
    unsigned int numMortonKeys,
    const float3* __restrict__ queries,  // query points (device)
    unsigned int numQueries,
    unsigned int* __restrict__ outIndices,
    float3* __restrict__ outPositions)
{
    unsigned qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= numQueries) return;

    float3 query = queries[qid];
    float bestDist2 = FLT_MAX;
    int bestIdx = -1;

    // per-thread stack
    const int STACK_SIZE = 128;
    unsigned stack[STACK_SIZE];
    int sp = 0;

    unsigned root = DeviceLBVH::FindRootInternal(nodes, numMortonKeys);
    if (root == UINT32_MAX) {
        outIndices[qid] = -1;
        outPositions[qid] = make_float3(0, 0, 0);
        return;
    }

    stack[sp++] = root;

	int fallbackCount = 0;

    while (sp > 0)
    {
        fallbackCount++;
		if (fallbackCount > 1000) {
			//printf("Error: fallback in nearest neighbor search (qid=%u)\n", qid);
			break;
		}

        unsigned idx = stack[--sp];
        const LBVHNode& node = nodes[idx];

        float nodeDist2 = cuAABB::Distance2(node.aabb, query);
        if (nodeDist2 >= bestDist2) continue; // prune

        if (idx >= numMortonKeys - 1) // leaf node
        {
            unsigned leafIdx = idx - (numMortonKeys - 1);
            const MortonKey& mk = mortonKeys[leafIdx];
            float3 objPos = points[mk.index];

            float d2 = length2(make_float3(query.x - objPos.x,
                query.y - objPos.y,
                query.z - objPos.z));
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestIdx = mk.index;
            }
        }
        else // internal node
        {
            unsigned L = node.leftNodeIndex;
            unsigned R = node.rightNodeIndex;
            if (L == UINT32_MAX || R == UINT32_MAX) continue;

            if(idx == L || idx == R) {
                printf("Error: self-loop detected at node %u (L=%u, R=%u)\n", idx, L, R);
                continue;
			}

            float dL = cuAABB::Distance2(nodes[L].aabb, query);
            float dR = cuAABB::Distance2(nodes[R].aabb, query);

            // push closer child last so it's popped first
            if (dL < dR) {
                if (dR < bestDist2 && sp < STACK_SIZE) stack[sp++] = R;
                if (dL < bestDist2 && sp < STACK_SIZE) stack[sp++] = L;
            }
            else {
                if (dL < bestDist2 && sp < STACK_SIZE) stack[sp++] = L;
                if (dR < bestDist2 && sp < STACK_SIZE) stack[sp++] = R;
            }
        }

        if (sp >= STACK_SIZE) {
            printf("Warning: stack overflow in nearest neighbor search (qid=%u)\n", qid);
            break;
		}
    }

    outIndices[qid] = bestIdx;
    outPositions[qid] = (bestIdx >= 0 ? points[bestIdx] : make_float3(0, 0, 0));
}