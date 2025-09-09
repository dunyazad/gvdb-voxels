#pragma once

#include <cuda_common.cuh>
#include <SimpleHashMap.hpp>
#include <MortonCode.cuh>

#include <stack>

struct MortonKey
{
    uint64_t code;
    unsigned int index;

    __host__ __device__ bool operator<(const MortonKey& other) const
    {
        if (code < other.code) return true;
        if (code > other.code) return false;
        return index < other.index;
    }

    __host__ __device__
        bool operator==(const MortonKey& other) const {
        return (code == other.code && index == other.index);
    }
};

struct LBVHNode
{
    unsigned int parentNodeIndex = UINT32_MAX;
    unsigned int leftNodeIndex = UINT32_MAX;
    unsigned int rightNodeIndex = UINT32_MAX;
	unsigned pending = 2;

    cuAABB aabb = { make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
                    make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX) };
};

struct HostLBVH
{
    LBVHNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;

    void Initialize(const std::vector<MortonKey>& mortonKeys, float3 aabbMin, float3 aabbMax)
    {
        unsigned int n = (unsigned int)mortonKeys.size();
        allocatedNodes = 2 * n - 1;
        nodes = new LBVHNode[allocatedNodes];

        // Leaf nodes: [n-1, 2n-2]
        for (unsigned int i = 0; i < n; i++)
        {
            InitializeLeafNodes(
                n - 1 + i,
                mortonKeys[i],
                nodes,
                aabbMin,
                aabbMax
            );
        }

        // Internal nodes: [0, n-2]
        for (unsigned int i = 0; i < n - 1; i++)
        {
            InitializeInternalNodes(
                i,
                mortonKeys.data(),
                n,
                nodes
            );
        }

        RefitAABB(nodes, n);
    }

    void Terminate()
    {
        if (nodes != nullptr)
        {
            delete[] nodes;
            nodes = nullptr;
        }
    }

    static void InitializeLeafNodes(
        unsigned int leafNodeIndex,
        const MortonKey& mortonKey,
        LBVHNode* nodes,
        const float3& aabbMin,
        const float3& aabbMax)
    {
        LBVHNode& node = nodes[leafNodeIndex];

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

    static void InitializeInternalNodes(
        unsigned int internalIndex,
        const MortonKey* mortonKeys,
        unsigned int numberOfMortonCodes,
        LBVHNode* nodes)
    {
        unsigned int first, last;
        FindRange(mortonKeys, numberOfMortonCodes, internalIndex, first, last);

        unsigned int split = FindSplit(mortonKeys, first, last);

        unsigned int leftIndex = split;
        unsigned int rightIndex = split + 1;

        if (std::min(first, last) == split)
        {
            leftIndex += numberOfMortonCodes - 1;
        }
        if (std::max(first, last) == split + 1)
        {
            rightIndex += numberOfMortonCodes - 1;
        }

        nodes[internalIndex].leftNodeIndex = leftIndex;
        nodes[internalIndex].rightNodeIndex = rightIndex;

        if (UINT32_MAX != nodes[leftIndex].parentNodeIndex)
        {
            printf("Error: left child %u already has parent %u (new parent %u)\n",
				leftIndex, nodes[leftIndex].parentNodeIndex, internalIndex);

            return;
        }
        else
        {
            nodes[leftIndex].parentNodeIndex = internalIndex;
        }
        if (UINT32_MAX != nodes[rightIndex].parentNodeIndex)
        {
			printf("Error: right child %u already has parent %u (new parent %u)\n",
                rightIndex, nodes[rightIndex].parentNodeIndex, internalIndex);

            return;
		}
        else
        {
            nodes[rightIndex].parentNodeIndex = internalIndex;
        }
    }

    static unsigned int CountLeadingZeros64(uint64_t x)
    {
        if (x == 0) return 64;
#if defined(_MSC_VER)
        unsigned long index;
        _BitScanReverse64(&index, x);
        return 63 - index;
#else
        return __builtin_clzll(x);
#endif
    }

    static unsigned int CommonPrefixLength(const MortonKey& a, const MortonKey& b)
    {
        if (a.code == b.code) {
            return 64 + (a.index == b.index ? 32 : CountLeadingZeros64((uint64_t)(a.index ^ b.index)));
        }
        return CountLeadingZeros64(a.code ^ b.code);
    }

    static void FindRange(
        const MortonKey* mortonKeys,
        unsigned int numCodes,
        unsigned int index,
        unsigned int& first,
        unsigned int& last)
    {
        if (index == 0)
        {
            first = 0;
            last = numCodes - 1;
            return;
        }

        const MortonKey& key = mortonKeys[index];

        const int L_delta = CommonPrefixLength(key, mortonKeys[index - 1]);
        const int R_delta = CommonPrefixLength(key, mortonKeys[index + 1]);
        const int direction = (R_delta > L_delta) ? 1 : -1;

        const int delta_min = std::min(L_delta, R_delta);

        int l_max = 2;
        int delta = -1;
        int j = index + direction * l_max;
        if (0 <= j && j < (int)numCodes)
        {
            delta = CommonPrefixLength(key, mortonKeys[j]);
        }
        while (delta > delta_min)
        {
            l_max <<= 1;
            j = index + direction * l_max;
            delta = -1;
            if (0 <= j && j < (int)numCodes)
            {
                delta = CommonPrefixLength(key, mortonKeys[j]);
            }
        }

        int l = 0;
        int t = l_max >> 1;
        while (t > 0)
        {
            j = index + (l + t) * direction;
            delta = -1;
            if (0 <= j && j < (int)numCodes)
            {
                delta = CommonPrefixLength(key, mortonKeys[j]);
            }
            if (delta > delta_min)
            {
                l += t;
            }
            t >>= 1;
        }

        unsigned int jdx = index + l * direction;

        if (direction < 0)
        {
            std::swap(index, jdx);
        }

        first = index;
        last = jdx;
    }

    static unsigned int FindSplit(
        const MortonKey* mortonKeys,
        unsigned int first,
        unsigned int last)
    {
        const MortonKey& firstKey = mortonKeys[first];
        const MortonKey& lastKey = mortonKeys[last];

        if (firstKey.code == lastKey.code)
        {
            return (first + last) >> 1;
        }

        const int delta_node = CommonPrefixLength(firstKey, lastKey);

        int split = first;
        int stride = last - first;
        do
        {
            stride = (stride + 1) >> 1;
            const int middle = split + stride;
            if (middle < (int)last)
            {
                const int delta = CommonPrefixLength(firstKey, mortonKeys[middle]);
                if (delta > delta_node)
                {
                    split = middle;
                }
            }
        } while (stride > 1);

        return split;
    }

    static __forceinline__ unsigned FindRootInternal(const LBVHNode* nodes, unsigned n)
    {
        // 내부 노드 범위: [0, n-2]
        unsigned root = UINT32_MAX;
        for (unsigned i = 0; i < n - 1; ++i)
        {
            if (nodes[i].parentNodeIndex == UINT32_MAX) { root = i; break; }
        }
        return root;
    }

    static void RefitAABB(LBVHNode* nodes, unsigned numberOfMortonCodes)
    {
        const unsigned n = numberOfMortonCodes;
        const unsigned leafBegin = n - 1;
        const unsigned leafEnd = 2 * n - 2;

        unsigned root = FindRootInternal(nodes, n);
        if (root == UINT32_MAX) return;

        struct Frame { unsigned idx; bool expanded; };
        std::vector<Frame> stack;
        stack.reserve(n);
        stack.push_back({ root, false });

        while (!stack.empty())
        {
            auto [idx, expanded] = stack.back();
            stack.pop_back();

            const bool isLeaf = (idx >= leafBegin && idx <= leafEnd);
            if (isLeaf) continue;

            if (!expanded)
            {
                // 다시 나 자신을 'expanded=true'로 푸시하고, 자식들을 먼저 처리
                stack.push_back({ idx, true });
                stack.push_back({ nodes[idx].rightNodeIndex, false });
                stack.push_back({ nodes[idx].leftNodeIndex,  false });
            }
            else
            {
                // 양쪽 자식이 처리되었으므로 합치기
                const LBVHNode& L = nodes[nodes[idx].leftNodeIndex];
                const LBVHNode& R = nodes[nodes[idx].rightNodeIndex];
                nodes[idx].aabb = cuAABB::merge(L.aabb, R.aabb);
            }
        }
    }

    static bool ValidateBVH(const LBVHNode* nodes,
        unsigned int numberOfMortonCodes)
    {
        unsigned int nodeCount = 2 * numberOfMortonCodes - 1;
        unsigned int rootIndex = 0;

        std::vector<bool> visited(nodeCount, false);

        std::stack<unsigned int> stack;
        stack.push(rootIndex);
        visited[rootIndex] = true;

        while (!stack.empty())
        {
            unsigned int idx = stack.top();
            stack.pop();
            const LBVHNode& node = nodes[idx];

            if (idx == rootIndex)
            {
                if (node.parentNodeIndex != UINT32_MAX)
                {
                    printf("Error: root node %u should have no parent (got %u)!\n",
                        idx, node.parentNodeIndex);
                    return false;
                }
            }
            else
            {
                if (node.parentNodeIndex >= nodeCount)
                {
                    printf("Error: node %u has invalid parent %u!\n",
                        idx, node.parentNodeIndex);
                    return false;
                }
                const LBVHNode& parent = nodes[node.parentNodeIndex];
                if (parent.leftNodeIndex != idx && parent.rightNodeIndex != idx)
                {
                    printf("Error: node %u is not listed as a child of its parent %u!\n",
                        idx, node.parentNodeIndex);
                    return false;
                }
            }

            if (node.leftNodeIndex != UINT32_MAX)
            {
                if (nodes[node.leftNodeIndex].parentNodeIndex != idx)
                {
                    printf("Error: node %u left child %u has wrong parent %u!\n",
                        idx, node.leftNodeIndex,
                        nodes[node.leftNodeIndex].parentNodeIndex);
                    return false;
                }
                if (!visited[node.leftNodeIndex])
                {
                    visited[node.leftNodeIndex] = true;
                    stack.push(node.leftNodeIndex);
                }
            }

            if (node.rightNodeIndex != UINT32_MAX)
            {
                if (nodes[node.rightNodeIndex].parentNodeIndex != idx)
                {
                    printf("Error: node %u right child %u has wrong parent %u!\n",
                        idx, node.rightNodeIndex,
                        nodes[node.rightNodeIndex].parentNodeIndex);
                    return false;
                }
                if (!visited[node.rightNodeIndex])
                {
                    visited[node.rightNodeIndex] = true;
                    stack.push(node.rightNodeIndex);
                }
            }
        }

        for (unsigned int i = 0; i < nodeCount; i++)
        {
            if (!visited[i])
            {
                printf("Error: node %u is disconnected from root!\n", i);
                return false;
            }
        }

        return true;
    }

    static bool ExtraValidate(const LBVHNode* nodes, unsigned n)
    {
        const unsigned total = 2u * n - 1u;
        const unsigned leafBegin = n - 1;

        unsigned root = FindRootInternal(nodes, n);
        if (root == UINT32_MAX) { printf("Error: no root found!\n"); return false; }

        std::vector<char> visited(total, 0);
        std::vector<unsigned> st; st.reserve(n);
        st.push_back(root);

        while (!st.empty())
        {
            unsigned u = st.back(); st.pop_back();
            if (visited[u]) continue;
            visited[u] = 1;

            const bool isLeaf = (u >= leafBegin);
            if (!isLeaf)
            {
                unsigned L = nodes[u].leftNodeIndex;
                unsigned R = nodes[u].rightNodeIndex;
                if (L >= total || R >= total) { printf("Error: child index OOB\n"); return false; }
                st.push_back(L); st.push_back(R);
            }
        }

        for (unsigned i = 0; i < total; ++i)
        {
            if (!visited[i]) { printf("Error: node %u is disconnected from root!\n", i); return false; }
        }
        return true;
    }

    int NearestNeighbor(const float3& query, const std::vector<MortonKey>& mortonKeys, const std::vector<float3>& points, float3& nearestPosition)
    {
        if (!nodes) return -1;

        unsigned n = (unsigned)mortonKeys.size();
        unsigned root = FindRootInternal(nodes, n);
        if (root == UINT32_MAX) return -1;

        float bestDist2 = FLT_MAX;
        int bestIdx = -1;

        struct Frame { unsigned idx; float dist2; };
        std::stack<Frame> st;
        st.push({ root, 0.0f });

        while (!st.empty())
        {
            auto [idx, _] = st.top(); st.pop();
            const LBVHNode& node = nodes[idx];

            float nodeDist2 = cuAABB::Distance2(node.aabb, query);
            if (nodeDist2 >= bestDist2) continue; // prune

            if (idx >= n - 1) // leaf node
            {
                unsigned leafIdx = idx - (n - 1);
                const MortonKey& mk = mortonKeys[leafIdx];

                float3 objPos = points[mk.index];

                float dist2 = length2(query - objPos);
                if (dist2 < bestDist2)
                {
                    bestDist2 = dist2;
                    bestIdx = mk.index;
					nearestPosition = objPos;
                }
            }
            else // internal node
            {
                unsigned L = node.leftNodeIndex;
                unsigned R = node.rightNodeIndex;
                float dL = cuAABB::Distance2(nodes[L].aabb, query);
                float dR = cuAABB::Distance2(nodes[R].aabb, query);

                if (dL < dR) { st.push({ R, dR }); st.push({ L, dL }); }
                else { st.push({ L, dL }); st.push({ R, dR }); }
            }
        }
        return bestIdx;
    }
};



__global__ void Kernel_DeviceLBVH_BuildMortonKeys(
    float3* positions,
    uint3* faces,
    float3 aabbMin,
    float3 aabbMax,
    unsigned int numberOfMortonKeys,
    MortonKey* mortonKeys);

__global__ void Kernel_DeviceLBVH_InitializeLeafNodes(
    MortonKey* mortonKeys,
    unsigned int numberOfMortonKeys,
    LBVHNode* nodes,
    float3 aabbMin,
    float3 aabbMax);

__global__ void Kernel_DeviceLBVH_InitializeInternalNodes(
    MortonKey* mortonKeys,
    unsigned int numberOfMortonKeys,
	LBVHNode* nodes);

__global__ void Kernel_DeviceLBVH_SetParentNodes(
    LBVHNode* nodes,
    unsigned int numberOfMortonKeys);

__global__ void Kernel_DeviceLBVH_RefitAABB(
    LBVHNode* nodes,
    unsigned numberOfLeaves);

__global__ void Kernel_DeviceLBVH_NearestNeighbor(
    const LBVHNode* __restrict__ nodes,
    const MortonKey* __restrict__ mortonKeys,
    const float3* __restrict__ points,   // candidate point positions (device)
    unsigned int numMortonKeys,
    const float3* __restrict__ queries,  // query points (device)
    unsigned int numQueries,
    unsigned int* __restrict__ outIndices,
    float3* __restrict__ outPositions);

struct DeviceLBVH
{
    LBVHNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;
    MortonKey* mortonKeys = nullptr;
    unsigned int numberOfMortonKeys = 0;

    void Initialize(
        float3* positions,
        uint3* faces,
        const float3& aabbMin,
        const float3& aabbMax,
        unsigned int numberOfFaces);

    void Terminate();

    void NearestNeighbor(const float3* d_queries,
        unsigned int numQueries,
        const float3* d_points,
        unsigned int* d_outIndices,
        float3* d_outPositions);

	void ValidateHierarchy();

    __host__ __device__ static unsigned int CountLeadingZeros64(uint64_t x)
    {
        if (x == 0) return 64;

#ifdef __CUDA_ARCH__
        unsigned int hi = (unsigned int)(x >> 32);
        if (hi) {
            return __clz(hi);
        }
        else {
            unsigned int lo = (unsigned int)x;
            return 32 + __clz(lo);
        }
#else
#  if defined(_MSC_VER)
        unsigned long index;
        _BitScanReverse64(&index, x);
        return 63 - index;
#  else
        return __builtin_clzll(x);
#  endif
#endif
    }

    __host__ __device__ static unsigned int CommonPrefixLength(const MortonKey& a, const MortonKey& b)
    {
        if (a.code == b.code) {
            return 64 + CountLeadingZeros64((uint64_t)(a.index ^ b.index));
        }

        return CountLeadingZeros64(a.code ^ b.code);
    }

    __host__ __device__ static void FindRange(
        const MortonKey* mortonKeys,
        unsigned int numCodes,
        unsigned int index,
        unsigned int& first,
        unsigned int& last)
    {
        if (index == 0) {
            first = 0;
            last = numCodes - 1;
            return;
        }

        const MortonKey& key = mortonKeys[index];
        int L_delta = CommonPrefixLength(key, mortonKeys[index - 1]);
        int R_delta = CommonPrefixLength(key, mortonKeys[index + 1]);
        int direction = (R_delta > L_delta) ? 1 : -1;
        int delta_min = (L_delta < R_delta ? L_delta : R_delta);

        int l_max = 2;
        int j = index + direction * l_max;
        int delta = -1;
        if (j >= 0 && j < (int)numCodes) delta = CommonPrefixLength(key, mortonKeys[j]);

        while (delta > delta_min) {
            l_max <<= 1;
            j = index + direction * l_max;
            delta = -1;
            if (j >= 0 && j < (int)numCodes) delta = CommonPrefixLength(key, mortonKeys[j]);
        }

        int l = 0;
        int t = l_max >> 1;
        while (t > 0) {
            j = index + (l + t) * direction;
            delta = -1;
            if (j >= 0 && j < (int)numCodes) delta = CommonPrefixLength(key, mortonKeys[j]);
            if (delta > delta_min) l += t;
            t >>= 1;
        }

        unsigned int jdx = index + l * direction;
        if (direction < 0) {
            unsigned int tmp = index;
            index = jdx;
            jdx = tmp;
        }

        first = index;
        last = jdx;
    }

    __host__ __device__ static unsigned int FindSplit(
        const MortonKey* mortonKeys,
        unsigned int first,
        unsigned int last)
    {
        const MortonKey& firstKey = mortonKeys[first];
        const MortonKey& lastKey = mortonKeys[last];

        const int delta_node = CommonPrefixLength(firstKey, lastKey);

        int split = first;
        int stride = last - first;
        do {
            stride = (stride + 1) >> 1;
            const int middle = split + stride;
            if (middle < (int)last) {
                const int delta = CommonPrefixLength(firstKey, mortonKeys[middle]);
                if (delta > delta_node) {
                    split = middle;
                }
            }
        } while (stride > 1);

        return split;
    }

    static __host__ __device__ unsigned FindRootInternal(const LBVHNode* nodes, unsigned n)
    {
        // 내부 노드 범위: [0, n-2]
        unsigned root = UINT32_MAX;
        for (unsigned i = 0; i < n - 1; ++i)
        {
            if (nodes[i].parentNodeIndex == UINT32_MAX) { root = i; break; }
        }
        return root;
    }

    __host__ __device__ static void BruteForceRange(
        const MortonKey* mortonKeys,
        unsigned int numCodes,
        unsigned int index,
        unsigned int& first,
        unsigned int& last)
    {
        if (index == 0) {
            first = 0;
            last = numCodes - 1;
            return;
        }

        const MortonKey& key = mortonKeys[index];

        const int L_delta = CommonPrefixLength(key, mortonKeys[index - 1]);
        const int R_delta = CommonPrefixLength(key, mortonKeys[index + 1]);
        const int direction = (R_delta > L_delta) ? 1 : -1;

        const int delta_min = min(L_delta, R_delta);

        int j = index;
        while (true) {
            int next = j + direction;
            if (next < 0 || next >= (int)numCodes) break;

            int delta = CommonPrefixLength(key, mortonKeys[next]);
            if (delta <= delta_min) break;

            j = next;
        }

        if (direction > 0) {
            first = index;
            last = j;
        }
        else {
            first = j;
            last = index;
        }
    }

    __host__ __device__ static unsigned int FindSplitBruteForce(
        const MortonKey* mortonKeys,
        unsigned int first,
        unsigned int last)
    {
        if (first == last) return first; // degenerate

        const MortonKey& firstKey = mortonKeys[first];
        const MortonKey& lastKey = mortonKeys[last];

        if (firstKey.code == lastKey.code) {
            return (first + last) >> 1;
        }

        const int delta_node = CommonPrefixLength(firstKey, lastKey);

        unsigned int split = first;
        for (unsigned int i = first + 1; i < last; i++)
        {
            int delta = CommonPrefixLength(firstKey, mortonKeys[i]);
            if (delta > delta_node)
                split = i;
            else
                break;
        }

        return split;
    }
};
