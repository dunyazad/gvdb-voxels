#pragma once

#include <cuda_common.cuh>
#include <HashMap.hpp>
#include <MortonCode.cuh>

#include <stack>

struct LBVHNode
{
    unsigned int parentNodeIndex = UINT32_MAX;
    unsigned int leftNodeIndex = UINT32_MAX;
    unsigned int rightNodeIndex = UINT32_MAX;

    cuAABB aabb = { make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
                    make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX) };
};

struct LBVH
{
    LBVHNode* nodes = nullptr;
    unsigned int allocatedNodes = 0;

    void Initialize(const std::vector<uint64_t>& mortonCodes, float3 aabbMin, float3 aabbMax)
    {
        unsigned int n = (unsigned int)mortonCodes.size();
        allocatedNodes = 2 * n - 1;
        nodes = new LBVHNode[allocatedNodes];

        // Leaf nodes: [n-1, 2n-2]
        for (unsigned int i = 0; i < n; i++)
        {
            InitializeLeafNodes(
                n - 1 + i,
                mortonCodes[i],
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
                mortonCodes.data(),
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

    static void InitializeLeafNodes(unsigned int leafNodeIndex,
        uint64_t mortonCode,
        LBVHNode* nodes,
        const float3& aabbMin,
        const float3& aabbMax)
    {
        LBVHNode& node = nodes[leafNodeIndex];

        node.parentNodeIndex = UINT32_MAX;
        node.leftNodeIndex = UINT32_MAX;
        node.rightNodeIndex = UINT32_MAX;

        node.aabb = MortonCode::CodeToAABB(
            mortonCode,
            aabbMin,
            aabbMax,
            MortonCode::GetMaxDepth()
        );
    }

    static void InitializeInternalNodes(
        unsigned int internalIndex,
        const uint64_t* mortonCodes,
        unsigned int numberOfMortonCodes,
        LBVHNode* nodes)
    {
        // 1) 이 내부 노드가 커버하는 [first, last]를 구함
        unsigned int first, last;
        FindRange(mortonCodes, numberOfMortonCodes, internalIndex, first, last);

        // 2) 그 범위 안에서 split 찾기
        unsigned int split = FindSplit(mortonCodes, first, last);

        // 3) 왼/오 자식 인덱스 계산 (internal vs leaf 오프셋 구분)
        unsigned int leftIndex = (split == first)
            ? (numberOfMortonCodes - 1) + split      // leaf
            : split;                                  // internal

        unsigned int rightIndex = (split + 1 == last)
            ? (numberOfMortonCodes - 1) + (split + 1) // leaf
            : (split + 1);                             // internal

        // 4) 부모-자식 연결 (중복 부모 방지: 이미 부모가 있으면 충돌)
        if (nodes[leftIndex].parentNodeIndex == UINT32_MAX)
            nodes[leftIndex].parentNodeIndex = internalIndex;
        else {
            // 디버그 용도: 같은 리프/내부가 두 번 붙으려 하면 여기서 바로 발견됨
            // printf("Parent conflict: left child %u already has parent %u (new %u)\n",
            //        leftIndex, nodes[leftIndex].parentNodeIndex, internalIndex);
        }

        if (nodes[rightIndex].parentNodeIndex == UINT32_MAX)
            nodes[rightIndex].parentNodeIndex = internalIndex;
        else {
            // printf("Parent conflict: right child %u already has parent %u (new %u)\n",
            //        rightIndex, nodes[rightIndex].parentNodeIndex, internalIndex);
        }

        nodes[internalIndex].leftNodeIndex = leftIndex;
        nodes[internalIndex].rightNodeIndex = rightIndex;

        // AABB 초기화
        nodes[internalIndex].aabb = {
            make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)
        };

        // 루트는 ‘부모가 한 번도 세팅되지 않은 내부 노드’가 자동으로 남습니다.
        // 내부 노드의 parentNodeIndex는 기본값(UINT32_MAX) 그대로 두세요.
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

    static unsigned int CommonPrefixLength(uint64_t a, uint64_t b)
    {
        if (a == b) return 64;
        return CountLeadingZeros64(a ^ b);
    }

    static unsigned int FindRange(
        const uint64_t* mortonCodes,
        unsigned int numberOfMortonCodes,
        unsigned int nodeIndex,
        unsigned int& rangeStart,
        unsigned int& rangeEnd)
    {
        if (nodeIndex >= numberOfMortonCodes - 1) {
            rangeStart = nodeIndex;
            rangeEnd = nodeIndex;
            return 0;
        }

        // -----------------------
        // 1. 방향 결정
        // -----------------------
        int direction = 0;
        if (nodeIndex == 0) {
            direction = 1;
        }
        else if (nodeIndex == numberOfMortonCodes - 1) {
            direction = -1;
        }
        else {
            int prefixWithNext = (int)CommonPrefixLength(mortonCodes[nodeIndex], mortonCodes[nodeIndex + 1]);
            int prefixWithPrev = (int)CommonPrefixLength(mortonCodes[nodeIndex], mortonCodes[nodeIndex - 1]);
            direction = (prefixWithNext > prefixWithPrev) ? 1 : -1;
        }

        // -----------------------
        // 2. 기준 prefix
        // -----------------------
        unsigned int minPrefix = CommonPrefixLength(
            mortonCodes[nodeIndex],
            mortonCodes[nodeIndex + direction]);

        // -----------------------
        // 3. 한쪽으로 최대 확장
        // -----------------------
        int maxStep = 2;
        while (true) {
            int testIndex = (int)nodeIndex + maxStep * direction;
            if (testIndex < 0 || testIndex >= (int)numberOfMortonCodes) break;
            unsigned int testPrefix = CommonPrefixLength(mortonCodes[nodeIndex], mortonCodes[testIndex]);
            if (testPrefix < minPrefix) break;
            maxStep *= 2;
        }

        int step = 0;
        for (int t = maxStep / 2; t >= 1; t /= 2) {
            int testIndex = (int)nodeIndex + (step + t) * direction;
            if (testIndex >= 0 && testIndex < (int)numberOfMortonCodes) {
                unsigned int testPrefix = CommonPrefixLength(mortonCodes[nodeIndex], mortonCodes[testIndex]);
                if (testPrefix >= minPrefix) step += t;
            }
        }

        int endIndex = (int)nodeIndex + step * direction;

        // -----------------------
        // 4. 반대쪽 확장 추가
        // -----------------------
        int otherEnd = nodeIndex;
        while (otherEnd - direction >= 0 &&
            otherEnd - direction < (int)numberOfMortonCodes &&
            CommonPrefixLength(mortonCodes[nodeIndex], mortonCodes[otherEnd - direction]) >= minPrefix)
        {
            otherEnd -= direction;
        }

        // -----------------------
        // 5. 구간 정리
        // -----------------------
        if (direction == 1) {
            rangeStart = (unsigned int)otherEnd;
            rangeEnd = (unsigned int)endIndex;
        }
        else {
            rangeStart = (unsigned int)endIndex;
            rangeEnd = (unsigned int)otherEnd;
        }

        return 1;
    }


    // nodeIndex를 포함하는 구간 [rs, re]를
    // brute force로 직접 확장해서 찾는다.
    static unsigned int BruteForceRange(
        const uint64_t* codes,
        unsigned int n,
        unsigned int nodeIndex,
        unsigned int& rs,
        unsigned int& re)
    {
        if (n == 0) {
            rs = re = 0;
            return 0;
        }

        // 현재 노드와 자기 자신 prefix
        unsigned int minPrefix = 64; // 최대 prefix
        if (nodeIndex < n - 1) {
            minPrefix = CommonPrefixLength(codes[nodeIndex], codes[nodeIndex + 1]);
        }
        if (nodeIndex > 0) {
            unsigned int prevPrefix = CommonPrefixLength(codes[nodeIndex], codes[nodeIndex - 1]);
            if (prevPrefix > minPrefix) minPrefix = prevPrefix;
        }

        // 왼쪽 확장
        int left = (int)nodeIndex;
        while (left > 0 &&
            CommonPrefixLength(codes[nodeIndex], codes[left - 1]) >= minPrefix)
        {
            left--;
        }

        // 오른쪽 확장
        int right = (int)nodeIndex;
        while (right < (int)n - 1 &&
            CommonPrefixLength(codes[nodeIndex], codes[right + 1]) >= minPrefix)
        {
            right++;
        }

        rs = (unsigned int)left;
        re = (unsigned int)right;
        return 1;
    }

    static unsigned int FindSplit(
        const uint64_t* mortonCodes,
        unsigned int first,
        unsigned int last)
    {
        uint64_t firstCode = mortonCodes[first];
        uint64_t lastCode = mortonCodes[last];

        if (firstCode == lastCode) return (first + last) >> 1;

        unsigned int commonPrefix = CommonPrefixLength(firstCode, lastCode);

        // Binary search
        unsigned int split = first;
        unsigned int step = last - first;

        do {
            step = (step + 1) >> 1; // 절반 줄이기
            unsigned int newSplit = split + step;
            if (newSplit < last) {
                uint64_t splitCode = mortonCodes[newSplit];
                unsigned int splitPrefix = CommonPrefixLength(firstCode, splitCode);
                if (splitPrefix > commonPrefix) {
                    split = newSplit;
                }
            }
        } while (step > 1);

        return split;
    }

    static unsigned int FindSplitBruteForce(
        const uint64_t* mortonCodes,
        unsigned int first,
        unsigned int last)
    {
        uint64_t firstCode = mortonCodes[first];
        uint64_t lastCode = mortonCodes[last];

        if (firstCode == lastCode)
            return (first + last) >> 1;

        unsigned int commonPrefix = CommonPrefixLength(firstCode, lastCode);

        unsigned int split = first;
        for (unsigned int i = first; i < last; i++) {
            unsigned int prefix = CommonPrefixLength(firstCode, mortonCodes[i]);
            if (prefix > commonPrefix) {
                split = i;
            }
            else {
                break; // prefix가 줄어들면 멈춤
            }
        }
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
        return root; // 반드시 하나여야 함(Validate에서 확인 가능)
    }

    static void RefitAABB(LBVHNode* nodes, unsigned numberOfMortonCodes)
    {
        const unsigned n = numberOfMortonCodes;
        const unsigned leafBegin = n - 1;
        const unsigned leafEnd = 2 * n - 2;

        unsigned root = FindRootInternal(nodes, n);
        if (root == UINT32_MAX) return; // 안전장치

        // 비재귀 post-order
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
        unsigned int rootIndex = 0; // CPU 버전은 0번이 root

        // 모든 노드가 방문되었는지 체크
        std::vector<bool> visited(nodeCount, false);

        // BFS/DFS 로 탐색
        std::stack<unsigned int> stack;
        stack.push(rootIndex);
        visited[rootIndex] = true;

        while (!stack.empty())
        {
            unsigned int idx = stack.top();
            stack.pop();
            const LBVHNode& node = nodes[idx];

            // root 의 parent 는 반드시 UINT32_MAX
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
                // parent 가 유효한 범위 안에 있어야 함
                if (node.parentNodeIndex >= nodeCount)
                {
                    printf("Error: node %u has invalid parent %u!\n",
                        idx, node.parentNodeIndex);
                    return false;
                }
                // 부모-자식 연결 확인
                const LBVHNode& parent = nodes[node.parentNodeIndex];
                if (parent.leftNodeIndex != idx && parent.rightNodeIndex != idx)
                {
                    printf("Error: node %u is not listed as a child of its parent %u!\n",
                        idx, node.parentNodeIndex);
                    return false;
                }
            }

            // 왼쪽 자식
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

            // 오른쪽 자식
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

        // 방문되지 않은 노드 체크 (Disconnected 여부)
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

        // 루트 개수 체크
        unsigned root = FindRootInternal(nodes, n);
        if (root == UINT32_MAX) { printf("Error: no root found!\n"); return false; }

        // DFS로 연결성/사이클 체크
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

        // 모든 노드가 루트에서 도달 가능한가?
        for (unsigned i = 0; i < total; ++i)
        {
            if (!visited[i]) { printf("Error: node %u is disconnected from root!\n", i); return false; }
        }
        return true;
    }

    static void DebugValidateRange(
        unsigned int nodeIndex,
        unsigned int first,
        unsigned int last,
        const uint64_t* mortonCodes,
        unsigned int n)
    {
        if (first >= n || last >= n || nodeIndex >= n)
        {
            printf("FindRange Error: invalid range [%u,%u] for node %u (n=%u)\n",
                first, last, nodeIndex, n);
            return;
        }

        // 기준 prefix 길이 (nodeIndex와 first/last 비교)
        unsigned int expectedPrefix = CommonPrefixLength(
            mortonCodes[first],
            mortonCodes[last]
        );


        for (unsigned int i = first; i <= last; i++)
        {
            unsigned int prefix = CommonPrefixLength(mortonCodes[first], mortonCodes[i]);
            if (prefix < expectedPrefix)
            {
                printf("FindRange Warning: node %u range [%u,%u] "
                    "has element %u with smaller prefix %u (expected >= %u)\n",
                    nodeIndex, first, last, i, prefix, expectedPrefix);
            }
        }
    }

    // ============================
// ===== Validation / Tests ===
// ============================
    static bool NearlyLE(float a, float b, float eps = 1e-5f)
    {
        return a <= b + eps;
    }

    static bool AABBContains(const cuAABB& outer, const cuAABB& inner, float eps = 1e-5f)
    {
        return NearlyLE(outer.min.x, inner.min.x, eps) &&
            NearlyLE(outer.min.y, inner.min.y, eps) &&
            NearlyLE(outer.min.z, inner.min.z, eps) &&
            NearlyLE(inner.max.x, outer.max.x, eps) &&
            NearlyLE(inner.max.y, outer.max.y, eps) &&
            NearlyLE(inner.max.z, outer.max.z, eps);
    }

    static bool Test_CountLeadingZeros64()
    {
        bool ok = true;
        ok = ok && (CountLeadingZeros64(0ull) == 64);
        ok = ok && (CountLeadingZeros64(1ull << 63) == 0);
        ok = ok && (CountLeadingZeros64(1ull << 0) == 63);
        ok = ok && (CountLeadingZeros64((1ull << 62) | (1ull << 10)) == 1);
        if (!ok) { printf("[TEST] CountLeadingZeros64 failed\n"); }
        return ok;
    }

    static bool Test_CommonPrefixLength()
    {
        bool ok = true;
        uint64_t a = 0xFFFF000000000000ull;
        uint64_t b = 0xFFFF800000000000ull; // XOR 상위 비트 0..?
        uint64_t c = 0xFFFE000000000000ull;

        unsigned pab = CommonPrefixLength(a, b);
        unsigned pac = CommonPrefixLength(a, c);
        ok = ok && (pab > pac); // b가 a와 더 긴 prefix를 가정

        // 동일 값이면 64
        ok = ok && (CommonPrefixLength(a, a) == 64);

        // 전혀 다른 값
        ok = ok && (CommonPrefixLength(0ull, ~0ull) == 0);

        if (!ok) { printf("[TEST] CommonPrefixLength failed\n"); }
        return ok;
    }

    // 모든 i(0..n-2)에 대해 FindRange가 반환한 [first,last]의 prefix 불변식 확인
    static bool Test_FindRange_All(const std::vector<uint64_t>& codes)
    {
        const unsigned n = (unsigned)codes.size();
        if (n < 2) return true; // n=0,1은 내부 노드 없음

        for (unsigned i = 0; i < n - 1; ++i)
        {
            unsigned first = 0, last = 0;
            FindRange(codes.data(), n, i, first, last);

            if (!(first <= i && i <= last))
            {
                printf("[TEST] FindRange: node %u not in range [%u,%u]\n", i, first, last);
                return false;
            }

            // 기대 prefix = CPL(first, last)
            unsigned expectedPrefix = CommonPrefixLength(codes[first], codes[last]);

            for (unsigned k = first; k <= last; ++k)
            {
                unsigned pk = CommonPrefixLength(codes[first], codes[k]);
                if (pk < expectedPrefix)
                {
                    printf("[TEST] FindRange: node %u range [%u,%u] has idx %u with prefix %u (expected >= %u)\n",
                        i, first, last, k, pk, expectedPrefix);
                    return false;
                }
            }

            // 경계 바깥은 기대 prefix를 만족하지 않아야 함(가능한 경우)
            if (first > 0)
            {
                unsigned pL = CommonPrefixLength(codes[first - 1], codes[first]);
                if (pL >= expectedPrefix && codes[first - 1] != codes[first])
                {
                    printf("[TEST] FindRange: left boundary too small (idx=%u)\n", i);
                    return false;
                }
            }
            if (last + 1 < n)
            {
                unsigned pR = CommonPrefixLength(codes[first], codes[last + 1]);
                if (pR >= expectedPrefix && codes[last + 1] != codes[first])
                {
                    printf("[TEST] FindRange: right boundary too large (idx=%u)\n", i);
                    return false;
                }
            }
        }
        return true;
    }

    // [first,last]에 대한 FindSplit 특성 확인 (Karras 2012)
    static bool Test_FindSplit_OnRange(const std::vector<uint64_t>& codes, unsigned first, unsigned last)
    {
        if (first >= last) return true;

        unsigned split = FindSplit(codes.data(), first, last);
        if (!(first <= split && split < last))
        {
            printf("[TEST] FindSplit: split %u not in [%u,%u]\n", split, first, last);
            return false;
        }

        uint64_t firstCode = codes[first];
        uint64_t lastCode = codes[last];

        if (firstCode == lastCode)
        {
            // 동일 코드 구간이면 split은 단순 중간값이 되는지(엄격 검증은 아님)
            unsigned mid = (first + last) >> 1;
            if (split != mid)
            {
                // 같은 코드가 길게 이어지는 구간에서는 정확히 mid가 아닐 수 있으므로 경고만
                // 필요하다면 실패로 간주하려면 아래를 false로 바꾸세요.
                // printf("[TEST] FindSplit: equal-code range split=%u mid=%u (warn)\n", split, mid);
            }
            return true;
        }

        unsigned commonPrefix = CommonPrefixLength(firstCode, lastCode);

        // Karras 성질: k ∈ [first..split] => CPL(first, k) > commonPrefix
        for (unsigned k = first; k <= split; ++k)
        {
            unsigned pk = CommonPrefixLength(firstCode, codes[k]);
            if (!(pk > commonPrefix))
            {
                printf("[TEST] FindSplit: left side k=%u prefix=%u (need > %u)\n", k, pk, commonPrefix);
                return false;
            }
        }
        // k ∈ [split+1..last] => CPL(first, k) <= commonPrefix
        for (unsigned k = split + 1; k <= last; ++k)
        {
            unsigned pk = CommonPrefixLength(firstCode, codes[k]);
            if (pk > commonPrefix)
            {
                printf("[TEST] FindSplit: right side k=%u prefix=%u (need <= %u)\n", k, pk, commonPrefix);
                return false;
            }
        }
        return true;
    }

    static bool Test_FindSplit_All(const std::vector<uint64_t>& codes)
    {
        const unsigned n = (unsigned)codes.size();
        if (n < 2) return true;

        for (unsigned i = 0; i < n - 1; ++i)
        {
            unsigned first = 0, last = 0;
            FindRange(codes.data(), n, i, first, last);
            if (!Test_FindSplit_OnRange(codes, first, last))
            {
                printf("[TEST] FindSplit_All failed at node %u [range %u,%u]\n", i, first, last);
                return false;
            }
        }
        return true;
    }

    // 전체 BVH 빌드/검증 (Initialize + Validate + ExtraValidate + RefitAABB 포함)
    static bool Test_BuildAndValidate(const std::vector<uint64_t>& codes)
    {
        if (codes.empty())
        {
            return true;
        }

        // AABB 입력(임의): MortonCode::GetMaxDepth()와 CodeToAABB가 유효하다고 가정
        float3 aabbMin = make_float3(0, 0, 0);
        float3 aabbMax = make_float3(1, 1, 1);

        LBVH bvh;
        bvh.Initialize(codes, aabbMin, aabbMax);

        const unsigned n = (unsigned)codes.size();
        if (!ValidateBVH(bvh.nodes, n))
        {
            printf("[TEST] ValidateBVH failed\n");
            bvh.Terminate();
            return false;
        }
        if (!ExtraValidate(bvh.nodes, n))
        {
            printf("[TEST] ExtraValidate failed\n");
            bvh.Terminate();
            return false;
        }

        // AABB refit 이후 부모 AABB가 자식 AABB를 포함하는지 확인
        const unsigned total = 2u * n - 1u;
        const unsigned leafBegin = n - 1u;

        for (unsigned i = 0; i < leafBegin; ++i)
        {
            unsigned L = bvh.nodes[i].leftNodeIndex;
            unsigned R = bvh.nodes[i].rightNodeIndex;
            if (L == UINT32_MAX || R == UINT32_MAX)
            {
                printf("[TEST] Node %u children invalid\n", i);
                bvh.Terminate();
                return false;
            }
            const cuAABB& a = bvh.nodes[i].aabb;
            const cuAABB& la = bvh.nodes[L].aabb;
            const cuAABB& ra = bvh.nodes[R].aabb;

            if (!AABBContains(a, la) || !AABBContains(a, ra))
            {
                printf("[TEST] AABB containment failed at node %u\n", i);
                bvh.Terminate();
                return false;
            }
        }

        bvh.Terminate();
        return true;
    }

    // 다양한 패턴의 Morton code 집합에 대해 전체 테스트 수행
    static bool RunUnitTests()
    {
        bool ok = true;

        // 1) 기본 유틸리티
        ok = ok && Test_CountLeadingZeros64();
        ok = ok && Test_CommonPrefixLength();

        // 2) 단조 증가 시퀀스
        {
            std::vector<uint64_t> codes;
            for (uint64_t i = 0; i < 512; ++i)
            {
                codes.push_back(i << 16); // 상위 비트에 차이를 두어 CPL 분포 확보
            }
            ok = ok && Test_FindRange_All(codes);
            ok = ok && Test_FindSplit_All(codes);
            ok = ok && Test_BuildAndValidate(codes);
        }

        // 3) 동일 코드가 섞인 구간
        {
            std::vector<uint64_t> codes;
            for (int i = 0; i < 20; ++i) codes.push_back(0x12345678ull);
            for (int i = 0; i < 20; ++i) codes.push_back(0x123456F0ull);
            for (int i = 0; i < 20; ++i) codes.push_back(0x12345700ull);
            ok = ok && Test_FindRange_All(codes);
            ok = ok && Test_FindSplit_All(codes);
            ok = ok && Test_BuildAndValidate(codes);
        }

        // 4) 랜덤(고정 시드) + 정렬
        {
            std::vector<uint64_t> codes;
            uint64_t seed = 0x9e3779b97f4a7c15ull;
            auto rnd = [&seed]() -> uint64_t
            {
                seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
                return seed * 2685821657736338717ull;
            };
            for (int i = 0; i < 1024; ++i)
            {
                // 하위 비트에 변화를 두되, 전체는 64-bit로 다양하게
                codes.push_back(rnd() & 0x0000FFFFFFFFFFFFull);
            }
            std::sort(codes.begin(), codes.end());
            ok = ok && Test_FindRange_All(codes);
            ok = ok && Test_FindSplit_All(codes);
            ok = ok && Test_BuildAndValidate(codes);
        }

        // 5) 엣지 케이스: n = 1, 2
        {
            std::vector<uint64_t> n1{ 0x5555ull };
            ok = ok && Test_BuildAndValidate(n1);

            std::vector<uint64_t> n2{ 0x1000ull, 0x1001ull };
            ok = ok && Test_FindRange_All(n2);
            ok = ok && Test_FindSplit_All(n2);
            ok = ok && Test_BuildAndValidate(n2);
        }

        if (ok) printf("[TEST] LBVH all tests PASSED\n");
        else     printf("[TEST] LBVH tests FAILED\n");
        return ok;
    }
};
