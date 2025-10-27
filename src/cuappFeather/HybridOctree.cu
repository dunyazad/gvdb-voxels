#include <HybridOctree.cuh>

#include <numeric>

#include <IVisualDebugging.h>
using VD = IVisualDebugging;

#define STACK_MAX 1024

__device__ unsigned int QueryPoint(const HybridDeviceOctreeNode* nodes, unsigned int nodeCount, const float3& p);

__global__ void Kernel_QueryPoint(const HybridDeviceOctreeNode* nodes,
    unsigned int nodeCount, const float3* queryPoints, unsigned int queryCount, unsigned int* outNodeIndices);

__device__ void QueryRange(const HybridDeviceOctreeNode* nodes,
    unsigned int nodeCount, const cuAABB& range, unsigned int* outIndices, unsigned int maxOut, unsigned int& outCount);

__global__ void Kernel_QueryRange(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const cuAABB* ranges, unsigned int queryCount,
    unsigned int* outIndices /*[queryCount * maxOutPerQuery]*/,
    unsigned int* outCounts  /*[queryCount]*/,
    unsigned int maxOutPerQuery);

__device__ void QueryRadius(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3& center, float radius,
    unsigned int* outIndices, unsigned int maxOut, unsigned int& outCount);

__global__ void Kernel_QueryRadius(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* centers, const float* radii, unsigned int queryCount,
    unsigned int* outIndices, unsigned int* outCounts, unsigned int maxOutPerQuery);

__device__ unsigned int QueryNearestPointDevice(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* points, unsigned int pointCount,
    const unsigned int* flatIndices,
    const float3& p);

__global__ void Kernel_QueryNearestPoint(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* points, unsigned int pointCount,
    const unsigned int* flatIndices,
    const float3* queryPoints, unsigned int queryCount,
    unsigned int* outNearest);
















HybridOctree::HybridOctree(unsigned int maxDepth)
	: maxDepth(maxDepth)
{
	interpolatedColors = Color::InterpolateColors({ Color::blue(), Color::red() }, maxDepth);
}

HybridOctree::~HybridOctree()
{
	CUDA_SAFE_FREE(device_nodes);
	CUDA_SAFE_FREE(device_flatIndices);
}

void HybridOctree::Subdivide(
    unsigned int nodeIndex,
    const std::vector<float3>& points,
    unsigned int maxPoints, unsigned int maxDepth, unsigned int depth)
{
    HybridHostOctreeNode& node = host_nodes[nodeIndex];

    // 종료 조건: 리프로 유지
    if (node.indices.size() <= (size_t)maxPoints || depth >= maxDepth)
        return;

    const float3 c = (node.bounds.min + node.bounds.max) * 0.5f;
    const float3 mn = node.bounds.min;
    const float3 mx = node.bounds.max;

    // 1) 임시 버킷에만 분배 (원본 indices는 자식 생성 확정 전까지 유지)
    std::vector<unsigned int> childIdx[8];
    for (unsigned int idx : node.indices)
    {
        const float3& p = points[idx];
        const unsigned int ox = (p.x >= c.x) ? 1 : 0;
        const unsigned int oy = (p.y >= c.y) ? 1 : 0;
        const unsigned int oz = (p.z >= c.z) ? 1 : 0;
        const unsigned int oct = (ox) | (oy << 1) | (oz << 2);
        childIdx[oct].push_back(idx);
    }

    // 2) 실제로 생성되는 자식 수를 계산
    unsigned int created = 0;
    unsigned int newChildIndex[8]; // 생성된 경우에만 인덱스 기록
    std::fill(std::begin(newChildIndex), std::end(newChildIndex), -1);

    const float eps = 1e-7f;

    for (unsigned int i = 0; i < 8; ++i)
    {
        if (childIdx[i].empty()) continue;

        const unsigned int ox = i & 1;
        const unsigned int oy = (i >> 1) & 1;
        const unsigned int oz = (i >> 2) & 1;

        const float3 childMin = make_float3(
            ox ? c.x : mn.x,
            oy ? c.y : mn.y,
            oz ? c.z : mn.z);

        const float3 childMax = make_float3(
            ox ? mx.x : c.x,
            oy ? mx.y : c.y,
            oz ? mx.z : c.z);

        // 너무 얇은 셀은 스킵
        if ((childMax.x - childMin.x) < eps ||
            (childMax.y - childMin.y) < eps ||
            (childMax.z - childMin.z) < eps)
        {
            continue;
        }

        // 자식 노드 생성
        const unsigned int newIdx = (int)host_nodes.size();
        host_nodes.emplace_back();

        HybridHostOctreeNode& child = host_nodes.back();
        child.bounds = { childMin, childMax };
        child.parent = (unsigned int)nodeIndex;
        child.indices = std::move(childIdx[i]);

        newChildIndex[i] = newIdx;
        ++created;
    }

    // 3) 자식이 하나도 없다면 리프로 유지하고 반환 (인덱스 보존!)
    if (created == 0)
    {
        node.isLeaf = 1;
        return;
    }

    // 4) 자식이 하나 이상이면 내부 노드로 전환하고 인덱스 비움
    node.isLeaf = 0;
    //node.indices.clear();
    for (unsigned int i = 0; i < 8; ++i)
        node.children[i] = newChildIndex[i];

    // 5) 재귀 분할
    for (unsigned int i = 0; i < 8; ++i)
    {
        const unsigned int cidx = node.children[i];
        if (cidx != -1)
            Subdivide(cidx, points, maxPoints, maxDepth, depth + 1);
    }
}

void HybridOctree::Build(
	const std::vector<float3>& points,
	const cuAABB& aabb,
    unsigned int maxPoints, unsigned int maxDepth)
{
    if (this->maxDepth != maxDepth) {
        this->maxDepth = maxDepth;
        interpolatedColors = Color::InterpolateColors({ Color::blue(), Color::red() }, this->maxDepth);
    }

    host_nodes.clear();
	host_nodes.reserve(points.size() * 8 / (maxPoints == 0 ? 1 : maxPoints));
	host_nodes.emplace_back();

	HybridHostOctreeNode& root = host_nodes[0];
	root.bounds = aabb;
	root.indices.resize(points.size());
	std::iota(root.indices.begin(), root.indices.end(), 0);

	Subdivide(0, points, maxPoints, maxDepth, 0);
	printf("[Octree] Total Nodes: %zu\n", host_nodes.size());



	unsigned int nodeCount = host_nodes.size();

	std::vector<unsigned int> flatIndices;
	flatIndices.reserve(points.size());

	std::vector<HybridDeviceOctreeNode> nodesToCopy(nodeCount);
	size_t offset = 0;
	for (size_t i = 0; i < nodeCount; ++i)
	{
		const HybridHostOctreeNode& src = host_nodes[i];
		HybridDeviceOctreeNode& dst = nodesToCopy[i];

		dst.bounds = src.bounds;
		dst.parent = src.parent;
		memcpy(dst.children, src.children.data(), sizeof(unsigned int) * 8);
		dst.isLeaf = src.isLeaf;

		dst.V = src.V;
		dst.divergence = src.divergence;
		dst.chi = src.chi;

		dst.indexOffset = (unsigned int)offset;
		dst.indexCount = (unsigned int)src.indices.size();

		flatIndices.insert(flatIndices.end(), src.indices.begin(), src.indices.end());
		offset += src.indices.size();
	}

    CUDA_SAFE_FREE(device_nodes);
    CUDA_SAFE_FREE(device_flatIndices);

    if (nodeCount > 0)
    {
        CUDA_MALLOC(&device_nodes, sizeof(HybridDeviceOctreeNode) * nodeCount);
        CUDA_COPY_H2D(device_nodes, nodesToCopy.data(), sizeof(HybridDeviceOctreeNode) * nodeCount);
    }
    else
    {
        device_nodes = nullptr;
    }

    if (!flatIndices.empty())
    {
        CUDA_MALLOC(&device_flatIndices, sizeof(unsigned int) * flatIndices.size());
        CUDA_COPY_H2D(device_flatIndices, flatIndices.data(), sizeof(unsigned int) * flatIndices.size());
    }
    else
    {
        device_flatIndices = nullptr;
    }
}

void HybridOctree::DrawNode(unsigned int nodeIndex, unsigned int depth)
{
	const HybridHostOctreeNode& node = host_nodes[nodeIndex];
    glm::vec4 color = glm::vec4(1, 1, 1, 1);
    if (!interpolatedColors.empty())
        color = interpolatedColors[depth % interpolatedColors.size()];

	VD::AddWiredBox(
		"octree_" + std::to_string(depth),
		{ glm::vec3(XYZ(node.bounds.min)), glm::vec3(XYZ(node.bounds.max)) },
		color
	);

	if (0 == node.isLeaf)
	{
		for (unsigned int i = 0; i < 8; ++i)
		{
            unsigned int c = node.children[i];
			if (c != -1)
				DrawNode(c, depth + 1);
		}
	}
}

void HybridOctree::Draw()
{
    if (host_nodes.empty()) return;

	DrawNode(0, 0);
}

std::vector<unsigned int> HybridOctree::QueryPoints(const std::vector<float3>& queryPoints)
{
    if (!device_nodes || queryPoints.empty()) return {};

    unsigned int queryCount = (unsigned int)queryPoints.size();

    float3* d_points = nullptr;
    unsigned int* d_results = nullptr;

    CUDA_MALLOC(&d_points, sizeof(float3) * queryCount);
    CUDA_MALLOC(&d_results, sizeof(unsigned int) * queryCount);

    CUDA_COPY_H2D(d_points, queryPoints.data(), sizeof(float3) * queryCount);

    LaunchKernel(Kernel_QueryPoint, queryCount,
        device_nodes,
        (int)host_nodes.size(),
        d_points,
        queryCount,
        d_results);

    std::vector<unsigned int> results(queryCount);
    CUDA_COPY_D2H(results.data(), d_results, sizeof(unsigned int) * queryCount);

    CUDA_SYNC();

    // 메모리 정리
    CUDA_SAFE_FREE(d_points);
    CUDA_SAFE_FREE(d_results);

    return results;
}

void HybridOctree::QueryRange(
    const std::vector<cuAABB>& ranges,
    unsigned int maxOutPerQuery,
    std::vector<unsigned int>& outFlatIndices,
    std::vector<unsigned int>& outCounts)
{
    if (!device_nodes || ranges.empty() || maxOutPerQuery <= 0) { outFlatIndices.clear(); outCounts.clear(); return; }

    unsigned int queryCount = (unsigned int)ranges.size();
    cuAABB* d_ranges = nullptr;
    unsigned int* d_idx = nullptr;
    unsigned int* d_counts = nullptr;

    CUDA_MALLOC(&d_ranges, sizeof(cuAABB) * queryCount);
    CUDA_COPY_H2D(d_ranges, ranges.data(), sizeof(cuAABB) * queryCount);

    CUDA_MALLOC(&d_idx, sizeof(unsigned int) * queryCount * maxOutPerQuery);
    CUDA_MALLOC(&d_counts, sizeof(unsigned int) * queryCount);

    LaunchKernel(Kernel_QueryRange, queryCount,
        device_nodes, (unsigned int)host_nodes.size(),
        d_ranges, queryCount,
        d_idx, d_counts, maxOutPerQuery);

    outFlatIndices.resize((size_t)queryCount * maxOutPerQuery);
    outCounts.resize(queryCount);
    CUDA_COPY_D2H(outFlatIndices.data(), d_idx, sizeof(unsigned int) * queryCount * maxOutPerQuery);
    CUDA_COPY_D2H(outCounts.data(), d_counts, sizeof(unsigned int) * queryCount);
    CUDA_SYNC();

    CUDA_SAFE_FREE(d_ranges);
    CUDA_SAFE_FREE(d_idx);
    CUDA_SAFE_FREE(d_counts);
}

void HybridOctree::QueryRadius(
    const std::vector<float3>& centers,
    const std::vector<float>& radii,
    unsigned int maxOutPerQuery,
    std::vector<unsigned int>& outFlatIndices,
    std::vector<unsigned int>& outCounts)
{
    unsigned int q = (unsigned int)centers.size();
    if (!device_nodes || q == 0 || (unsigned int)radii.size() != q || maxOutPerQuery <= 0) { outFlatIndices.clear(); outCounts.clear(); return; }

    float3* d_centers = nullptr;
    float* d_radii = nullptr;
    unsigned int* d_idx = nullptr;
    unsigned int* d_counts = nullptr;
    CUDA_MALLOC(&d_centers, sizeof(float3) * q);
    CUDA_MALLOC(&d_radii, sizeof(float) * q);
    CUDA_MALLOC(&d_idx, sizeof(unsigned int) * q * maxOutPerQuery);
    CUDA_MALLOC(&d_counts, sizeof(unsigned int) * q);

    CUDA_COPY_H2D(d_centers, centers.data(), sizeof(float3) * q);
    CUDA_COPY_H2D(d_radii, radii.data(), sizeof(float) * q);

    LaunchKernel(Kernel_QueryRadius, q,
        device_nodes, (unsigned int)host_nodes.size(), d_centers, d_radii, q,
        d_idx, d_counts, maxOutPerQuery);

    outFlatIndices.resize((size_t)q * maxOutPerQuery);
    outCounts.resize(q);
    CUDA_COPY_D2H(outFlatIndices.data(), d_idx, sizeof(unsigned int) * q * maxOutPerQuery);
    CUDA_COPY_D2H(outCounts.data(), d_counts, sizeof(unsigned int) * q);
    CUDA_SYNC();

    CUDA_SAFE_FREE(d_centers);
    CUDA_SAFE_FREE(d_radii);
    CUDA_SAFE_FREE(d_idx);
    CUDA_SAFE_FREE(d_counts);
}

std::vector<unsigned int> HybridOctree::QueryNearestPoint(const std::vector<float3>& queryPoints,
    const std::vector<float3>& points)
{
    std::vector<unsigned int> out;
    if (!device_nodes || !device_flatIndices || queryPoints.empty() || points.empty()) return out;

    unsigned int q = (unsigned int)queryPoints.size();
    unsigned int p = (unsigned int)points.size();

    float3* d_points = nullptr;
    float3* d_queries = nullptr;
    unsigned int* d_out = nullptr;

    CUDA_MALLOC(&d_points, sizeof(float3) * p);
    CUDA_MALLOC(&d_queries, sizeof(float3) * q);
    CUDA_MALLOC(&d_out, sizeof(unsigned int) * q);

    CUDA_COPY_H2D(d_points, points.data(), sizeof(float3) * p);
    CUDA_COPY_H2D(d_queries, queryPoints.data(), sizeof(float3) * q);

    LaunchKernel(Kernel_QueryNearestPoint, q,
        device_nodes, (unsigned int)host_nodes.size(),
        d_points, p,
        device_flatIndices,
        d_queries, q,
        d_out);

    out.resize(q);
    CUDA_COPY_D2H(out.data(), d_out, sizeof(unsigned int) * q);
    CUDA_SYNC();

    CUDA_SAFE_FREE(d_points);
    CUDA_SAFE_FREE(d_queries);
    CUDA_SAFE_FREE(d_out);
    return out;
}






__device__ unsigned int QueryPoint(const HybridDeviceOctreeNode* nodes, unsigned int nodeCount, const float3& p)
{
    unsigned int current = 0;
    while (true)
    {
        if (current < 0 || current >= nodeCount)
            return UINT32_MAX;

        const HybridDeviceOctreeNode& node = nodes[current];

        // 점이 현재 노드 bounds에 포함되지 않으면 실패
        if (!(p.x >= node.bounds.min.x && p.x <= node.bounds.max.x &&
            p.y >= node.bounds.min.y && p.y <= node.bounds.max.y &&
            p.z >= node.bounds.min.z && p.z <= node.bounds.max.z))
        {
            return UINT32_MAX;
        }

        if (node.isLeaf)
            return current;

        bool found = false;
        for (unsigned int i = 0; i < 8; ++i)
        {
            unsigned int c = node.children[i];
            if (c == UINT32_MAX) continue;

            const HybridDeviceOctreeNode& child = nodes[c];
            if (p.x >= child.bounds.min.x && p.x <= child.bounds.max.x &&
                p.y >= child.bounds.min.y && p.y <= child.bounds.max.y &&
                p.z >= child.bounds.min.z && p.z <= child.bounds.max.z)
            {
                current = c;
                found = true;
                break;
            }
        }
        if (!found)
            return current; // 해당 자식이 없으면 현재 노드 리턴
    }
}

__global__ void Kernel_QueryPoint(
    const HybridDeviceOctreeNode* nodes,
    unsigned int nodeCount,
    const float3* queryPoints,
    unsigned int queryCount,
    unsigned int* outNodeIndices)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;

    outNodeIndices[tid] = QueryPoint(nodes, nodeCount, queryPoints[tid]);
}





__device__ void QueryRange(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const cuAABB& range,
    unsigned int* outIndices, unsigned int maxOut, unsigned int& outCount)
{
    unsigned int stack[STACK_MAX];
    unsigned int sp = 0;
    stack[sp++] = 0;
    outCount = 0;

    while (sp > 0)
    {
        unsigned int idx = stack[--sp];
        if (idx >= nodeCount) continue;

        const auto& node = nodes[idx];
        if (!node.bounds.intersects(range)) continue;

        if (node.isLeaf)
        {
            if (outCount < maxOut)
                outIndices[outCount] = idx;
            ++outCount;
        }
        else
        {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i)
            {
                unsigned int c = node.children[i];
                if (c != UINT32_MAX && sp < STACK_MAX)
                    stack[sp++] = c;
            }
        }
    }
}

__global__ void Kernel_QueryRange(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const cuAABB* ranges, unsigned int queryCount,
    unsigned int* outIndices /*[queryCount * maxOutPerQuery]*/,
    unsigned int* outCounts  /*[queryCount]*/,
    unsigned int maxOutPerQuery)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;

    unsigned int* myOut = outIndices + tid * maxOutPerQuery;
    unsigned int count = 0;
    QueryRange(nodes, nodeCount, ranges[tid], myOut, maxOutPerQuery, count);

    if (count > maxOutPerQuery)
        count = maxOutPerQuery;

    outCounts[tid] = count;
}

__device__ void QueryRadius(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3& center, float radius,
    unsigned int* outIndices, unsigned int maxOut, unsigned int& outCount)
{
    float r2 = radius * radius;
    unsigned int stack[STACK_MAX];
    unsigned int sp = 0;
    stack[sp++] = 0; outCount = 0;

    while (sp > 0) {
        unsigned int idx = stack[--sp];
        if (idx < 0 || idx >= nodeCount) continue;
        const auto& node = nodes[idx];

        float d2 = cuAABB::Distance2(node.bounds, center);
        if (d2 > r2) continue;

        if (outCount < maxOut) outIndices[outCount] = idx;
        ++outCount;

        if (!node.isLeaf) {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                unsigned int c = node.children[i];
                if (c != -1 && sp < STACK_MAX) stack[sp++] = c;
            }
        }
    }
}

__global__ void Kernel_QueryRadius(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* centers, const float* radii, unsigned int queryCount,
    unsigned int* outIndices, unsigned int* outCounts, unsigned int maxOutPerQuery)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;

    unsigned int* myOut = outIndices + tid * maxOutPerQuery;
    unsigned int count = 0;
    QueryRadius(nodes, nodeCount, centers[tid], radii[tid], myOut, maxOutPerQuery, count);
    outCounts[tid] = count;
}






__device__ unsigned int QueryNearestPointDevice(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* points, unsigned int pointCount,
    const unsigned int* flatIndices,
    const float3& p)
{
    unsigned int bestPoint = UINT32_MAX;
    float bestD2 = FLT_MAX;

    unsigned int stack[STACK_MAX];
    unsigned int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        unsigned int idx = stack[--sp];
        if (idx >= nodeCount) continue;
        const auto& node = nodes[idx];

        // AABB까지의 최소거리로 가지치기
        float d2 = cuAABB::Distance2(node.bounds, p);
        if (d2 >= bestD2) continue;

        if (node.isLeaf) {
            // 실제 점들과 거리 계산
            for (unsigned int i = 0; i < node.indexCount; ++i) {
                unsigned int pi = flatIndices[node.indexOffset + i];
                if (pi >= pointCount) continue;
                float3 q = points[pi];
                float3 diff = make_float3(q.x - p.x, q.y - p.y, q.z - p.z);
                float d2p = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if (d2p < bestD2) {
                    bestD2 = d2p;
                    bestPoint = pi;
                }
            }
        }
        else {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                unsigned int c = node.children[i];
                if (c != UINT32_MAX && sp < STACK_MAX)
                    stack[sp++] = c;
            }
        }
    }

    return bestPoint;
}

__global__ void Kernel_QueryNearestPoint(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* points, unsigned int pointCount,
    const unsigned int* flatIndices,
    const float3* queryPoints, unsigned int queryCount,
    unsigned int* outNearest)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;
    outNearest[tid] = QueryNearestPointDevice(
        nodes, nodeCount, points, pointCount,
        flatIndices, queryPoints[tid]);
}










template<unsigned int K>
struct SmallKList {
    // 거리 오름차순 유지
    unsigned int   idx[K];
    float d2[K];
    unsigned int n;
    __device__ void init() { n = 0; }
    __device__ void try_push(unsigned int id, float dist2) {
        // 이미 K개면, 마지막(최대)보다 작은 경우만 삽입
        if (n == K) {
            if (dist2 >= d2[n - 1]) return;
            // 한 칸 비우기 위해 마지막 버림
            --n;
        }
        // 뒤에서부터 삽입 위치 탐색 (오름차순)
        unsigned int pos = n;
        while (pos > 0 && d2[pos - 1] > dist2) {
            if (pos < K) { idx[pos] = idx[pos - 1]; d2[pos] = d2[pos - 1]; }
            --pos;
        }
        if (pos < K) { idx[pos] = id; d2[pos] = dist2; }
        if (n < K) ++n;
    }
};

template<unsigned int K>
__device__ unsigned int QueryKNNNodeDevice(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3& p,
    unsigned int* outIdx /*size K*/)
{
    SmallKList<K> heap;
    heap.init();

    unsigned int stack[STACK_MAX]; unsigned int sp = 0; stack[sp++] = 0;

    while (sp > 0) {
        unsigned int idx = stack[--sp];
        if (idx < 0 || idx >= nodeCount) continue;
        const auto& node = nodes[idx];

        float d2 = cuAABB::Distance2(node.bounds, p);

        // 현재 리스트가 가득 찼고, 이 서브트리 AABB가 더 멀면 가지치기
        if (heap.n == K && d2 >= heap.d2[heap.n - 1]) continue;

        if (node.isLeaf) {
            heap.try_push(idx, d2);
        }
        else {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                unsigned int c = node.children[i];
                if (c != -1 && sp < STACK_MAX) stack[sp++] = c;
            }
        }
    }

    // 결과 복사
    unsigned int ret = heap.n; // 실제 채워진 개수
    for (unsigned int i = 0; i < K; ++i) outIdx[i] = (i < heap.n) ? heap.idx[i] : -1;
    return ret;
}

// K는 템플릿로 고정: 빌드시 자주 쓰는 값에 대해 인스턴스화 추천(예: 4,8,16)
template<unsigned int K>
__global__ void Kernel_QueryKNNNode(
    const HybridDeviceOctreeNode* nodes, unsigned int nodeCount,
    const float3* queryPoints, unsigned int queryCount,
    unsigned int* outKnn /*[queryCount*K]*/, unsigned int* outCounts /*[queryCount]*/)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= queryCount) return;
    unsigned int* myOut = outKnn + tid * K;
    unsigned int n = QueryKNNNodeDevice<K>(nodes, nodeCount, queryPoints[tid], myOut);
    outCounts[tid] = n;
}

//// 필요 K에 대해 커널 인스턴스 선언
//template __global__ void Kernel_QueryKNNNode<8>(
//    const HybridDeviceOctreeNode*, int, const float3*, int, int*, int*);

void HybridOctree::QueryKNNNode_K8(
    const std::vector<float3>& querypoints,
    std::vector<unsigned int>& outFlatK,  // size = queryCount * 8
    std::vector<unsigned int>& outCounts) // size = queryCount
{
    const unsigned int K = 8;
    outFlatK.clear(); outCounts.clear();
    if (!device_nodes || querypoints.empty()) return;

    unsigned int q = (unsigned int)querypoints.size();
    float3* d_points = nullptr; unsigned int* d_out = nullptr; unsigned int* d_counts = nullptr;

    CUDA_MALLOC(&d_points, sizeof(float3) * q);
    CUDA_MALLOC(&d_out, sizeof(unsigned int) * q * K);
    CUDA_MALLOC(&d_counts, sizeof(unsigned int) * q);

    CUDA_COPY_H2D(d_points, querypoints.data(), sizeof(float3) * q);

    LaunchKernel(Kernel_QueryKNNNode<8>, q,
        device_nodes, (unsigned int)host_nodes.size(), d_points, q, d_out, d_counts);

    outFlatK.resize((size_t)q * K);
    outCounts.resize(q);
    CUDA_COPY_D2H(outFlatK.data(), d_out, sizeof(unsigned int) * q * K);
    CUDA_COPY_D2H(outCounts.data(), d_counts, sizeof(unsigned int) * q);
    CUDA_SYNC();

    CUDA_SAFE_FREE(d_points);
    CUDA_SAFE_FREE(d_out);
    CUDA_SAFE_FREE(d_counts);
}
