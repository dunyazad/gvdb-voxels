#pragma warning(disable : 4819)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "main.cuh"
#include <libFeather.h>

#include "HPOctree.cuh"

#include <Eigen/Sparse>
using SparseMatrixf = Eigen::SparseMatrix<float>;
using VecXf = Eigen::VectorXf;

using VD = VisualDebugging;

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001

bool ForceGPUPerformance()
{
	NvAPI_Status status;

	status = NvAPI_Initialize();
	if (status != NVAPI_OK)
	{
		return false;
	}

	NvDRSSessionHandle hSession = 0;
	status = NvAPI_DRS_CreateSession(&hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	// (2) load all the system settings into the session
	status = NvAPI_DRS_LoadSettings(hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	NvDRSProfileHandle hProfile = 0;
	status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
	if (status != NVAPI_OK)
	{
		return false;
	}

	NVDRS_SETTING drsGet = { 0, };
	drsGet.version = NVDRS_SETTING_VER;
	status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
	if (status != NVAPI_OK)
	{
		return false;
	}
	auto m_gpu_performance = drsGet.u32CurrentValue;

	NVDRS_SETTING drsSetting = { 0, };
	drsSetting.version = NVDRS_SETTING_VER;
	drsSetting.settingId = PREFERRED_PSTATE_ID;
	drsSetting.settingType = NVDRS_DWORD_TYPE;
	drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;

	status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
	if (status != NVAPI_OK)
	{
		return false;
	}

	status = NvAPI_DRS_SaveSettings(hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	// (6) We clean up. This is analogous to doing a free()
	NvAPI_DRS_DestroySession(hSession);
	hSession = 0;

	return true;
}

CUDAInstance cuInstance;

std::vector<glm::vec4> inpterpolatedColors;

struct H_OctreeNode {
	cuAABB bounds;
	std::vector<unsigned int> indices;   // float3 대신 index
	std::unique_ptr<H_OctreeNode> children[8];
	bool isLeaf = true;

	float3 V = make_float3(0, 0, 0);  // 벡터장 값
	float divergence = 0.0f;
	int index = 0;
	float chi;
};

float3 ComputeVectorField(H_OctreeNode& node,
	const std::vector<float3>& points,
	const std::vector<float3>& normals)
{
	// 리프 노드
	if (node.isLeaf)
	{
		if (node.indices.empty())
		{
			node.V = make_float3(0, 0, 0);
			return node.V;
		}

		float3 avg = make_float3(0, 0, 0);
		for (unsigned int idx : node.indices)
		{
			const auto& n = normals[idx];
			avg.x += n.x;
			avg.y += n.y;
			avg.z += n.z;
		}

		float inv = 1.0f / static_cast<float>(node.indices.size());
		avg.x *= inv;
		avg.y *= inv;
		avg.z *= inv;

		// 단위화(normalize)
		float len = sqrtf(avg.x * avg.x + avg.y * avg.y + avg.z * avg.z);
		if (len > 1e-6f)
		{
			avg.x /= len;
			avg.y /= len;
			avg.z /= len;
		}

		node.V = avg;
		return node.V;
	}

	// 비리프 노드 → 자식들의 V 평균
	float3 sum = make_float3(0, 0, 0);
	int childCount = 0;

	for (int i = 0; i < 8; ++i)
	{
		if (node.children[i])
		{
			float3 childV = ComputeVectorField(*node.children[i], points, normals);
			sum.x += childV.x;
			sum.y += childV.y;
			sum.z += childV.z;
			childCount++;
		}
	}

	if (childCount > 0)
	{
		sum.x /= childCount;
		sum.y /= childCount;
		sum.z /= childCount;
	}

	node.V = sum;
	return node.V;
}

void VisualizeVectorField(const H_OctreeNode& node,
	float scale = 0.05f,
	int depth = 0)
{
	// 1. 현재 노드 중심 계산
	float3 c = (node.bounds.min + node.bounds.max) * 0.5f;

	// 2. 방향 벡터 (스케일 조정)
	float3 v = node.V;
	float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	if (len > 1e-6f)
	{
		v.x = v.x / len * scale;
		v.y = v.y / len * scale;
		v.z = v.z / len * scale;
	}

	float3 e = make_float3(c.x + v.x, c.y + v.y, c.z + v.z);

	// 3. 색상 선택 (깊이에 따라 변화)
	glm::vec4 color = inpterpolatedColors[depth % inpterpolatedColors.size()];

	// 4. 선분으로 벡터 표시
	if(node.isLeaf)
	VD::AddLine("V", glm::vec3(XYZ(c)), glm::vec3(XYZ(e)), color);

	// (선택) 박스도 같이 표시해도 좋습니다.
	//VD::AddWiredBox("Box_" + std::to_string(depth),
	//    { glm::vec3(XYZ(node.bounds.min)), glm::vec3(XYZ(node.bounds.max)) },
	//    color
	//);

	// 5. 재귀적으로 자식 노드 처리
	if (!node.isLeaf)
	{
		for (int i = 0; i < 8; ++i)
		{
			if (node.children[i])
				VisualizeVectorField(*node.children[i], scale, depth + 1);
		}
	}
}

void subdivide(H_OctreeNode& node,
	const std::vector<float3>& points,
	int maxPoints, int maxDepth, int depth = 0)
{
	if (node.indices.size() <= maxPoints || depth >= maxDepth)
		return;

	const float3 c = (node.bounds.max + node.bounds.min) * 0.5f;
	const float3& mn = node.bounds.min;
	const float3& mx = node.bounds.max;

	// 8개 자식 인덱스 버킷
	std::vector<unsigned int> childIdx[8];

	// 반열림 규칙: (low: < c) (high: >= c)
	for (unsigned int idx : node.indices) {
		const float3& p = points[idx];

		int ox = (p.x >= c.x) ? 1 : 0;
		int oy = (p.y >= c.y) ? 1 : 0;
		int oz = (p.z >= c.z) ? 1 : 0;
		int oct = (ox) | (oy << 1) | (oz << 2);

		childIdx[oct].push_back(idx);
	}

	node.indices.clear();
	node.isLeaf = false;

	// 자식 AABB 생성 및 재귀
	for (int i = 0; i < 8; ++i) {
		if (childIdx[i].empty()) continue;

		int ox = i & 1;
		int oy = (i >> 1) & 1;
		int oz = (i >> 2) & 1;

		float3 childMin = make_float3(ox ? c.x : mn.x,
			oy ? c.y : mn.y,
			oz ? c.z : mn.z);

		float3 childMax = make_float3(ox ? mx.x : c.x,
			oy ? mx.y : c.y,
			oz ? mx.z : c.z);

		// (선택) 너무 작은 셀로 더 쪼개지지 않게 epsilon 체크
		const float eps = 1e-7f;
		if ((childMax.x - childMin.x) < eps ||
			(childMax.y - childMin.y) < eps ||
			(childMax.z - childMin.z) < eps) {
			continue; // 더 분할하지 않음
		}

		node.children[i] = std::make_unique<H_OctreeNode>();
		node.children[i]->bounds = { childMin, childMax };
		node.children[i]->indices = std::move(childIdx[i]);

		subdivide(*node.children[i], points, maxPoints, maxDepth, depth + 1);
	}
}


H_OctreeNode buildOctree(const std::vector<float3>& points,
	const cuAABB& aabb,
	int maxPoints = 10, int maxDepth = 8)
{
	H_OctreeNode root;
	root.bounds = aabb;

	root.indices.resize(points.size());
	std::iota(root.indices.begin(), root.indices.end(), 0); // 0,1,2,3,...

	subdivide(root, points, maxPoints, maxDepth);
	return root;
}

void drawOctreeNode(const H_OctreeNode& node,
	const std::vector<float3>& points,
	const std::string& name,
	int depth = 0,
	bool drawPoints = false)
{
	auto color = inpterpolatedColors[depth];

	VD::AddWiredBox(
		name + std::to_string(depth),
		{ glm::vec3(XYZ(node.bounds.min)), glm::vec3(XYZ(node.bounds.max)) },
		color
	);

	if (!node.isLeaf) {
		for (int i = 0; i < 8; ++i) {
			if (node.children[i])
				drawOctreeNode(*node.children[i], points, name, depth + 1);
		}
	}
}

void QueryAABB(const H_OctreeNode& node,
	const cuAABB& queryAABB,
	const std::vector<float3>& points,
	std::vector<unsigned int>& results)
{
	if (!node.bounds.intersects(queryAABB))
		return;

	if (node.isLeaf)
	{
		for (unsigned int idx : node.indices)
		{
			const float3& p = points[idx];
			if (queryAABB.Contains(p))
				results.push_back(idx);
		}
		return;
	}

	for (int i = 0; i < 8; ++i)
	{
		if (node.children[i])
			QueryAABB(*node.children[i], queryAABB, points, results);
	}
}

void QuerySphere(const H_OctreeNode& node,
	const float3& center,
	float radius,
	const std::vector<float3>& points,
	std::vector<unsigned int>& results)
{
	float3 boxCenter = (node.bounds.min + node.bounds.max) * 0.5f;
	float3 boxHalf = (node.bounds.max - node.bounds.min) * 0.5f;

	float3 diff = make_float3(fabs(center.x - boxCenter.x),
		fabs(center.y - boxCenter.y),
		fabs(center.z - boxCenter.z));
	float3 clamped = make_float3(
		fmaxf(diff.x - boxHalf.x, 0.0f),
		fmaxf(diff.y - boxHalf.y, 0.0f),
		fmaxf(diff.z - boxHalf.z, 0.0f)
	);
	float dist2 = clamped.x * clamped.x + clamped.y * clamped.y + clamped.z * clamped.z;
	if (dist2 > radius * radius)
		return;

	if (node.isLeaf)
	{
		float r2 = radius * radius;
		for (unsigned int idx : node.indices)
		{
			const float3& p = points[idx];
			float dx = p.x - center.x;
			float dy = p.y - center.y;
			float dz = p.z - center.z;
			if (dx * dx + dy * dy + dz * dz <= r2)
				results.push_back(idx);
		}
		return;
	}

	for (int i = 0; i < 8; ++i)
	{
		if (node.children[i])
			QuerySphere(*node.children[i], center, radius, points, results);
	}
}

void QueryClosestPoint(const H_OctreeNode& node,
	const float3& query,
	const std::vector<float3>& points,
	unsigned int& bestIdx,
	float& bestDist2)
{
	float3 p = make_float3(
		fmaxf(node.bounds.min.x, fminf(query.x, node.bounds.max.x)),
		fmaxf(node.bounds.min.y, fminf(query.y, node.bounds.max.y)),
		fmaxf(node.bounds.min.z, fminf(query.z, node.bounds.max.z))
	);
	float dx = query.x - p.x;
	float dy = query.y - p.y;
	float dz = query.z - p.z;
	float dist2 = dx * dx + dy * dy + dz * dz;

	if (dist2 > bestDist2)
		return;

	if (node.isLeaf)
	{
		for (unsigned int idx : node.indices)
		{
			const float3& pt = points[idx];
			float ddx = query.x - pt.x;
			float ddy = query.y - pt.y;
			float ddz = query.z - pt.z;
			float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
			if (d2 < bestDist2)
			{
				bestDist2 = d2;
				bestIdx = idx;
			}
		}
		return;
	}

	for (int i = 0; i < 8; ++i)
		if (node.children[i])
			QueryClosestPoint(*node.children[i], query, points, bestIdx, bestDist2);
}

inline float DistanceSq(const float3& a, const float3& b)
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	float dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

inline float DistanceSqToAABB(const float3& p, const cuAABB& box)
{
	float dx = std::max({ box.min.x - p.x, 0.0f, p.x - box.max.x });
	float dy = std::max({ box.min.y - p.y, 0.0f, p.y - box.max.y });
	float dz = std::max({ box.min.z - p.z, 0.0f, p.z - box.max.z });
	return dx * dx + dy * dy + dz * dz;
}

struct KNNResult {
	unsigned int index;
	float dist2;
};

// 거리 큰 순으로 정렬되는 최대 힙 comparator
struct CompareDist {
	bool operator()(const KNNResult& a, const KNNResult& b) const {
		return a.dist2 < b.dist2; // 큰 값이 top
	}
};

void QueryKNN(const H_OctreeNode& node,
	const std::vector<float3>& points,
	const float3& query,
	int k,
	std::priority_queue<KNNResult, std::vector<KNNResult>, CompareDist>& knnHeap)
{
	// AABB-Query 거리 기반 가지치기 (pruning)
	float distBox2 = DistanceSqToAABB(query, node.bounds);
	if (knnHeap.size() == k && distBox2 > knnHeap.top().dist2)
		return; // 현재 노드 전체가 너무 멈

	if (node.isLeaf)
	{
		for (unsigned int idx : node.indices)
		{
			const float3& p = points[idx];
			float d2 = DistanceSq(p, query);

			if (knnHeap.size() < k)
			{
				knnHeap.push({ idx, d2 });
			}
			else if (d2 < knnHeap.top().dist2)
			{
				knnHeap.pop();
				knnHeap.push({ idx, d2 });
			}
		}
		return;
	}

	// 비리프 노드: 자식 노드들을 거리순으로 탐색 (효율적)
	std::vector<std::pair<float, const H_OctreeNode*>> childList;
	childList.reserve(8);
	for (int i = 0; i < 8; ++i)
	{
		if (node.children[i])
		{
			float d2 = DistanceSqToAABB(query, node.children[i]->bounds);
			childList.push_back({ d2, node.children[i].get() });
		}
	}

	// 가까운 노드부터 방문
	std::sort(childList.begin(), childList.end(),
		[](auto& a, auto& b) { return a.first < b.first; });

	for (auto& [dist, child] : childList)
	{
		if (knnHeap.size() == k && dist > knnHeap.top().dist2)
			break;
		QueryKNN(*child, points, query, k, knnHeap);
	}
}

std::vector<std::pair<unsigned int, float>> FindKNN(
	const H_OctreeNode& root,
	const std::vector<float3>& points,
	const float3& query,
	int k)
{
	std::priority_queue<KNNResult, std::vector<KNNResult>, CompareDist> knnHeap;
	QueryKNN(root, points, query, k, knnHeap);

	std::vector<std::pair<unsigned int, float>> results;
	results.reserve(knnHeap.size());
	while (!knnHeap.empty())
	{
		results.emplace_back(knnHeap.top().index, knnHeap.top().dist2);
		knnHeap.pop();
	}

	// 가까운 순서로 정렬
	std::reverse(results.begin(), results.end());
	return results;
}








void ComputeDivergence_ChildrenBased(H_OctreeNode& node)
{
	if (node.isLeaf)
	{
		node.divergence = node.V.x + node.V.y + node.V.z;
		return;
	}

	float3 center = (node.bounds.min + node.bounds.max) * 0.5f;
	float3 half = (node.bounds.max - node.bounds.min) * 0.5f;

	float3 grad = make_float3(0, 0, 0);
	int count = 0;

	for (int i = 0; i < 8; ++i)
	{
		if (!node.children[i]) continue;
		float3 childCenter = (node.children[i]->bounds.min + node.children[i]->bounds.max) * 0.5f;
		float3 d = childCenter - center;
		float3 dv = node.children[i]->V - node.V;

		if (fabsf(d.x) > 1e-6f) grad.x += dv.x / d.x;
		if (fabsf(d.y) > 1e-6f) grad.y += dv.y / d.y;
		if (fabsf(d.z) > 1e-6f) grad.z += dv.z / d.z;

		count++;
	}

	if (count > 0)
		node.divergence = (grad.x + grad.y + grad.z) / (float)count;

	for (int i = 0; i < 8; ++i)
		if (node.children[i])
			ComputeDivergence_ChildrenBased(*node.children[i]);
}

void VisualizeDivergence(const H_OctreeNode& node)
{
	float3 c = (node.bounds.min + node.bounds.max) * 0.5f;
	float val = node.divergence;

	// 파랑(음수) ~ 흰색(0) ~ 빨강(양수)
	float t = 0.5f + 0.5f * tanh(val * 0.1f);
	glm::vec4 color = glm::mix(Color::blue(), Color::red(), t);

	if (node.isLeaf)
	VD::AddSphere("div", glm::vec3(XYZ(c)), 0.03f, color);

	if (!node.isLeaf)
		for (int i = 0; i < 8; ++i)
			if (node.children[i])
				VisualizeDivergence(*node.children[i]);
}

void ApproximateSDF(H_OctreeNode& node,
	const std::vector<float3>& points,
	const std::vector<float3>& normals)
{
	if (node.isLeaf)
	{
		if (node.indices.empty())
		{
			node.chi = 0.0f;
			return;
		}

		// leaf 중심점
		float3 center = (node.bounds.min + node.bounds.max) * 0.5f;
		float3 avgNormal = make_float3(0, 0, 0);
		float distSum = 0.0f;
		int count = 0;

		for (auto idx : node.indices)
		{
			const float3& p = points[idx];
			const float3& n = normals[idx];

			float3 d = center - p;
			float dist = d.x * n.x + d.y * n.y + d.z * n.z; // dot(center - p, n)
			distSum += dist;
			count++;
		}

		node.chi = (count > 0) ? (distSum / count) : 0.0f;
		return;
	}

	// 비리프 → 자식 평균
	float sum = 0.0f;
	int childCount = 0;
	for (int i = 0; i < 8; ++i)
	{
		if (node.children[i])
		{
			ApproximateSDF(*node.children[i], points, normals);
			sum += node.children[i]->chi;
			childCount++;
		}
	}

	if (childCount > 0)
		node.chi = sum / childCount;
	else
		node.chi = 0.0f;
}

void SmoothChi(H_OctreeNode& node)
{
	if (node.isLeaf) return;

	float sum = 0;
	int count = 0;
	for (int i = 0; i < 8; ++i)
		if (node.children[i]) { sum += node.children[i]->chi; count++; }

	if (count > 0)
		node.chi = 0.5f * node.chi + 0.5f * (sum / count); // 간단한 평균 smoothing

	for (int i = 0; i < 8; ++i)
		if (node.children[i]) SmoothChi(*node.children[i]);
}

void FindMinMaxChi(const H_OctreeNode& node, float& minChi, float& maxChi)
{
	minChi = std::min(minChi, node.chi);
	maxChi = std::max(maxChi, node.chi);

	if (!node.isLeaf)
	{
		for (int i = 0; i < 8; ++i)
		{
			if (node.children[i])
				FindMinMaxChi(*node.children[i], minChi, maxChi);
		}
	}
}

AABB ApplyPointCloudToEntity(Entity entity, const HostPointCloud<PointCloudProperty>& h_pointCloud)
{
	Feather.CreateEventCallback<KeyEvent>(entity, [](Entity entity, const KeyEvent& event) {
		auto renderable = Feather.GetComponent<Renderable>(entity);
		if (nullptr == renderable) return;

		if (0 == event.action)
		{
			if (GLFW_KEY_GRAVE_ACCENT == event.keyCode)
			{
				renderable->NextDrawingMode();
			}
			else if (GLFW_KEY_1 == event.keyCode)
			{
				renderable->SetActiveShaderIndex(0);
			}
			else if (GLFW_KEY_2 == event.keyCode)
			{
				renderable->SetActiveShaderIndex(1);
			}
		}
		});

	auto renderable = Feather.GetComponent<Renderable>(entity);
	if (nullptr == renderable)
	{
		renderable = Feather.CreateComponent<Renderable>(entity);

		renderable->Initialize(Renderable::GeometryMode::Triangles);
		renderable->AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
		renderable->AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
		renderable->SetActiveShaderIndex(1);
	}
	else
	{
		renderable->Clear();
	}

	auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere({ 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6);
	//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
	renderable->AddIndices(indices);
	renderable->AddVertices(vertices);
	renderable->AddNormals(normals);
	renderable->AddColors(colors);
	renderable->AddUVs(uvs);

	AABB aabb{ {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX} };

	auto cs = Color::GetContrastingColors(32);
	std::map<unsigned int, unsigned int> colorMap;

	for (size_t i = 0; i < h_pointCloud.numberOfPoints; i++)
	{
		auto& p = h_pointCloud.positions[i];
		if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
		auto& n = h_pointCloud.normals[i];
		auto& c = h_pointCloud.colors[i];
		auto l = h_pointCloud.properties[i].label;

		//if (0 == l) continue;

		if (colorMap.end() == colorMap.find(l))
		{
			colorMap[l] = colorMap.size() % 32;
		}

		auto mc = cs[colorMap[l]];

		glm::vec3 position(XYZ(p));
		glm::vec3 normal(XYZ(n));
		glm::vec4 color(XYZ(c), 1.0f);

		aabb.min.x = std::min(aabb.min.x, position.x);
		aabb.min.y = std::min(aabb.min.y, position.y);
		aabb.min.z = std::min(aabb.min.z, position.z);

		aabb.max.x = std::max(aabb.max.x, position.x);
		aabb.max.y = std::max(aabb.max.y, position.y);
		aabb.max.z = std::max(aabb.max.z, position.z);

		renderable->AddInstanceColor(color);
		//renderable->AddInstanceColor(mc);
		renderable->AddInstanceNormal(normal);

		glm::mat4 tm = glm::identity<glm::mat4>();
		glm::mat4 rot = glm::mat4(1.0f);
		if (glm::length(normal) > 0.0001f)
		{
			glm::vec3 axis = glm::normalize(glm::cross(glm::vec3(0, 0, 1), normal));
			float angle = acos(glm::dot(glm::normalize(normal), glm::vec3(0, 0, 1)));
			if (glm::length(axis) > 0.0001f)
				rot = glm::rotate(glm::mat4(1.0f), angle, axis);
		}
		tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
		renderable->AddInstanceTransform(tm);
		renderable->IncreaseNumberOfInstances();
	}

	return aabb;
}

AABB GetDomainAABB(const AABB& aabb)
{
	auto delta = aabb.max - aabb.min;
	float hl = std::max({ delta.x, delta.y, delta.z });
	auto center = (aabb.min + aabb.max) * 0.5f;
	float half_length = hl * 0.5f;
	return {
		{center.x - half_length, center.y - half_length, center.z - half_length},
		{center.x + half_length, center.y + half_length, center.z + half_length}
	};
}

std::tuple<AABB, std::vector<AABB>> SplitAABB(const AABB& input_aabb, unsigned int cellsPerAxis)
{
	AABB totalAABB{ {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX} };

	std::vector<AABB> subAABBs;
	if (cellsPerAxis == 0) return { totalAABB, subAABBs };

	auto aabb = GetDomainAABB(input_aabb);
	auto domainSize = aabb.max - aabb.min;
	auto cellSize = domainSize / static_cast<float>(cellsPerAxis);

	domainSize += cellSize;
	cellSize = domainSize / static_cast<float>(cellsPerAxis);

	for (unsigned int z = 0; z < cellsPerAxis; ++z)
	{
		for (unsigned int y = 0; y < cellsPerAxis; ++y)
		{
			for (unsigned int x = 0; x < cellsPerAxis; ++x)
			{
				glm::vec3 cellMin =
				{
					aabb.min.x + x * cellSize.x - cellSize.x * 0.5f,
					aabb.min.y + y * cellSize.y - cellSize.y * 0.5f,
					aabb.min.z + z * cellSize.z - cellSize.z * 0.5f
				};
				glm::vec3 cellMax = cellMin + cellSize;
				AABB subAABB = { cellMin, cellMax };

				if (input_aabb.Intersects(subAABB))
				{
					totalAABB.min.x = std::min(totalAABB.min.x, cellMin.x);
					totalAABB.min.y = std::min(totalAABB.min.y, cellMin.y);
					totalAABB.min.z = std::min(totalAABB.min.z, cellMin.z);

					totalAABB.max.x = std::max(totalAABB.max.x, cellMax.x);
					totalAABB.max.y = std::max(totalAABB.max.y, cellMax.y);
					totalAABB.max.z = std::max(totalAABB.max.z, cellMax.z);

					subAABBs.push_back({ cellMin, cellMax });
				}
			}
		}
	}

	return { totalAABB, subAABBs };
}

void SortByAABBs(HostPointCloud<PointCloudProperty>& pointCloud, const AABB& totalAABB, const std::vector<AABB>& subAABBs)
{
	if (pointCloud.numberOfPoints == 0 || subAABBs.empty())
	{
		return;
	}

	std::vector<std::vector<unsigned int>> bins(subAABBs.size());
	for (unsigned int i = 0; i < pointCloud.numberOfPoints; ++i)
	{
		const auto& p = pointCloud.positions[i];
		for (size_t j = 0; j < subAABBs.size(); ++j)
		{
			if (subAABBs[j].Contains({ XYZ(p) }))
			{
				bins[j].push_back(i);
				break;
			}
		}
	}

	HostPointCloud<PointCloudProperty> sortedPointCloud;
	sortedPointCloud.Initialize(pointCloud.numberOfPoints);

	unsigned int writeIndex = 0;

	auto colors = Color::GetContrastingColors(bins.size());
	int colorIndex = 0;
	for (const auto& bin : bins)
	{
		auto color = colors[colorIndex++];

		for (const auto originalIndex : bin)
		{
			sortedPointCloud.positions[writeIndex] = pointCloud.positions[originalIndex];
			sortedPointCloud.normals[writeIndex] = pointCloud.normals[originalIndex];
			sortedPointCloud.colors[writeIndex] = pointCloud.colors[originalIndex];

			if constexpr (!std::is_void_v<PointCloudProperty>)
			{
				sortedPointCloud.properties[writeIndex] = pointCloud.properties[originalIndex];
			}

			//VD::AddWiredBox("sorted", { XYZ(pointCloud.positions[originalIndex]) }, { 0.05f, 0.05f, 0.05f }, color);

			writeIndex++;
		}
	}

	pointCloud = std::move(sortedPointCloud);
}

int main(int argc, char** argv)
{
	ForceGPUPerformance();
	CUDA_FREE(0);

	std::cout << "AppFeather" << std::endl;

	Feather.Initialize(1920, 1080);
	Feather.SetClearColor(Color::slategray());

	auto w = Feather.GetFeatherWindow();

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&](Entity entity, const KeyEvent& event) {
			//printf("KeyEvent : keyCode=%d, scanCode=%d, action=%d, mods=%d\n", event.keyCode, event.scanCode, event.action, event.mods);

			if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			else if (GLFW_KEY_SPACE == event.keyCode)
			{
				if (event.action == 1)
				{
					Feather.GetImmediateModeRenderSystem()->ToggleEnable();
				}
			}
			else if (GLFW_KEY_BACKSPACE == event.keyCode)
			{
				VD::ClearAll();
			}
			else if (GLFW_KEY_UP == event.keyCode)
			{
				if (event.action == 1)
				{
					auto index = VD::ShowNextSelection();
					printf("index : %d\n", index);
				}
			}
			else if (GLFW_KEY_DOWN == event.keyCode)
			{
				if (event.action == 1)
				{
					auto index = VD::ShowPreviousSelection();
					printf("index : %d\n", index);
				}
			}
			/*
			else if(GLFW_KEY_GRAVE_ACCENT == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_0");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							auto current = renderable->GetActiveShaderIndex();
							auto next = (current + 1) % renderable->GetShaders().size();
							renderable->SetActiveShaderIndex(next);
							printf("Shader index : %d\n", next);
						}
					}
				}
			}
			else if (GLFW_KEY_1 == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_0");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							renderable->ToggleVisible();
						}
					}
				}
			}
			else if (GLFW_KEY_2 == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_1");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							renderable->ToggleVisible();
						}
					}
				}
			}
			*/
			});
	}
#pragma endregion

#pragma region Camera
	{
		Entity cam = Feather.CreateEntity("Camera");
		auto pcam = Feather.CreateComponent<PerspectiveCamera>(cam);
		auto pcamMan = Feather.CreateComponent<CameraManipulatorTrackball>(cam);
		pcamMan->SetCamera(pcam);

		Feather.CreateEventCallback<FrameBufferResizeEvent>(cam, [pcam](Entity entity, const FrameBufferResizeEvent& event) {
			auto window = Feather.GetFeatherWindow();
			auto aspectRatio = (f32)window->GetWidth() / (f32)window->GetHeight();
			pcam->SetAspectRatio(aspectRatio);
			});

		Feather.CreateEventCallback<KeyEvent>(cam, [](Entity entity, const KeyEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnKey(event);
			});

		Feather.CreateEventCallback<MousePositionEvent>(cam, [](Entity entity, const MousePositionEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMousePosition(event);
			});

		Feather.CreateEventCallback<MouseButtonEvent>(cam, [&](Entity entity, const MouseButtonEvent& event) {
			auto manipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);
			manipulator->OnMouseButton(event);
			auto camera = manipulator->GetCamera();

			auto renderable = Feather.GetComponent<Renderable>(entity);
			if (event.button == 0 && event.action == 0)
			{
			}
			});

		Feather.CreateEventCallback<MouseWheelEvent>(cam, [](Entity entity, const MouseWheelEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMouseWheel(event);
			});
	}
#pragma endregion

#pragma region Panels
	{
		auto gui = Feather.CreateEntity("Status Panel");

		auto statusPanel = Feather.CreateComponent<StatusPanel>(gui);
		Feather.CreateEventCallback<MousePositionEvent>(gui, [](Entity entity, const MousePositionEvent& event) {
			auto component = Feather.GetComponent<StatusPanel>(entity);
			component->mouseX = event.xpos;
			component->mouseY = event.ypos;
			});
	}
	{
		auto entity = Feather.CreateEntity("Control Panel");
		auto controlPanel = Feather.CreateComponent<ControlPanel>(entity, "Control Panel");

		auto Temp = [&]() {
			};
	
		//controlPanel->AddButton("Temp", 0, 0, Temp);
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {

		unsigned int OCTREE_DEPTH = 16;

		//inpterpolatedColors = Color::InterpolateColors({ Color::blue(), Color::yellow(), Color::black(), Color::green(), Color::red() }, 16);
		inpterpolatedColors = Color::InterpolateColors({ Color::blue(), Color::red() }, OCTREE_DEPTH);

		//cuInstance.Test();

			HostPointCloud<PointCloudProperty> h_input;
			h_input.DeserializePLY("D:\\Debug\\PLY\\input.ply");

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);
			
				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			auto entity = Feather.CreateEntity("PointCloud");
			auto aabb = ApplyPointCloudToEntity(entity, h_input);

			auto center = (aabb.min + aabb.max) * 0.5f;
			auto axisLength = aabb.max - aabb.min;
			auto maxLength = fmaxf(fmaxf(axisLength.x, axisLength.y), axisLength.z);
			auto halfLength = maxLength * 0.5f;

			aabb.min.x = center.x - halfLength;
			aabb.min.y = center.y - halfLength;
			aabb.min.z = center.z - halfLength;

			aabb.max.x = center.x + halfLength;
			aabb.max.y = center.y + halfLength;
			aabb.max.z = center.z + halfLength;

			{
				for (size_t i = 0; i <= OCTREE_DEPTH; i++)
				{
					std::string name = "octree_" + std::to_string(i);
					VD::AddToSelectionList(name);
				}

				TS(BuildOCtree_CPU);

				std::vector<float3> inputPositions(h_input.numberOfPoints);
				std::vector<float3> inputNormals(h_input.numberOfPoints);
				for (size_t i = 0; i < h_input.numberOfPoints; i++)
				{
					inputPositions[i] = h_input.positions[i];
					inputNormals[i] = h_input.normals[i];
				}
				auto root = buildOctree(inputPositions, { {XYZ(aabb.min)}, {XYZ(aabb.max)}}, 1, OCTREE_DEPTH);
				//drawOctreeNode(node, inputPositions, "octree_");

				ComputeVectorField(root, inputPositions, inputNormals);
				//VisualizeVectorField(root);

				ComputeDivergence_ChildrenBased(root);
				//VisualizeDivergence(root);

				TE(BuildOCtree_CPU);

				//{ // VisualizingChi
				//	float minChi = 1e9f, maxChi = -1e9f;
				//	std::function<void(const H_OctreeNode&)> FindMinMax = [&](const H_OctreeNode& n)
				//		{
				//			minChi = std::min(minChi, n.chi);
				//			maxChi = std::max(maxChi, n.chi);
				//			for (int i = 0; i < 8; ++i)
				//				if (n.children[i]) FindMinMax(*n.children[i]);
				//		};
				//	FindMinMax(root);
				//	VisualizeChi(root, minChi, maxChi);
				//}


				{
					//TS(QueryAABB);
					//std::vector<unsigned int> results;
					//QueryAABB(root, { { -10.0f, -10.0f, -10.0f }, { 10.0f, 10.0f, 10.0f } }, inputPositions, results);
					//TE(QueryAABB);

					//for (size_t i = 0; i < results.size(); i++)
					//{
					//	auto& p = inputPositions[results[i]];

					//	VD::AddSphere("in AABB", { XYZ(p) }, 0.05f, Color::red());
					//}
				}


				{
					//TS(QuerySphere);
					//std::vector<unsigned int> results;
					//QuerySphere(root, make_float3(0.0f, 0.0f, 0.0f), 10.0f, inputPositions, results);
					//TE(QuerySphere);

					//for (size_t i = 0; i < results.size(); i++)
					//{
					//	auto& p = inputPositions[results[i]];

					//	VD::AddSphere("in AABB", { XYZ(p) }, 0.05f, Color::red());
					//}
				}

				{
					TS(FindKNN);
					float3 query = make_float3(0.2f, 0.1f, 0.5f);
					int k = 5000;

					auto knn = FindKNN(root, inputPositions, query, k);


					printf("KNN (k=%d):\n", k);
					for (auto& [idx, dist2] : knn)
					{
						//printf("  idx=%u, dist=%.4f\n", idx, sqrtf(dist2));

						auto& p = inputPositions[idx];

						VD::AddSphere("in AABB", { XYZ(p) }, 0.05f, Color::red());
					}
					TE(FindKNN);
				}
			}

			TS(Octree);
			HPOctree octree;
			octree.Initialize(h_input.positions, h_input.numberOfPoints, { {XYZ(aabb.min)}, {XYZ(aabb.max)}}, OCTREE_DEPTH);
			TE(Octree);

			//{
			//	auto domain_aabb = octree.domainAABB;
			//	auto domain_length = octree.domain_length;

			//	std::vector<HPOctreeNode> h_nodes(octree.numberOfNodes);
			//	CUDA_COPY_D2H(h_nodes.data(), octree.d_nodes, sizeof(HPOctreeNode) * octree.numberOfNodes);

			//	std::function<void(unsigned int, unsigned int)> DrawNode;
			//	DrawNode = [&](unsigned int nodeIndex, unsigned int depth) {
			//		const HPOctreeNode& node = h_nodes[nodeIndex];
			//		const uint64_t& code = node.key.ToCode();
			//		auto aabb = node.key.GetAABB(domain_aabb, domain_length);
			//		std::string name = "HPOctreeKey_" + std::to_string(node.key.d);
			//		VD::AddWiredBox(name, { glm::vec3(XYZ(aabb.min)), glm::vec3(XYZ(aabb.max)) }, Color::blue());

			//		if (node.childCount > 0)
			//		{
			//			for (unsigned int i = 0; i < node.childCount; ++i)
			//			{
			//				DrawNode(node.firstChildIndex + i, depth + 1);
			//			}
			//		}
			//		};

			//	DrawNode(0, 0);

			//	for (size_t i = 0; i <= octree.maxDepth; i++)
			//	{
			//		std::string name = "HPOctreeKey_" + std::to_string(i);
			//		VD::AddToSelectionList(name);
			//	}
			//}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
