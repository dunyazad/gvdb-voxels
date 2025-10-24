#pragma once

#include <cuda_common.cuh>

#include <queue>

struct HybridHostOctreeNode
{
	cuAABB bounds;
	unsigned int parent = UINT32_MAX;
	std::array<unsigned int, 8> children;  // -1이면 없음
	bool isLeaf = true;

	float3 V = make_float3(0, 0, 0);
	float divergence = 0.0f;
	float chi = 0.0f;

	std::vector<unsigned int> indices;
	HybridHostOctreeNode() { children.fill(UINT32_MAX); }
};


struct HybridDeviceOctreeNode
{
	cuAABB bounds;
	unsigned int parent = UINT32_MAX;
	unsigned int children[8];  // -1이면 없음
	uint8_t isLeaf = 1;
	uint8_t _pad0[3]{};

	float3 V = make_float3(0, 0, 0);
	float divergence = 0.0f;
	float chi = 0.0f;

	unsigned int indexOffset = 0;
	unsigned int indexCount = 0;
};


struct HybridOctree
{
	HybridOctree(unsigned int maxDepth = 12);
	~HybridOctree();

	std::vector<HybridHostOctreeNode> host_nodes;
	HybridDeviceOctreeNode* device_nodes = nullptr;
	unsigned int* device_flatIndices = nullptr;

	void Subdivide(
		unsigned int nodeIndex,
		const std::vector<float3>& points,
		unsigned int maxPoints, unsigned int maxDepth, unsigned int depth);

	void Build(
		const std::vector<float3>& points,
		const cuAABB& aabb,
		unsigned int maxPoints, unsigned int maxDepth);

	void DrawNode(unsigned int nodeIndex, unsigned int depth);

	void Draw();

	std::vector<unsigned int> QueryPoints(const std::vector<float3>& queryPoints);

	void QueryRange(const std::vector<cuAABB>& ranges, unsigned int maxOutPerQuery,
		std::vector<unsigned int>& outFlatIndices, std::vector<unsigned int>& outCounts);

	void HybridOctree::QueryRadius(
		const std::vector<float3>& centers,
		const std::vector<float>& radii,
		unsigned int maxOutPerQuery,
		std::vector<unsigned int>& outFlatIndices,
		std::vector<unsigned int>& outCounts);

	std::vector<unsigned int> QueryNearestNode(const std::vector<float3>& queryPoints);

	void QueryKNNNode_K8(
		const std::vector<float3>& queryPoints,
		std::vector<unsigned int>& outFlatK,  /* size = queryCount * 8 */
		std::vector<unsigned int>& outCounts /* size = queryCount */);
	
	unsigned int maxDepth = 12;
	std::vector<glm::vec4> interpolatedColors;
};