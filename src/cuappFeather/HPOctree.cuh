#pragma once

#include <cuda_common.cuh>

#include <SimpleHashMap.hpp>

namespace MortonUtils
{
	__host__ __device__ inline uint64_t
		Part1By2(uint32_t n)
	{
		uint64_t res = n;
		res &= 0x000fffff;
		res = (res | res << 32) & 0x001f00000000ffff;
		res = (res | res << 16) & 0x001f0000ff0000ff;
		res = (res | res << 8) & 0x100f00f00f00f00f;
		res = (res | res << 4) & 0x10c30c30c30c30c3;
		res = (res | res << 2) & 0x1249249249249249;
		return res;
	}

	__host__ __device__ inline uint32_t
		Compact1By2(uint64_t n)
	{
		n &= 0x1249249249249249;
		n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3;
		n = (n ^ (n >> 4)) & 0x100f00f00f00f00f;
		n = (n ^ (n >> 8)) & 0x001f0000ff0000ff;
		n = (n ^ (n >> 16)) & 0x001f00000000ffff;
		n = (n ^ (n >> 32)) & 0x00000000000fffff;
		return (uint32_t)n;
	}

	__host__ __device__ inline uint64_t
		EncodeMorton3D(unsigned int x, unsigned int y, unsigned int z)
	{
		return (Part1By2(x) << 2) | (Part1By2(y) << 1) | Part1By2(z);
	}

	__host__ __device__ inline void
		DecodeMorton3D(uint64_t code, unsigned int* x, unsigned int* y, unsigned int* z)
	{
		*z = Compact1By2(code); *y = Compact1By2(code >> 1); *x = Compact1By2(code >> 2);
	}
}

struct HPOctreeKey
{
	unsigned int d = UINT32_MAX;
	unsigned int x = UINT32_MAX;
	unsigned int y = UINT32_MAX;
	unsigned int z = UINT32_MAX;

	static const int DEPTH_BITS = 4;
	static const int COORD_BITS = 20;
	static const int DEPTH_SHIFT = COORD_BITS * 3;
	static const uint64_t DEPTH_MASK = (1ULL << DEPTH_BITS) - 1;
	static const uint64_t COORD_MASK = (1ULL << DEPTH_SHIFT) - 1;

	__host__ __device__ static float GetDomainLength(const cuAABB& aabb)
	{
		const float dx = aabb.max.x - aabb.min.x;
		const float dy = aabb.max.y - aabb.min.y;
		const float dz = aabb.max.z - aabb.min.z;
		return max(dx, max(dy, dz));
	}

	__host__ __device__ static cuAABB GetDomainAABB(const cuAABB& aabb)
	{
		auto domain_length = GetDomainLength(aabb);
		float half_len = domain_length * 0.5f;
		auto center = (aabb.min + aabb.max) * 0.5f;
		cuAABB domain_aabb;
		domain_aabb.min = make_float3(center.x - half_len, center.y - half_len, center.z - half_len);
		domain_aabb.max = make_float3(center.x + half_len, center.y + half_len, center.z + half_len);
		return domain_aabb;
	}

	__host__ __device__ static float GetCellSize(unsigned int depth, const cuAABB& aabb)
	{
		const float domain_length = GetDomainLength(aabb);
		if (domain_length <= 0.0f || depth >= 31)
		{
			return 0.0f;
		}

		return GetCellSize(depth, domain_length);
	}

	__host__ __device__ static float GetCellSize(unsigned int depth, float domain_length)
	{
		const unsigned int num_cells_along_axis = 1u << depth;
		return domain_length / (float)num_cells_along_axis;
	}

	__host__ __device__ float GetCellSize(const cuAABB& aabb) const
	{
		return GetCellSize(this->d, aabb);
	}

	__host__ __device__ float GetCellSize(float domain_length) const
	{
		return GetCellSize(this->d, domain_length);
	}

	__host__ __device__ static inline cuAABB GetAABB(const HPOctreeKey& key, const cuAABB& octree_aabb)
	{
		const float domain_length = HPOctreeKey::GetDomainLength(octree_aabb);

		return GetAABB(key, octree_aabb, domain_length);
	}

	__host__ __device__ static inline cuAABB GetAABB(const HPOctreeKey& key, const cuAABB& octree_aabb, float domain_length)
	{
		const float cell_size = HPOctreeKey::GetCellSize(key.d, domain_length);

		float3 min_corner;
		min_corner.x = octree_aabb.min.x + static_cast<float>(key.x) * cell_size;
		min_corner.y = octree_aabb.min.y + static_cast<float>(key.y) * cell_size;
		min_corner.z = octree_aabb.min.z + static_cast<float>(key.z) * cell_size;

		float3 max_corner;
		max_corner.x = min_corner.x + cell_size;
		max_corner.y = min_corner.y + cell_size;
		max_corner.z = min_corner.z + cell_size;

		cuAABB cell_aabb;
		cell_aabb.min = min_corner;
		cell_aabb.max = max_corner;

		return cell_aabb;
	}

	__host__ __device__ inline cuAABB GetAABB(const cuAABB& octree_aabb) const
	{
		const float domain_length = HPOctreeKey::GetDomainLength(octree_aabb);

		return GetAABB(*this, octree_aabb, domain_length);
	}

	__host__ __device__ inline cuAABB GetAABB(const cuAABB& octree_aabb, float domain_length) const
	{
		return GetAABB(*this, octree_aabb, domain_length);
	}

	__host__ __device__ float3 ToPosition(const cuAABB& aabb) const
	{
		const float domain_length = GetDomainLength(aabb);
		if (domain_length <= 0.0f || d >= 31)
		{
			return aabb.min;
		}

		return ToPosition(aabb, domain_length);
	}

	__host__ __device__ float3 ToPosition(const cuAABB& aabb, float domain_length) const
	{
		const float cell_size = GetCellSize(this->d, domain_length);

		const float px = aabb.min.x + ((float)x + 0.5f) * cell_size;
		const float py = aabb.min.y + ((float)y + 0.5f) * cell_size;
		const float pz = aabb.min.z + ((float)z + 0.5f) * cell_size;

		return make_float3(px, py, pz);
	}

	__host__ __device__ void FromPosition(const float3& position, const cuAABB& aabb, unsigned int target_depth = 0)
	{
		const float domain_length = GetDomainLength(aabb);
		if (domain_length <= 0.0f)
		{
			this->d = 0; this->x = 0; this->y = 0; this->z = 0;
			return;
		}

		FromPosition(position, aabb, domain_length);
	}

	__host__ __device__ void FromPosition(const float3& position, const cuAABB& aabb, float domain_length, unsigned int target_depth = UINT32_MAX)
	{
		if(UINT32_MAX == target_depth) target_depth = (1u << DEPTH_BITS) - 1; // 15

		this->d = target_depth;

		const float norm_x = (position.x - aabb.min.x) / domain_length;
		const float norm_y = (position.y - aabb.min.y) / domain_length;
		const float norm_z = (position.z - aabb.min.z) / domain_length;

		const unsigned int num_cells_along_axis = 1u << target_depth;
		int ix = (int)(norm_x * num_cells_along_axis);
		int iy = (int)(norm_y * num_cells_along_axis);
		int iz = (int)(norm_z * num_cells_along_axis);

		const int max_coord = (int)num_cells_along_axis - 1;
		this->x = max(0, min(ix, max_coord));
		this->y = max(0, min(iy, max_coord));
		this->z = max(0, min(iz, max_coord));
	}

	__host__ __device__ uint64_t ToCode() const
	{
		uint64_t morton_code = MortonUtils::EncodeMorton3D(x, y, z);
		return ((uint64_t)d << DEPTH_SHIFT) | morton_code;
	}

	__host__ __device__ void FromCode(uint64_t code)
	{
		this->d = (unsigned int)((code >> DEPTH_SHIFT) & DEPTH_MASK);
		uint64_t morton_code = code & COORD_MASK;
		MortonUtils::DecodeMorton3D(morton_code, &this->x, &this->y, &this->z);
	}

	__host__ __device__ HPOctreeKey GetParentKey() const
	{
		if (this->d == 0)
		{
			return *this;
		}

		HPOctreeKey parent_key;

		parent_key.d = this->d - 1;

		parent_key.x = this->x >> 1;
		parent_key.y = this->y >> 1;
		parent_key.z = this->z >> 1;

		return parent_key;
	}

	__host__ __device__ HPOctreeKey GetAncestorKey(unsigned int target_depth) const
	{
		if (target_depth >= this->d)
		{
			return *this;
		}

		HPOctreeKey ancestor_key;

		const unsigned int depth_diff = this->d - target_depth;

		ancestor_key.d = target_depth;

		ancestor_key.x = this->x >> depth_diff;
		ancestor_key.y = this->y >> depth_diff;
		ancestor_key.z = this->z >> depth_diff;

		return ancestor_key;
	}

	__host__ __device__ explicit operator size_t() const
	{
		return static_cast<size_t>(this->ToCode());
	}

	__host__ __device__ bool operator<(const HPOctreeKey& other) const
	{
		return this->ToCode() < other.ToCode();
	}

	__host__ __device__ bool operator==(const HPOctreeKey& other) const
	{
		return d == other.d && x == other.x && y == other.y && z == other.z;
	}
};

namespace detail
{
	// host/device 양쪽에서 사용 가능한 float3 거리 제곱 계산 함수
	__host__ __device__ inline float DistanceSq(const float3& p1, const float3& p2)
	{
		float dx = p1.x - p2.x;
		float dy = p1.y - p2.y;
		float dz = p1.z - p2.z;
		return dx * dx + dy * dy + dz * dz;
	}

	// host/device 양쪽에서 사용 가능한 점에서 AABB까지의 거리 제곱 계산 함수
	__host__ __device__ inline float PointToAABBDistanceSq(const float3& p, const cuAABB& aabb)
	{
		float dist_sq = 0.0f;

		// x 축
		if (p.x < aabb.min.x) dist_sq += (aabb.min.x - p.x) * (aabb.min.x - p.x);
		else if (p.x > aabb.max.x) dist_sq += (p.x - aabb.max.x) * (p.x - aabb.max.x);

		// y 축
		if (p.y < aabb.min.y) dist_sq += (aabb.min.y - p.y) * (aabb.min.y - p.y);
		else if (p.y > aabb.max.y) dist_sq += (p.y - aabb.max.y) * (p.y - aabb.max.y);

		// z 축
		if (p.z < aabb.min.z) dist_sq += (aabb.min.z - p.z) * (aabb.min.z - p.z);
		else if (p.z > aabb.max.z) dist_sq += (p.z - aabb.max.z) * (p.z - aabb.max.z);

		return dist_sq;
	}
}

struct NNS_Result
{
	unsigned int index = UINT32_MAX;
	float distance_sq = FLT_MAX;
};

struct HPOctreeNode
{
	unsigned int parentIndex = UINT32_MAX;
	unsigned int firstChildIndex = UINT32_MAX;
	unsigned int childCount = UINT32_MAX;
	unsigned int pointIndex = UINT32_MAX;

	HPOctreeKey key;

	__host__ __device__ HPOctreeNode()
		: parentIndex(UINT32_MAX), firstChildIndex(UINT32_MAX), childCount(0)
	{
	}
};

struct HPOctree
{
	SimpleHashMap<uint64_t, unsigned int> keys;
	SimpleHashMap<uint64_t, unsigned int> leaf_map;

	void Initialize(const vector<float3>& positions, const cuAABB& aabb, unsigned int maxDepth = 0);
	void Initialize(const float3* h_positions, unsigned int numberOfPositions, const cuAABB& aabb, unsigned int maxDepth = 0);

	void Terminate();

	std::vector<NNS_Result> Search(const std::vector<float3>& h_queries);

	vector<HPOctreeKey> Dump();

	cuAABB originalAABB;
	cuAABB domainAABB;
	float domain_length =0.0f;
	unsigned int maxDepth = 0;
	float3* d_positions = nullptr;
	unsigned int numberOfPositions = 0;
	HPOctreeNode* d_nodes = nullptr;
	unsigned int numberOfNodes = 0;

	std::vector<NNS_Result> SearchOnDevice(const std::vector<float3>& h_query_points);
};

namespace std
{
	template<>
	struct hash<HPOctreeKey>
	{
		size_t operator()(const HPOctreeKey& key) const noexcept
		{
			return static_cast<size_t>(key.ToCode());
		}
	};
}



__device__ NNS_Result FindNearestNeighbor(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	const float3& query_pos,
	const float3* d_positions,
	unsigned int numPositions,
	cuAABB domainAABB,
	float domain_length);

__global__ void FindNearestNeighbor_kernel(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	const float3* d_query_points,
	NNS_Result* d_results,
	const float3* d_positions,
	unsigned int numQueries,
	unsigned int numPositions,
	cuAABB domainAABB,
	float domain_length,
	unsigned int maxDepth);