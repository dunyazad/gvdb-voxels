#pragma once

#include <cuda_common.cuh>

#include <HashMap.hpp>

struct MarginLineFinder
{
	HashMap<uint64_t, uint64_t> pointMap;
	HashMap<uint64_t, uint64_t> countMap;

	void Initialize(float voxelSize, size_t capacity, uint8_t maxProbe = 64);

	void Terminate();

	void Clear();

	void InsertPoints(const std::vector<float3>& points, uint64_t tag);
	void InsertPoints(const float3* d_points, int numberOfPoints, uint64_t tag);

	void Dump(std::vector<float3>& resultPositions, std::vector<uint64_t>& resultTags);
	void Dump(float3* d_resultPositions, uint64_t* d_resultTags, unsigned int* d_numberofResultPositions);

	void FindMarginLinePoints(std::vector<float3>& result);
	void FindMarginLinePoints(float3* d_resultPoints, unsigned int* d_numberofResultPoints);

	void MarginLineNoiseRemoval(std::vector<float3>& result);
	void MarginLineNoiseRemoval(float3* d_resultPoints, unsigned int* d_numberofResultPoints);

	void Clustering(std::vector<float3>& resultPositions, std::vector<uint64_t>& resultTags, std::vector<uint64_t>& resultTagCounts);
	void Clustering(float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions);

	float voxelSize = 0.1f;

	__host__ __device__ static inline uint64_t ToKey(float3 position, float voxelSize = 0.1f)
	{
		int ix = static_cast<int>(floorf(position.x / voxelSize));
		int iy = static_cast<int>(floorf(position.y / voxelSize));
		int iz = static_cast<int>(floorf(position.z / voxelSize));

		// 각 인덱스가 21비트 부호 있는 정수 범위 내에 있는지 확인
		constexpr int max_coord = (1 << 20) - 1; // 1048575
		constexpr int min_coord = -(1 << 20);    // -1048576
		assert(ix >= min_coord && ix <= max_coord);
		assert(iy >= min_coord && iy <= max_coord);
		assert(iz >= min_coord && iz <= max_coord);

		uint64_t key = (static_cast<uint64_t>(ix) & 0x1FFFFF) |
			((static_cast<uint64_t>(iy) & 0x1FFFFF) << 21) |
			((static_cast<uint64_t>(iz) & 0x1FFFFF) << 42);
		return key;
	}

	__host__ __device__ static inline uint64_t ToKey(int3 index)
	{
		uint64_t key =
			(static_cast<uint64_t>(index.x) & 0x1FFFFF) |
			((static_cast<uint64_t>(index.y) & 0x1FFFFF) << 21) |
			((static_cast<uint64_t>(index.z) & 0x1FFFFF) << 42);
		return key;
	}

	__host__ __device__ static inline int3 ToIndex(float3 position, float voxelSize)
	{
		int ix = static_cast<int>(floorf(position.x / voxelSize));
		int iy = static_cast<int>(floorf(position.y / voxelSize));
		int iz = static_cast<int>(floorf(position.z / voxelSize));
		return make_int3(ix, iy, iz);
	}

	__host__ __device__ static inline float3 FromKey(uint64_t key, float voxelSize)
	{
		int ix = static_cast<int>(key & 0x1FFFFF);
		int iy = static_cast<int>((key >> 21) & 0x1FFFFF);
		int iz = static_cast<int>((key >> 42) & 0x1FFFFF);
		if (ix & 0x100000) ix |= 0xFFE00000;
		if (iy & 0x100000) iy |= 0xFFE00000;
		if (iz & 0x100000) iz |= 0xFFE00000;
		return make_float3((ix + 0.5f) * voxelSize, (iy + 0.5f) * voxelSize, (iz + 0.5f) * voxelSize);
	}

	__host__ __device__ static inline float3 FromIndex(int3 index, float voxelSize = 0.1f)
	{
		return make_float3((index.x + 0.5f) * voxelSize, (index.y + 0.5f) * voxelSize, (index.z + 0.5f) * voxelSize);
	}

	__host__ __device__ static inline int3 KeyToIndex(uint64_t key)
	{
		int ix = static_cast<int>(key & 0x1FFFFF);
		int iy = static_cast<int>((key >> 21) & 0x1FFFFF);
		int iz = static_cast<int>((key >> 42) & 0x1FFFFF);
		if (ix & 0x100000) ix |= 0xFFE00000;
		if (iy & 0x100000) iy |= 0xFFE00000;
		if (iz & 0x100000) iz |= 0xFFE00000;
		return make_int3(ix, iy, iz);
	}

	__device__ static inline uint64_t FindRootVoxel(HashMapInfo<uint64_t, uint64_t> info, uint64_t key);
	__device__ static inline void UnionVoxel(HashMapInfo<uint64_t, uint64_t> info, uint64_t a, uint64_t b);
};
