#pragma once

#include <cuda_common.cuh>

using VoxelKey = uint64_t;
#ifndef EMPTY_KEY
#define EMPTY_KEY UINT64_MAX
#define VALID_KEY(k) ((k) != EMPTY_KEY)
#endif


__host__ __device__ inline uint64_t expandBits(uint32_t v)
{
	uint64_t x = v & 0x1fffff; // 21 bits
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__host__ __device__ inline uint32_t compactBits(uint64_t x)
{
	x &= 0x1249249249249249;
	x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
	x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
	x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
	x = (x ^ (x >> 16)) & 0x1f00000000ffff;
	x = (x ^ (x >> 32)) & 0x1fffff;
	return static_cast<uint32_t>(x);
}

__host__ __device__ inline VoxelKey IndexToVoxelKey(const int3& coord)
{
	// To handle negative values
	const int OFFSET = 1 << 20; // 2^20 = 1048576
	return (expandBits(coord.z + OFFSET) << 2) |
		(expandBits(coord.y + OFFSET) << 1) |
		expandBits(coord.x + OFFSET);
}

__host__ __device__ inline int3 VoxelKeyToIndex(VoxelKey key)
{
	const int OFFSET = 1 << 20;
	int x = static_cast<int>(compactBits(key));
	int y = static_cast<int>(compactBits(key >> 1));
	int z = static_cast<int>(compactBits(key >> 2));
	return make_int3(x - OFFSET, y - OFFSET, z - OFFSET);
}

__host__ inline int3 PositionToIndex_host(float3 pos, float voxelSize)
{
	return make_int3(
		static_cast<int>(std::floor(pos.x / voxelSize)),
		static_cast<int>(std::floor(pos.y / voxelSize)),
		static_cast<int>(std::floor(pos.z / voxelSize)));
}

#ifdef __CUDA_ARCH__
__device__ inline int3 PositionToIndex_device(float3 pos, float voxelSize)
{
	return make_int3(
		__float2int_rd(pos.x / voxelSize),
		__float2int_rd(pos.y / voxelSize),
		__float2int_rd(pos.z / voxelSize));
}
#endif

__host__ __device__ inline int3 PositionToIndex(float3 pos, float voxelSize)
{
#ifdef __CUDA_ARCH__
	return PositionToIndex_device(pos, voxelSize);
#else
	return PositionToIndex_host(pos, voxelSize);
#endif
}

__host__ __device__ inline float3 IndexToPosition(int3 index, float voxelSize)
{
	return make_float3(
		(index.x + 0.5f) * voxelSize,
		(index.y + 0.5f) * voxelSize,
		(index.z + 0.5f) * voxelSize);
}

__host__ __device__ inline size_t VoxelKeyHash(VoxelKey key, size_t capacity)
{
	key ^= (key >> 33);
	key *= 0xff51afd7ed558ccd;
	key ^= (key >> 33);
	key *= 0xc4ceb9fe1a85ec53;
	key ^= (key >> 33);
	return static_cast<size_t>(key) % capacity;
}
