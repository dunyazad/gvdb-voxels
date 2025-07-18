#pragma once

#include <cuda_common.cuh>

#include <PointCloud.cuh>

using VoxelKey = uint64_t;
#define EMPTY_KEY UINT64_MAX
#define VALID_KEY(k) ((k) != EMPTY_KEY)

struct Voxel
{
	int3 coordinate = make_int3(INT32_MAX, INT32_MAX, INT32_MAX);
	float3 normalSum = make_float3(0, 0, 0);
	float3 colorSum = make_float3(0, 0, 0);
	float sdfSum = 0.0f;
	unsigned int count = 0;
};

struct VoxelHashEntry
{
	VoxelKey key;
	Voxel voxel;
};

struct VoxelHashMapInfo
{
	VoxelHashEntry* entries = nullptr;
	size_t capacity = 1 << 24;
	unsigned int maxProbe = 64;
	float voxelSize = 0.1f;

	unsigned int* d_numberOfOccupiedVoxels = nullptr;
	int3* d_occupiedVoxelIndices = nullptr;
	unsigned int h_numberOfOccupiedVoxels = 0;
	unsigned int h_occupiedCapacity = 0;
};

__global__ void Kernel_ClearHashMap(VoxelHashMapInfo info);

__global__ void Kernel_OccupyVoxelHashMap(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors,
	unsigned int numberOfPoints);

__global__ void Kernel_OccupySDF(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors,
	unsigned int numberOfPoints,
	int offset = 1);

__global__ void Kernel_SerializeVoxelHashMap(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors);

__global__ void Kernel_SerializeVoxelHashMap_SDF(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors);

__global__ void Kernel_VoxelHashMap_FindOverlap(VoxelHashMapInfo info);

__global__ void Kernel_SmoothSDF_VoxelHashMap(VoxelHashMapInfo info, float smoothingFactor);

__global__ void Kernel_FilterOppositeNormals(VoxelHashMapInfo info, float thresholdDotCos);

__global__ void Kernel_FilterByNormalGradient(VoxelHashMapInfo info, float gradientThreshold, bool remove);

struct VoxelHashMap
{
	VoxelHashMapInfo info;

	void Initialize(float voxelSize = 0.1f, size_t capacity = 1 << 24, unsigned int maxProbe = 64);
	void Terminate();

	void CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy);

	void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints);
	void Occupy(const DevicePointCloud& d_input);

	void Occupy_SDF(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints, int offset = 1);
	void Occupy_SDF(const DevicePointCloud& d_input, int offset = 1);

	HostPointCloud Serialize();
	HostPointCloud Serialize_SDF();

	void FindOverlap();

	void SmoothSDF(float smoothingFactor = 1.0f, int iterations = 1);

	void FilterOppositeNormals();

	void FilterByNormalGradient(float gradientThreshold, bool remove);

	__host__ __device__ static uint64_t expandBits(uint32_t v);

	__host__ __device__ static uint32_t compactBits(uint64_t x);

	__host__ __device__ static VoxelKey IndexToVoxelKey(const int3& coord);

	__host__ __device__ static int3 VoxelKeyToIndex(VoxelKey key);

	__host__ static int3 PositionToIndex_host(float3 pos, float voxelSize);

	__device__ static int3 PositionToIndex_device(float3 pos, float voxelSize);

	__host__ __device__ static int3 PositionToIndex(float3 pos, float voxelSize);

	__host__ __device__ static float3 IndexToPosition(int3 index, float voxelSize);

	__host__ __device__ static size_t hash(VoxelKey key, size_t capacity);

	__device__ static Voxel* GetVoxel(VoxelHashMapInfo& info, const int3& index);

	__device__ static bool isZeroCrossing(VoxelHashMapInfo& info, int3 index, float sdfCenter);

	__device__ static bool computeInterpolatedSurfacePoint_6(
		VoxelHashMapInfo& info,
		int3 index,
		float sdfCenter,
		float3& outPosition,
		float3& outNormal,
		float3& outColor);

	__device__ static bool computeInterpolatedSurfacePoint_26(
		VoxelHashMapInfo& info,
		int3 index,
		float sdfCenter,
		float3& outPosition,
		float3& outNormal,
		float3& outColor);

	__device__ static bool isSimpleVoxel(VoxelHashMapInfo& info, int3 index);
};
