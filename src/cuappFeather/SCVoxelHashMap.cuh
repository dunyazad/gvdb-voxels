#pragma once

#include <cuda_common.cuh>

#include <VoxelKey.hpp>
#include <PointCloud.cuh>
#include <HalfEdgeMesh.cuh>
#include <VEFM.cuh>

struct SCVoxel
{
	int3 coordinate = make_int3(INT32_MAX);
	float3 normalSum = make_float3(0.0f);
	float3 colorSum = make_float3(0.0f);
	float sdfSum = FLT_MAX;
	unsigned int count = 0;

	uint3 zeroCrossingPointIndex = make_uint3(UINT32_MAX);
};

struct SCVoxelHashMapEntry
{
	VoxelKey key;
	SCVoxel voxel;
};

struct SCVoxelHashMapInfo
{
	SCVoxelHashMapEntry* entries = nullptr;
	size_t capacity = 1 << 24;
	unsigned int maxProbe = 64;
	float voxelSize = 0.1f;

	unsigned int* d_numberOfOccupiedVoxels = nullptr;
	int3* d_occupiedVoxelIndices = nullptr;
	unsigned int h_numberOfOccupiedVoxels = 0;
	unsigned int h_occupiedCapacity = 0;
};

struct SCVoxelHashMap
{
	SCVoxelHashMapInfo info;

	void Initialize(float voxelSize = 0.1f, size_t capacity = 1 << 24, unsigned int maxProbe = 64);
	void Terminate();

	void CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy);

	void Occupy(const DevicePointCloud& d_input, int offset = 1);
	void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints, int offset = 1);

	HostPointCloud Serialize();

	void MarchingCubes(DeviceHalfEdgeMesh& mesh, float isoValue = 0.0f);

	void SurfaceProjection_SDF(SCVoxelHashMapInfo info, DeviceHalfEdgeMesh& mesh, int maxIters = 5, float stepScale = 0.5f);

	__device__ static SCVoxel* GetVoxel(SCVoxelHashMapInfo& info, const int3& index);

	__device__ static SCVoxel* InsertVoxel(
		SCVoxelHashMapInfo& info, const int3& index, float sdf, float3 normal, float3 color);

	__device__ static bool CheckZeroCrossing(SCVoxelHashMapInfo& info,
		float osdf, const float3& op, const float3& on, const float3& oc,
		const int3& index, float3& outPosition, float3& outNormal, float3& outColor);

	__device__ static float GetSDFValue(SCVoxelHashMapInfo info, float isoValue, const int3& voxelIndex);

	__device__ static unsigned int BuildCubeIndex(SCVoxelHashMapInfo info, float isoValue, const int3& voxelIndex);

	__device__ static unsigned int GetZeroCrossingIndex(
		SCVoxelHashMapInfo& info,const int3& baseIndex, const SCVoxel* baseVoxel, int edge);

	__device__ static float SampleSDF(const SCVoxelHashMapInfo& info, const float3& p);
	__device__ static float3 SampleSDFGradient(const SCVoxelHashMapInfo& info, const float3& p);

	__device__ static float TrilinearSampleSDF(const SCVoxelHashMapInfo& info, const float3& p);
	__device__ static float3 TrilinearSampleSDFGradient(const SCVoxelHashMapInfo& info, const float3& p);
};



__global__ void Kernel_SCVoxelHashMap_Clear(SCVoxelHashMapInfo info);

__global__ void Kernel_SCVoxelHashMap_Occupy(
	SCVoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors,
	unsigned int numberOfPoints,
	int offset);

__global__ void Kernel_SCVoxelHashMap_CreateZeroCrossingPoints(
	SCVoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors,
	unsigned int* d_numberOfPoints);

__global__ void Kernel_SCVoxelHashMap_MarchingCubes(
	SCVoxelHashMapInfo info, float isoValue, uint3* d_faces, unsigned int* d_numberOfFaces);

__global__ void Kernel_SCVoxelHashMap_SurfaceProjection_SDF(
	SCVoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	unsigned int numberOfPoints,
	int maxIters = 5,
	float stepScale = 0.5f);
