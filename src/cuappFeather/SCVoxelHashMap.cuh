#include <cuda_common.cuh>

#include <PointCloud.cuh>
#include <HalfEdgeMesh.cuh>
#include <VEFM.cuh>

using SCVoxelKey = uint64_t;
#ifndef EMPTY_KEY
#define EMPTY_KEY UINT64_MAX
#define VALID_KEY(k) ((k) != EMPTY_KEY)
#endif

struct SCVoxel
{
	int3 coordinate = make_int3(INT32_MAX);
	float3 normalSum = make_float3(0.0f);
	float3 colorSum = make_float3(0.0f);
	float sdfSum = FLT_MAX;
	unsigned int count = 0;

	uint3 zeroCrossingPointIndex = make_uint3(UINT32_MAX);
};

struct SCVoxelHashEntry
{
	SCVoxelKey key;
	SCVoxel voxel;
};

struct SCVoxelHashMapInfo
{
	SCVoxelHashEntry* entries = nullptr;
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

	HostPointCloud Serialize();

	void MarchingCubes(DeviceHalfEdgeMesh& mesh, float isoValue = 0.0f);

	void SurfaceProjection_SDF(SCVoxelHashMapInfo info, DeviceHalfEdgeMesh& mesh, int maxIters = 5, float stepScale = 0.5f);

	__host__ __device__ static uint64_t expandBits(uint32_t v);
	__host__ __device__ static uint32_t compactBits(uint64_t x);
	__host__ __device__ static size_t hash(SCVoxelKey key, size_t capacity);

	__host__ __device__ static SCVoxelKey IndexToVoxelKey(const int3& coord);
	__host__ __device__ static int3 VoxelKeyToIndex(SCVoxelKey key);
	__host__ __device__ static int3 PositionToIndex(const float3& pos, float voxelSize);
	__host__ __device__ static float3 IndexToPosition(const int3& index, float voxelSize);

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