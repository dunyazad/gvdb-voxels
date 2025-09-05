#pragma once

#include <cuda_common.cuh>

#include <VoxelKey.hpp>
#include <PointCloud.cuh>
#include <HalfEdgeMesh.cuh>
#include <VEFM.cuh>

template<typename T>
struct VoxelProperty {
	T property;
};

template<>
struct VoxelProperty<void> {};

template<typename T = void>
struct SCVoxel : VoxelProperty<T>
{
	int3 coordinate = make_int3(INT32_MAX);
	float3 normalSum = make_float3(0.0f);
	float3 colorSum = make_float3(0.0f);
	float sdfSum = FLT_MAX;
	unsigned int count = 0;

	uint3 zeroCrossingPointIndex = make_uint3(UINT32_MAX);
};

template<typename T = void>
struct SCVoxelHashMapEntry
{
	VoxelKey key;
	SCVoxel<T> voxel;
};

template<typename T = void>
struct SCVoxelHashMapInfo
{
	SCVoxelHashMapEntry<T>* entries = nullptr;
	size_t capacity = 1 << 24;
	unsigned int maxProbe = 64;
	float voxelSize = 0.1f;

	unsigned int* d_numberOfOccupiedVoxels = nullptr;
	int3* d_occupiedVoxelIndices = nullptr;
	unsigned int h_numberOfOccupiedVoxels = 0;
	unsigned int h_occupiedCapacity = 0;
};

template<typename T>
struct SCVoxelHashMap;

template<typename T>
__global__ __forceinline__ void Kernel_SCVoxelHashMap_Clear(SCVoxelHashMapInfo<T> info)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= info.capacity) return;

	info.entries[idx].key = EMPTY_KEY;
	info.entries[idx].voxel = {};
	info.entries[idx].voxel.zeroCrossingPointIndex = make_uint3(UINT32_MAX);
}

template<typename T>
__global__ __forceinline__ void Kernel_SCVoxelHashMap_Occupy(
	SCVoxelHashMapInfo<T> info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors,
	T* d_properties,
	unsigned int numberOfPoints,
	int offset)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numberOfPoints) return;

	float3 p = d_positions[tid];
	float3 n = d_normals[tid];
	float3 c = d_colors ? d_colors[tid] : make_float3(0, 0, 0);
	T& property = d_properties[tid];

	int3 baseIndex = PositionToIndex(p, info.voxelSize);

	for (int dz = -offset; dz <= offset; ++dz)
	{
		for (int dy = -offset; dy <= offset; ++dy)
		{
			for (int dx = -offset; dx <= offset; ++dx)
			{
				int3 index = make_int3(baseIndex.x + dx, baseIndex.y + dy, baseIndex.z + dz);
				float3 center = IndexToPosition(index, info.voxelSize);

				float3 dir = center - p;
				float dist = length(dir);
				if ((dist < 1e-6f) || (info.voxelSize * (float)offset < dist)) continue;

				float sign = (dot(n, dir) >= 0.0f) ? 1.0f : -1.0f;
				float sdf = dist * sign;

				if constexpr (std::is_void_v<T>)
				{
					SCVoxelHashMap<T>::InsertVoxel(info, index, sdf, n, c, nullptr);
				}
				else
				{
					SCVoxelHashMap<T>::InsertVoxel(info, index, sdf, n, c, &property);
				}
			}
		}
	}
}

template<typename T>
__global__ __forceinline__ void Kernel_SCVoxelHashMap_CreateZeroCrossingPoints(
	SCVoxelHashMapInfo<T> info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors,
	T* d_properties,
	unsigned int* d_numberOfPoints)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = SCVoxelHashMap<T>::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	float sdf = voxel->sdfSum / (float)max(1u, voxel->count);

	float3 voxelPosition = IndexToPosition(voxelIndex, info.voxelSize);
	float3 voxelNormal = voxel->normalSum / (float)max(1u, voxel->count);
	float3 voxelColor = voxel->colorSum / (float)max(1u, voxel->count);
	T& voxelProperty = voxel->property;

	// X Axis
	{
		auto neighborIndex = voxelIndex;
		neighborIndex.x += 1;
		float3 position;
		float3 normal;
		float3 color;
		T property;
		if (true == SCVoxelHashMap<T>::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color, &property))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
			if constexpr (!std::is_void_v<T>)
			{
				d_properties[index] = property;
			}
			voxel->zeroCrossingPointIndex.x = index;
		}
	}

	// Y Axis
	{
		auto neighborIndex = voxelIndex;
		neighborIndex.y += 1;
		float3 position;
		float3 normal;
		float3 color;
		T property;
		if (true == SCVoxelHashMap<T>::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color, &property))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
			if constexpr (!std::is_void_v<T>)
			{
				d_properties[index] = property;
			}
			voxel->zeroCrossingPointIndex.y = index;
		}
	}

	// Z Axis
	{
		auto neighborIndex = voxelIndex;
		neighborIndex.z += 1;
		float3 position;
		float3 normal;
		float3 color;
		T property;
		if (true == SCVoxelHashMap<T>::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color, &property))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
			if constexpr (!std::is_void_v<T>)
			{
				d_properties[index] = property;
			}
			voxel->zeroCrossingPointIndex.z = index;
		}
	}
}

template<typename T>
__global__ __forceinline__ void Kernel_SCVoxelHashMap_MarchingCubes(
	SCVoxelHashMapInfo<T> info, float isoValue, uint3* d_faces, unsigned int* d_numberOfFaces)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = SCVoxelHashMap<T>::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	auto cubeIndex = SCVoxelHashMap<T>::BuildCubeIndex(info, isoValue, voxelIndex);

	for (int i = 0; MC_TRI_TABLE[cubeIndex][i] != -1; i += 3)
	{
		int e0 = MC_TRI_TABLE[cubeIndex][i + 0];
		int e1 = MC_TRI_TABLE[cubeIndex][i + 1];
		int e2 = MC_TRI_TABLE[cubeIndex][i + 2];

		unsigned int v0 = SCVoxelHashMap<T>::GetZeroCrossingIndex(info, voxelIndex, voxel, e0);
		unsigned int v1 = SCVoxelHashMap<T>::GetZeroCrossingIndex(info, voxelIndex, voxel, e1);
		unsigned int v2 = SCVoxelHashMap<T>::GetZeroCrossingIndex(info, voxelIndex, voxel, e2);

		//if (v0 == UINT32_MAX || v1 == UINT32_MAX || v2 == UINT32_MAX)
		//{
		//	printf("Triangle dropped: e0=%d e1=%d e2=%d, v0=%u v1=%u v2=%u, base=(%d,%d,%d)\n",
		//		e0, e1, e2, v0, v1, v2, voxelIndex.x, voxelIndex.y, voxelIndex.z);
		//}

		if (v0 != UINT32_MAX && v1 != UINT32_MAX && v2 != UINT32_MAX)
		{
			unsigned int fidx = atomicAdd(d_numberOfFaces, 1u);
			d_faces[fidx] = make_uint3(v0, v2, v1);
		}
	}


	//if (UINT32_MAX != voxel->zeroCrossingPointIndex.x &&
	//	UINT32_MAX != voxel->zeroCrossingPointIndex.y &&
	//	UINT32_MAX != voxel->zeroCrossingPointIndex.z)
	//{
	//	auto index = atomicAdd(d_numberOfFaces, 1u);
	//	d_faces[index] = voxel->zeroCrossingPointIndex;
	//}
}

template<typename T>
__global__ __forceinline__ void Kernel_SCVoxelHashMap_SurfaceProjection_SDF(
	SCVoxelHashMapInfo<T> info,
	float3* positions,
	float3* normals,
	unsigned int numberOfPoints,
	int maxIters = 5,
	float stepScale = 0.5f)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numberOfPoints) return;

	float3 p = positions[tid];

	// Simple Sampling
	/*
	for (int iter = 0; iter < maxIters; ++iter)
	{
		float sdf = SCVoxelHashMap::SampleSDF(info, p);
		if (fabsf(sdf) < info.voxelSize * 0.05f)
			break;
		float3 grad = SCVoxelHashMap::SampleSDFGradient(info, p);
		if (length(grad) < 1e-6f) break;
		p = p - sdf * grad * stepScale;
	}
	*/

	for (int iter = 0; iter < maxIters; ++iter)
	{
		float sdf = SCVoxelHashMap<T>::TrilinearSampleSDF(info, p);
		if (fabsf(sdf) > info.voxelSize * 2.0f || sdf == FLT_MAX) break;
		if (fabsf(sdf) < info.voxelSize * 0.05f) break;
		float3 grad = SCVoxelHashMap<T>::TrilinearSampleSDFGradient(info, p);
		float glen = length(grad);
		if (glen < 1e-3f || isnan(glen)) break;
		p = p - sdf * grad * stepScale;
	}

	positions[tid] = p;

	// Update normal using SDF gradient (optional)
	if (normals)
	{
		float3 n = SCVoxelHashMap<T>::SampleSDFGradient(info, p);
		if (length(n) > 1e-6f)
			normals[tid] = n;
	}
}

template<typename T = void>
struct SCVoxelHashMap
{
	SCVoxelHashMapInfo<T> info;

	void Initialize(float voxelSize = 0.1f, size_t capacity = 1 << 24, unsigned int maxProbe = 64)
	{
		info.voxelSize = voxelSize;
		info.capacity = capacity;
		info.maxProbe = maxProbe;
		CUDA_MALLOC(&info.entries, sizeof(SCVoxelHashMapEntry<T>) * info.capacity);

		LaunchKernel(Kernel_SCVoxelHashMap_Clear<T>, (unsigned int)info.capacity, info);

		CUDA_SYNC();
	}

	void Terminate()
	{
		if (nullptr != info.entries)
		{
			CUDA_FREE(info.entries);
		}
		info.entries = nullptr;

		CUDA_SAFE_FREE(info.d_numberOfOccupiedVoxels);
		CUDA_SAFE_FREE(info.d_occupiedVoxelIndices);

		info.d_numberOfOccupiedVoxels = nullptr;
		info.d_occupiedVoxelIndices = nullptr;

		info.h_numberOfOccupiedVoxels = 0;
		info.h_occupiedCapacity = 0;
	}

	void CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy)
	{
		if (numberOfVoxelsToOccupy == 0) return;

		if (!info.d_occupiedVoxelIndices)
		{
			CUDA_MALLOC(&info.d_occupiedVoxelIndices, sizeof(uint3) * numberOfVoxelsToOccupy);
			CUDA_MALLOC(&info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
			unsigned int zero = 0;
			CUDA_COPY_H2D(info.d_numberOfOccupiedVoxels, &zero, sizeof(unsigned int));
			CUDA_SYNC();
			info.h_occupiedCapacity = numberOfVoxelsToOccupy;
		}
		else
		{
			CUDA_COPY_D2H(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
			CUDA_SYNC();
			unsigned int required = info.h_numberOfOccupiedVoxels + numberOfVoxelsToOccupy;
			if (required > info.h_occupiedCapacity)
			{
				int3* d_new = nullptr;
				CUDA_MALLOC(&d_new, sizeof(int3) * required);
				CUDA_COPY_D2D(d_new, info.d_occupiedVoxelIndices, sizeof(uint3) * info.h_numberOfOccupiedVoxels);
				CUDA_SYNC();
				CUDA_FREE(info.d_occupiedVoxelIndices);
				info.d_occupiedVoxelIndices = d_new;
				info.h_occupiedCapacity = required;
			}
		}

		printf("SCVoxelHashMap numberOfOccupiedVoxels capacity: %u\n", info.h_occupiedCapacity);
	}

	void Occupy(const DevicePointCloud<T>& d_input, int offset = 1)
	{
		CUDA_TS(SCVoxelHashMap_Occupy);

		if (1 > offset) offset = 1;
		int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

		CheckOccupiedIndicesLength(d_input.numberOfPoints * count);
		LaunchKernel(Kernel_SCVoxelHashMap_Occupy<T>, d_input.numberOfPoints,
			info, d_input.positions, d_input.normals, d_input.colors, d_input.properties, d_input.numberOfPoints, offset);
		CUDA_COPY_D2H(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
		CUDA_SYNC();

		printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
		CUDA_TE(SCVoxelHashMap_Occupy);
	}

	void Occupy(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints, int offset = 1)
	{
		CUDA_TS(SCVoxelHashMap_Occupy);

		if (1 > offset) offset = 1;
		int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

		CheckOccupiedIndicesLength(numberOfPoints * count);
		LaunchKernel(Kernel_SCVoxelHashMap_Occupy<T>, numberOfPoints,
			info, d_positions, d_normals, d_colors, numberOfPoints, offset);
		CUDA_COPY_D2H(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
		CUDA_SYNC();

		printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
		CUDA_TE(SCVoxelHashMap_Occupy);
	}

	HostPointCloud<T> Serialize()
	{
		HostPointCloud<T> h_result;
		if (info.h_numberOfOccupiedVoxels == 0) return h_result;

		float3* d_positions = nullptr;
		float3* d_normals = nullptr;
		float3* d_colors = nullptr;
		T* d_properties = nullptr;
		unsigned int* d_numberOfPoints = nullptr;

		CUDA_MALLOC(&d_positions, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
		CUDA_MALLOC(&d_normals, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
		CUDA_MALLOC(&d_colors, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
		if constexpr (!std::is_void_v<T>)
		{
			CUDA_MALLOC(&d_properties, sizeof(T) * info.h_numberOfOccupiedVoxels * 3);
		}
		CUDA_MALLOC(&d_numberOfPoints, sizeof(unsigned int));

		LaunchKernel(Kernel_SCVoxelHashMap_CreateZeroCrossingPoints<T>, info.h_numberOfOccupiedVoxels,
			info, d_positions, d_normals, d_colors, d_properties, d_numberOfPoints);
		CUDA_SYNC();

		unsigned int h_numberOfPoints = 0;
		CUDA_COPY_D2H(&h_numberOfPoints, d_numberOfPoints, sizeof(unsigned int));
		CUDA_SYNC();

		h_result.Initialize(h_numberOfPoints);
		CUDA_COPY_D2H(h_result.positions, d_positions, sizeof(float3) * h_numberOfPoints);
		CUDA_COPY_D2H(h_result.normals, d_normals, sizeof(float3) * h_numberOfPoints);
		CUDA_COPY_D2H(h_result.colors, d_colors, sizeof(float3) * h_numberOfPoints);
		if constexpr (!std::is_void_v<T>)
		{
			CUDA_COPY_D2H(h_result.properties, d_properties, sizeof(T) * h_numberOfPoints);
		}

		CUDA_SAFE_FREE(d_positions);
		CUDA_SAFE_FREE(d_normals);
		CUDA_SAFE_FREE(d_colors);
		if constexpr (!std::is_void_v<T>)
		{
			CUDA_SAFE_FREE(d_properties)
		}
		CUDA_SAFE_FREE(d_numberOfPoints);

		return h_result;
	}

	void MarchingCubes(DeviceHalfEdgeMesh<T>& mesh, float isoValue = 0.0f)
	{
		nvtxRangePushA("MarchingCubes");

		unsigned int maxPoints = info.h_numberOfOccupiedVoxels * 3;
		unsigned int maxFaces = info.h_numberOfOccupiedVoxels * 15;
		mesh.Terminate();
		mesh.Initialize(maxPoints, maxFaces);

		unsigned int* d_numberOfPoints = nullptr;
		unsigned int* d_numberOfFaces = nullptr;
		CUDA_MALLOC(&d_numberOfPoints, sizeof(unsigned int));
		CUDA_MALLOC(&d_numberOfFaces, sizeof(unsigned int));
		CUDA_MEMSET(d_numberOfPoints, 0, sizeof(unsigned int));
		CUDA_MEMSET(d_numberOfFaces, 0, sizeof(unsigned int));
		CUDA_SYNC();

		LaunchKernel(Kernel_SCVoxelHashMap_CreateZeroCrossingPoints<T>, info.h_numberOfOccupiedVoxels,
			info, mesh.positions, mesh.normals, mesh.colors, mesh.properties, d_numberOfPoints);
		CUDA_SYNC();

		LaunchKernel(Kernel_SCVoxelHashMap_MarchingCubes<T>, info.h_numberOfOccupiedVoxels,
			info, isoValue, mesh.faces, d_numberOfFaces);
		CUDA_SYNC();

		unsigned int h_numberOfPoints = 0, h_numberOfFaces = 0;
		CUDA_COPY_D2H(&h_numberOfPoints, d_numberOfPoints, sizeof(unsigned int));
		CUDA_COPY_D2H(&h_numberOfFaces, d_numberOfFaces, sizeof(unsigned int));
		CUDA_SYNC();

		mesh.numberOfPoints = h_numberOfPoints;
		mesh.numberOfFaces = h_numberOfFaces;

		mesh.BuildHalfEdges();

		//SCVoxelHashMap::SurfaceProjection_SDF(info, mesh);

		//mesh.LaplacianSmoothing(5, 1.0f, true);

		CUDA_FREE(d_numberOfPoints);
		CUDA_FREE(d_numberOfFaces);

		CUDA_SYNC();

		nvtxRangePop();
	}

	void SurfaceProjection_SDF(SCVoxelHashMapInfo<T> info, DeviceHalfEdgeMesh<T>& mesh, int maxIters = 5, float stepScale = 0.5f)
	{
		unsigned int numberOfPoints = mesh.numberOfPoints;
		float3* positions = mesh.positions;
		float3* normals = mesh.normals;

		LaunchKernel(Kernel_SCVoxelHashMap_SurfaceProjection_SDF<T>, numberOfPoints,
			info, positions, normals, numberOfPoints, maxIters, stepScale);
		CUDA_SYNC();
	}

#ifdef __CUDA_ARCH__
	__device__ static SCVoxel<T>* GetVoxel(SCVoxelHashMapInfo<T>& info, const int3& index)
	{
		VoxelKey key = IndexToVoxelKey(index);
		size_t hashIdx = VoxelKeyHash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t idx = (hashIdx + probe) % info.capacity;
			SCVoxelHashMapEntry<T>& entry = info.entries[idx];

			if (entry.key == key)
			{
				return &entry.voxel;
			}
			if (entry.key == EMPTY_KEY)
			{
				return nullptr;
			}
		}

		return nullptr;
	}

	__device__ static SCVoxel<T>* InsertVoxel(
		SCVoxelHashMapInfo<T>& info, const int3& index, float sdf, float3 normal, float3 color, T* property)
	{
		VoxelKey key = IndexToVoxelKey(index);
		size_t hashIdx = VoxelKeyHash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (hashIdx + probe) % info.capacity;
			SCVoxelHashMapEntry<T>* entry = &info.entries[slot];

			VoxelKey old = atomicCAS(reinterpret_cast<unsigned long long*>(&entry->key), EMPTY_KEY, key);

			if (old == EMPTY_KEY || old == key)
			{
				float prev =
					atomicCAS(reinterpret_cast<int*>(&entry->voxel.sdfSum), __float_as_int(FLT_MAX), __float_as_int(sdf));

				if (__int_as_float(prev) != FLT_MAX)
				{
					atomicAdd(&entry->voxel.sdfSum, sdf);
				}
				atomicAdd(&entry->voxel.normalSum.x, normal.x);
				atomicAdd(&entry->voxel.normalSum.y, normal.y);
				atomicAdd(&entry->voxel.normalSum.z, normal.z);

				atomicAdd(&entry->voxel.colorSum.x, color.x);
				atomicAdd(&entry->voxel.colorSum.y, color.y);
				atomicAdd(&entry->voxel.colorSum.z, color.z);

				unsigned int count = atomicAdd(&entry->voxel.count, 1u);
				if (count == 0)
				{
					unsigned int occIdx = atomicAdd(info.d_numberOfOccupiedVoxels, 1u);
					if (occIdx < info.h_occupiedCapacity)
						info.d_occupiedVoxelIndices[occIdx] = index;
				}

				entry->voxel.coordinate = index;
				entry->voxel.property = *property;

				return &entry->voxel;
			}
		}

		return nullptr;
	}

	__device__ static bool CheckZeroCrossing(SCVoxelHashMapInfo<T>& info,
		float osdf, const float3& op, const float3& on, const float3& oc,
		const int3& index, float3& outPosition, float3& outNormal, float3& outColor, T* outProperty)
	{
		auto voxel = SCVoxelHashMap<T>::GetVoxel(info, index);
		if (nullptr == voxel || 0 == voxel->count) return false;

		float nsdf = voxel->sdfSum / (float)max(1u, voxel->count);

		float ratio = osdf / (osdf - nsdf);

		if (0 > ratio || 1.f < ratio) return false;

		float3 np = IndexToPosition(index, info.voxelSize);
		float3 nn = voxel->normalSum / (float)max(1u, voxel->count);
		float3 nc = voxel->colorSum / (float)max(1u, voxel->count);
		
		outPosition = op + ratio * (np - op);;
		outNormal = on + ratio * (nn - on);
		outColor = oc + ratio * (nc - oc);
		if constexpr (!std::is_void_v<T>)
		{
			*outProperty = voxel->property;
		}

		return true;
	}

	__device__ static float GetSDFValue(SCVoxelHashMapInfo<T> info, float isoValue, const int3& voxelIndex)
	{
		auto voxel = SCVoxelHashMap<T>::GetVoxel(info, voxelIndex);
		if (!voxel || voxel->count == 0) return FLT_MAX;
		return voxel->sdfSum / (float)max(1u, voxel->count);
	}

	__device__ static unsigned int BuildCubeIndex(SCVoxelHashMapInfo<T> info, float isoValue, const int3& voxelIndex)
	{
		unsigned int cubeIndex = 0;

		const int3 cornerOffsets[8] =
		{
			make_int3(0, 0, 0), // 0
			make_int3(1, 0, 0), // 1
			make_int3(1, 1, 0), // 2
			make_int3(0, 1, 0), // 3
			make_int3(0, 0, 1), // 4
			make_int3(1, 0, 1), // 5
			make_int3(1, 1, 1), // 6
			make_int3(0, 1, 1)  // 7
		};

		for (int i = 0; i < 8; ++i)
		{
			int3 corner = voxelIndex + cornerOffsets[i];
			float sdf = GetSDFValue(info, isoValue, corner);
			if (sdf < isoValue)
			{
				cubeIndex |= (1u << i);
			}
		}
		return cubeIndex;
	}

	__device__ static unsigned int GetZeroCrossingIndex(
		SCVoxelHashMapInfo<T>& info,const int3& baseIndex, const SCVoxel<T>* baseVoxel, int edge)
	{
		switch (edge)
		{
		case 0:  // (0,0,0) - x
			return baseVoxel->zeroCrossingPointIndex.x;
		case 1:  // (1,0,0) - y
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(1, 0, 0));
			return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
		}
		case 2:  // (0,1,0) - x
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(0, 1, 0));
			return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
		}
		case 3:  // (0,0,0) - y
		{
			return baseVoxel->zeroCrossingPointIndex.y;
		}
		case 4:  // (0,0,1) - x
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(0, 0, 1));
			return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
		}
		case 5:  // (1,0,1) - y
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(1, 0, 1));
			return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
		}
		case 6:  // (0,1,1) - x
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(0, 1, 1));
			return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
		}
		case 7:  // (0,0,1) - y
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(0, 0, 1));
			return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
		}
		case 8:  // (0,0,0) - z
		{
			return baseVoxel->zeroCrossingPointIndex.z;
		}
		case 9:  // (1,0,0) - z
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(1, 0, 0));
			return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
		}
		case 10: // (1,1,0) - z
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(1, 1, 0));
			return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
		}
		case 11: // (0,1,0) - z
		{
			SCVoxel<T>* v = SCVoxelHashMap<T>::GetVoxel(info, baseIndex + make_int3(0, 1, 0));
			return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
		}
		default:
			return UINT32_MAX;
		}
	}

	__device__ static float SampleSDF(const SCVoxelHashMapInfo<T>& info, const float3& p)
	{
		int3 idx = PositionToIndex(p, info.voxelSize);
		auto v = SCVoxelHashMap<T>::GetVoxel((SCVoxelHashMapInfo<T>&)info, idx);
		if (!v || v->count == 0) return FLT_MAX;
		return v->sdfSum / (float)max(1u, v->count);
	}

	__device__ static float3 SampleSDFGradient(const SCVoxelHashMapInfo<T>& info, const float3& p)
	{
		// central difference
		const float h = info.voxelSize * 0.5f;
		float sdf_x1 = SampleSDF(info, p + make_float3(h, 0, 0));
		float sdf_x0 = SampleSDF(info, p - make_float3(h, 0, 0));
		float sdf_y1 = SampleSDF(info, p + make_float3(0, h, 0));
		float sdf_y0 = SampleSDF(info, p - make_float3(0, h, 0));
		float sdf_z1 = SampleSDF(info, p + make_float3(0, 0, h));
		float sdf_z0 = SampleSDF(info, p - make_float3(0, 0, h));
		float3 grad = make_float3(
			(sdf_x1 - sdf_x0) / (2.0f * h),
			(sdf_y1 - sdf_y0) / (2.0f * h),
			(sdf_z1 - sdf_z0) / (2.0f * h)
		);
		return normalize(grad);
	}

	__device__ static float TrilinearSampleSDF(const SCVoxelHashMapInfo<T>& info, const float3& p)
	{
		float3 ip = p / info.voxelSize;
		int3 base = make_int3(floorf(ip.x), floorf(ip.y), floorf(ip.z));
		float3 frac = ip - make_float3(base.x, base.y, base.z);

		float sdf = 0;
		float wsum = 0;
		for (int dz = 0; dz <= 1; ++dz)
			for (int dy = 0; dy <= 1; ++dy)
				for (int dx = 0; dx <= 1; ++dx)
				{
					int3 idx = base + make_int3(dx, dy, dz);
					float w = ((dx ? frac.x : 1 - frac.x) *
						(dy ? frac.y : 1 - frac.y) *
						(dz ? frac.z : 1 - frac.z));
					auto v = SCVoxelHashMap<T>::GetVoxel((SCVoxelHashMapInfo<T>&)info, idx);
					if (v && v->count > 0)
					{
						float s = v->sdfSum / (float)max(1u, v->count);
						if (fabsf(s) < FLT_MAX / 2) { sdf += w * s; wsum += w; }
					}
				}
		return wsum > 0 ? sdf / wsum : FLT_MAX;
	}

	__device__ static float3 TrilinearSampleSDFGradient(const SCVoxelHashMapInfo<T>& info, const float3& p)
	{
		const float h = info.voxelSize * 0.5f;
		float sdf_x1 = TrilinearSampleSDF(info, p + make_float3(h, 0, 0));
		float sdf_x0 = TrilinearSampleSDF(info, p - make_float3(h, 0, 0));
		float sdf_y1 = TrilinearSampleSDF(info, p + make_float3(0, h, 0));
		float sdf_y0 = TrilinearSampleSDF(info, p - make_float3(0, h, 0));
		float sdf_z1 = TrilinearSampleSDF(info, p + make_float3(0, 0, h));
		float sdf_z0 = TrilinearSampleSDF(info, p - make_float3(0, 0, h));
		float3 grad = make_float3(
			(sdf_x1 - sdf_x0) / (2.0f * h),
			(sdf_y1 - sdf_y0) / (2.0f * h),
			(sdf_z1 - sdf_z0) / (2.0f * h)
		);
		return normalize(grad);
	}
#endif
};
