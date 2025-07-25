#include <SCVoxelHashMap.cuh>

void SCVoxelHashMap::Initialize(float voxelSize, size_t capacity, unsigned int maxProbe)
{
	info.voxelSize = voxelSize;
	info.capacity = capacity;
	info.maxProbe = maxProbe;
	cudaMalloc(&info.entries, sizeof(SCVoxelHashEntry) * info.capacity);

	LaunchKernel(Kernel_SCVoxelHashMap_Clear, (unsigned int)info.capacity, info);

	CUDA_SYNC();
}

void SCVoxelHashMap::Terminate()
{
	if (nullptr != info.entries)
	{
		cudaFree(info.entries);
	}
	info.entries = nullptr;

	if (info.d_numberOfOccupiedVoxels) cudaFree(info.d_numberOfOccupiedVoxels);
	if (info.d_occupiedVoxelIndices) cudaFree(info.d_occupiedVoxelIndices);

	info.d_numberOfOccupiedVoxels = nullptr;
	info.d_occupiedVoxelIndices = nullptr;

	info.h_numberOfOccupiedVoxels = 0;
	info.h_occupiedCapacity = 0;
}

void SCVoxelHashMap::CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy)
{
	if (numberOfVoxelsToOccupy == 0) return;

	if (!info.d_occupiedVoxelIndices)
	{
		cudaMalloc(&info.d_occupiedVoxelIndices, sizeof(uint3) * numberOfVoxelsToOccupy);
		cudaMalloc(&info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
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
			cudaMalloc(&d_new, sizeof(int3) * required);
			CUDA_COPY_D2D(d_new, info.d_occupiedVoxelIndices, sizeof(uint3) * info.h_numberOfOccupiedVoxels);
			CUDA_SYNC();
			cudaFree(info.d_occupiedVoxelIndices);
			info.d_occupiedVoxelIndices = d_new;
			info.h_occupiedCapacity = required;
		}
	}

	printf("SCVoxelHashMap numberOfOccupiedVoxels capacity: %u\n", info.h_occupiedCapacity);
}

void SCVoxelHashMap::Occupy(const DevicePointCloud& d_input, int offset)
{
	CUDA_TS(SCVoxelHashMap_Occupy);

	if (1 > offset) offset = 1;
	int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

	CheckOccupiedIndicesLength(d_input.numberOfPoints * count);
	LaunchKernel(Kernel_SCVoxelHashMap_Occupy, d_input.numberOfPoints, info, d_input.positions, d_input.normals, d_input.colors, d_input.numberOfPoints, offset);
	CUDA_COPY_D2H(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
	CUDA_SYNC();

	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
	CUDA_TE(SCVoxelHashMap_Occupy);
}

HostPointCloud SCVoxelHashMap::Serialize()
{
	HostPointCloud h_result;
	if (info.h_numberOfOccupiedVoxels == 0) return h_result;

	float3* d_positions = nullptr;
	float3* d_normals = nullptr;
	float3* d_colors = nullptr;
	unsigned int* d_numberOfPoints = nullptr;

	cudaMalloc(&d_positions, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
	cudaMalloc(&d_normals, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
	cudaMalloc(&d_colors, sizeof(float3) * info.h_numberOfOccupiedVoxels * 3);
	cudaMalloc(&d_numberOfPoints, sizeof(unsigned int));

	LaunchKernel(Kernel_SCVoxelHashMap_CreateZeroCrossingPoints, info.h_numberOfOccupiedVoxels,
		info, d_positions, d_normals, d_colors, d_numberOfPoints);
	CUDA_SYNC();

	unsigned int h_numberOfPoints = 0;
	CUDA_COPY_D2H(&h_numberOfPoints, d_numberOfPoints, sizeof(unsigned int));
	CUDA_SYNC();

	h_result.Intialize(h_numberOfPoints);
	CUDA_COPY_D2H(h_result.positions, d_positions, sizeof(float3) * h_numberOfPoints);
	CUDA_COPY_D2H(h_result.normals, d_normals, sizeof(float3) * h_numberOfPoints);
	CUDA_COPY_D2H(h_result.colors, d_colors, sizeof(float3) * h_numberOfPoints);

	cudaFree(d_positions);
	cudaFree(d_normals);
	cudaFree(d_colors);
	cudaFree(d_numberOfPoints);

	return h_result;
}

void SCVoxelHashMap::MarchingCubes(DeviceHalfEdgeMesh& mesh, float isoValue)
{
	unsigned int maxPoints = info.h_numberOfOccupiedVoxels * 3;
	unsigned int maxFaces = info.h_numberOfOccupiedVoxels * 15;
	mesh.Terminate();
	mesh.Initialize(maxPoints, maxFaces);

	unsigned int* d_numberOfPoints = nullptr;
	unsigned int* d_numberOfFaces = nullptr;
	cudaMalloc(&d_numberOfPoints, sizeof(unsigned int));
	cudaMalloc(&d_numberOfFaces, sizeof(unsigned int));
	cudaMemset(d_numberOfPoints, 0, sizeof(unsigned int));
	cudaMemset(d_numberOfFaces, 0, sizeof(unsigned int));
	CUDA_SYNC();

	LaunchKernel(Kernel_SCVoxelHashMap_CreateZeroCrossingPoints, info.h_numberOfOccupiedVoxels,
		info, mesh.positions, mesh.normals, mesh.colors, d_numberOfPoints);
	CUDA_SYNC();

	LaunchKernel(Kernel_SCVoxelHashMap_MarchingCubes, info.h_numberOfOccupiedVoxels,
		info, isoValue, mesh.faces, d_numberOfFaces);
	CUDA_SYNC();

	unsigned int h_numberOfPoints = 0, h_numberOfFaces = 0;
	CUDA_COPY_D2H(&h_numberOfPoints, d_numberOfPoints, sizeof(unsigned int));
	CUDA_COPY_D2H(&h_numberOfFaces, d_numberOfFaces, sizeof(unsigned int));
	CUDA_SYNC();

	mesh.numberOfPoints = h_numberOfPoints;
	mesh.numberOfFaces = h_numberOfFaces;

	//std::vector<uint3> h_faces(mesh.numberOfFaces);
	//cudaMemcpy(h_faces.data(), mesh.faces, sizeof(uint3) * mesh.numberOfFaces, cudaMemcpyDeviceToHost);
	//for (unsigned int i = 0; i < mesh.numberOfFaces; ++i)
	//{
	//	if (h_faces[i].x >= mesh.numberOfPoints || h_faces[i].y >= mesh.numberOfPoints || h_faces[i].z >= mesh.numberOfPoints)
	//	{
	//		printf("[ERROR] Invalid device face index at face[%u]: %u, %u, %u\n",
	//			i, h_faces[i].x, h_faces[i].y, h_faces[i].z);
	//	}
	//}

	mesh.BuildHalfEdges();

	std::vector<HalfEdge> h_halfEdges(mesh.numberOfFaces * 3);
	cudaMemcpy(h_halfEdges.data(), mesh.halfEdges, sizeof(HalfEdge) * mesh.numberOfFaces * 3, cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < mesh.numberOfFaces * 3; ++i)
	{
		if (h_halfEdges[i].vertexIndex >= mesh.numberOfPoints)
		{
			printf("[ERROR] Invalid halfEdge[%u].vertexIndex = %u (numberOfPoints=%u)\n",
				i, h_halfEdges[i].vertexIndex, mesh.numberOfPoints);
		}
	}


	//mesh.LaplacianSmoothing(3, 1.0f);
	//mesh.LaplacianSmoothingNRing(2, 0.15f, 1);

	cudaFree(d_numberOfPoints);
	cudaFree(d_numberOfFaces);
}

__host__ __device__ uint64_t SCVoxelHashMap::expandBits(uint32_t v)
{
	uint64_t x = v & 0x1fffff; // 21 bits
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__host__ __device__ uint32_t SCVoxelHashMap::compactBits(uint64_t x)
{
	x &= 0x1249249249249249;
	x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
	x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
	x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
	x = (x ^ (x >> 16)) & 0x1f00000000ffff;
	x = (x ^ (x >> 32)) & 0x1fffff;
	return static_cast<uint32_t>(x);
}

__host__ __device__ size_t SCVoxelHashMap::hash(SCVoxelKey key, size_t capacity)
{
	key ^= (key >> 33);
	key *= 0xff51afd7ed558ccd;
	key ^= (key >> 33);
	key *= 0xc4ceb9fe1a85ec53;
	key ^= (key >> 33);
	return static_cast<size_t>(key) % capacity;
}

__host__ __device__ SCVoxelKey SCVoxelHashMap::IndexToVoxelKey(const int3& coord)
{
	// To handle negative values
	const int OFFSET = 1 << 20; // 2^20 = 1048576
	return (expandBits(coord.z + OFFSET) << 2) |
		(expandBits(coord.y + OFFSET) << 1) |
		expandBits(coord.x + OFFSET);
}

__host__ __device__ int3 SCVoxelHashMap::VoxelKeyToIndex(SCVoxelKey key)
{
	const int OFFSET = 1 << 20;
	int x = static_cast<int>(compactBits(key));
	int y = static_cast<int>(compactBits(key >> 1));
	int z = static_cast<int>(compactBits(key >> 2));
	return make_int3(x - OFFSET, y - OFFSET, z - OFFSET);
}

__host__ __device__ int3 SCVoxelHashMap::PositionToIndex(const float3& pos, float voxelSize)
{
#ifdef __CUDA_ARCH__
	return make_int3(
		__float2int_rd(pos.x / voxelSize),
		__float2int_rd(pos.y / voxelSize),
		__float2int_rd(pos.z / voxelSize));
#else
	return make_int3(
		static_cast<int>(std::floor(pos.x / voxelSize)),
		static_cast<int>(std::floor(pos.y / voxelSize)),
		static_cast<int>(std::floor(pos.z / voxelSize)));
#endif
}

__host__ __device__ float3 SCVoxelHashMap::IndexToPosition(const int3& index, float voxelSize)
{
	return make_float3(
		(float)index.x * voxelSize,
		(float)index.y * voxelSize,
		(float)index.z * voxelSize);
}

__device__ SCVoxel* SCVoxelHashMap::GetVoxel(SCVoxelHashMapInfo& info, const int3& index)
{
	SCVoxelKey key = IndexToVoxelKey(index);
	size_t hashIdx = hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t idx = (hashIdx + probe) % info.capacity;
		SCVoxelHashEntry& entry = info.entries[idx];

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

__device__ SCVoxel* SCVoxelHashMap::InsertVoxel(
	SCVoxelHashMapInfo& info, const int3& index, float sdf, float3 normal, float3 color)
{
	SCVoxelKey key = SCVoxelHashMap::IndexToVoxelKey(index);
	size_t hashIdx = SCVoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (hashIdx + probe) % info.capacity;
		SCVoxelHashEntry* entry = &info.entries[slot];

		SCVoxelKey old = atomicCAS(reinterpret_cast<unsigned long long*>(&entry->key), EMPTY_KEY, key);

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
	
			return &entry->voxel;
		}
	}

	return nullptr;
}










__global__ void Kernel_SCVoxelHashMap_Clear(SCVoxelHashMapInfo info)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= info.capacity) return;

	info.entries[idx].key = EMPTY_KEY;
	info.entries[idx].voxel = {};
	info.entries[idx].voxel.zeroCrossingPointIndex = make_uint3(UINT32_MAX);
}

__global__ void Kernel_SCVoxelHashMap_Occupy(
	SCVoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors,
	unsigned int numberOfPoints,
	int offset)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numberOfPoints) return;

	float3 p = positions[tid];
	float3 n = normals[tid];
	float3 c = colors ? colors[tid] : make_float3(0, 0, 0);

	int3 baseIndex = SCVoxelHashMap::PositionToIndex(p, info.voxelSize);

	for (int dz = -offset; dz <= offset; ++dz)
	{
		for (int dy = -offset; dy <= offset; ++dy)
		{
			for (int dx = -offset; dx <= offset; ++dx)
			{
				int3 index = make_int3(baseIndex.x + dx, baseIndex.y + dy, baseIndex.z + dz);
				float3 center = SCVoxelHashMap::IndexToPosition(index, info.voxelSize);

				float3 dir = center - p;
				float dist = length(dir);
				if ((dist < 1e-6f) || (info.voxelSize * (float)offset < dist)) continue;

				float sign = (dot(n, dir) >= 0.0f) ? 1.0f : -1.0f;
				float sdf = dist * sign;

				SCVoxelHashMap::InsertVoxel(info, index, sdf, n, c);
			}
		}
	}
}

__device__ bool SCVoxelHashMap::CheckZeroCrossing(SCVoxelHashMapInfo& info,
	float osdf, const float3& op, const float3& on, const float3& oc,
	const int3& index, float3& outPosition, float3& outNormal, float3& outColor)
{
	auto voxel = SCVoxelHashMap::GetVoxel(info, index);
	if (nullptr == voxel || 0 == voxel->count) return false;

	float nsdf = voxel->sdfSum / (float)max(1u, voxel->count);

	float ratio = osdf / (osdf - nsdf);

	if (0 > ratio || 1.f < ratio) return false;

	float3 np = SCVoxelHashMap::IndexToPosition(index, info.voxelSize);
	float3 nn = voxel->normalSum / (float)max(1u, voxel->count);
	float3 nc = voxel->colorSum / (float)max(1u, voxel->count);

	outPosition = op + ratio * (np - op);;
	outNormal = on + ratio * (nn - on);
	outColor = oc + ratio * (nc - oc);

	return true;
}

__global__ void Kernel_SCVoxelHashMap_CreateZeroCrossingPoints(
	SCVoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors,
	unsigned int* d_numberOfPoints)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = SCVoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	float sdf = voxel->sdfSum / (float)max(1u, voxel->count);

	float3 voxelPosition = SCVoxelHashMap::IndexToPosition(voxelIndex, info.voxelSize);
	float3 voxelNormal = voxel->normalSum / (float)max(1u, voxel->count);
	float3 voxelColor = voxel->colorSum / (float)max(1u, voxel->count);

	// X Axis
	{
		auto neighborIndex = voxelIndex;
		neighborIndex.x += 1;
		float3 position;
		float3 normal;
		float3 color;
		if (true == SCVoxelHashMap::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
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
		if (true == SCVoxelHashMap::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
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
		if (true == SCVoxelHashMap::CheckZeroCrossing(
			info, sdf, voxelPosition, voxelNormal, voxelColor, neighborIndex,
			position, normal, color))
		{
			auto index = atomicAdd(d_numberOfPoints, 1u);
			d_positions[index] = position;
			d_normals[index] = normal;
			d_colors[index] = color;
			voxel->zeroCrossingPointIndex.z = index;
		}
	}
}

__device__ float SCVoxelHashMap::GetSDFValue(SCVoxelHashMapInfo info, float isoValue, const int3& voxelIndex)
{
	auto voxel = SCVoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return FLT_MAX;
	return voxel->sdfSum / (float)max(1u, voxel->count);
}

__device__ unsigned int SCVoxelHashMap::BuildCubeIndex(SCVoxelHashMapInfo info, float isoValue, const int3& voxelIndex)
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

__device__ unsigned int SCVoxelHashMap::GetZeroCrossingIndex(
	SCVoxelHashMapInfo& info, const int3& baseIndex, const SCVoxel* baseVoxel, int edge)
{
	switch (edge)
	{
	case 0:  // (0,0,0) - x
		return baseVoxel->zeroCrossingPointIndex.x;
	case 1:  // (1,0,0) - y
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(1, 0, 0));
		return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
	}
	case 2:  // (0,1,0) - x
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(0, 1, 0));
		return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
	}
	case 3:  // (0,0,0) - y
	{
		return baseVoxel->zeroCrossingPointIndex.y;
	}
	case 4:  // (0,0,1) - x
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(0, 0, 1));
		return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
	}
	case 5:  // (1,0,1) - y
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(1, 0, 1));
		return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
	}
	case 6:  // (0,1,1) - x
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(0, 1, 1));
		return v ? v->zeroCrossingPointIndex.x : UINT32_MAX;
	}
	case 7:  // (0,0,1) - y
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(0, 0, 1));
		return v ? v->zeroCrossingPointIndex.y : UINT32_MAX;
	}
	case 8:  // (0,0,0) - z
	{
		return baseVoxel->zeroCrossingPointIndex.z;
	}
	case 9:  // (1,0,0) - z
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(1, 0, 0));
		return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
	}
	case 10: // (1,1,0) - z
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(1, 1, 0));
		return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
	}
	case 11: // (0,1,0) - z
	{
		SCVoxel* v = SCVoxelHashMap::GetVoxel(info, baseIndex + make_int3(0, 1, 0));
		return v ? v->zeroCrossingPointIndex.z : UINT32_MAX;
	}
	default:
		return UINT32_MAX;
	}
}

__global__ void Kernel_SCVoxelHashMap_MarchingCubes(
	SCVoxelHashMapInfo info, float isoValue, uint3* d_faces, unsigned int* d_numberOfFaces)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = SCVoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	auto cubeIndex = SCVoxelHashMap::BuildCubeIndex(info, isoValue, voxelIndex);

	for (int i = 0; MC_TRI_TABLE[cubeIndex][i] != -1; i += 3)
	{
		int e0 = MC_TRI_TABLE[cubeIndex][i + 0];
		int e1 = MC_TRI_TABLE[cubeIndex][i + 1];
		int e2 = MC_TRI_TABLE[cubeIndex][i + 2];

		unsigned int v0 = SCVoxelHashMap::GetZeroCrossingIndex(info, voxelIndex, voxel, e0);
		unsigned int v1 = SCVoxelHashMap::GetZeroCrossingIndex(info, voxelIndex, voxel, e1);
		unsigned int v2 = SCVoxelHashMap::GetZeroCrossingIndex(info, voxelIndex, voxel, e2);

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
