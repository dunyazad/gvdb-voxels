#include <VoxelHashMap.cuh>

void VoxelHashMap::Initialize(float voxelSize, size_t capacity, unsigned int maxProbe)
{
	info.voxelSize = voxelSize;
	info.capacity = capacity;
	info.maxProbe = maxProbe;
	cudaMalloc(&info.entries, sizeof(VoxelHashEntry) * info.capacity);

	LaunchKernel(Kernel_ClearHashMap, info.capacity, info);

	cudaDeviceSynchronize();
}

void VoxelHashMap::Terminate()
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

void VoxelHashMap::CheckOccupiedIndicesLength(unsigned int numberOfVoxelsToOccupy)
{
	if (numberOfVoxelsToOccupy == 0) return;

	if (!info.d_occupiedVoxelIndices)
	{
		cudaMalloc(&info.d_occupiedVoxelIndices, sizeof(uint3) * numberOfVoxelsToOccupy);
		cudaMalloc(&info.d_numberOfOccupiedVoxels, sizeof(unsigned int));
		unsigned int zero = 0;
		cudaMemcpy(info.d_numberOfOccupiedVoxels, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		info.h_occupiedCapacity = numberOfVoxelsToOccupy;
	}
	else
	{
		cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		unsigned int required = info.h_numberOfOccupiedVoxels + numberOfVoxelsToOccupy;
		if (required > info.h_occupiedCapacity)
		{
			int3* d_new = nullptr;
			cudaMalloc(&d_new, sizeof(int3) * required);
			cudaMemcpy(d_new, info.d_occupiedVoxelIndices,
				sizeof(uint3) * info.h_numberOfOccupiedVoxels, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			cudaFree(info.d_occupiedVoxelIndices);
			info.d_occupiedVoxelIndices = d_new;
			info.h_occupiedCapacity = required;
		}
	}

	printf("Dense Grid numberOfOccupiedVoxels capacity: %u\n", info.h_occupiedCapacity);
}

void VoxelHashMap::Occupy(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints)
{
	CheckOccupiedIndicesLength(numberOfPoints);
	LaunchKernel(Kernel_OccupyVoxelHashMap, numberOfPoints, info, d_positions, d_normals, d_colors, numberOfPoints);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy(const DevicePointCloud& d_input)
{
	CheckOccupiedIndicesLength(d_input.numberOfPoints);
	LaunchKernel(Kernel_OccupyVoxelHashMap, d_input.numberOfPoints, info, d_input.positions, d_input.normals, d_input.colors, d_input.numberOfPoints);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy_SDF(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints, int offset)
{
	int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

	CheckOccupiedIndicesLength(numberOfPoints * count);
	LaunchKernel(Kernel_OccupySDF, numberOfPoints, info, d_positions, d_normals, d_colors, numberOfPoints, offset);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy_SDF(const DevicePointCloud& d_input, int offset)
{
	CUDA_TS(Occupy_SDF)
		int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

	CheckOccupiedIndicesLength(d_input.numberOfPoints * count);
	LaunchKernel(Kernel_OccupySDF, d_input.numberOfPoints, info, d_input.positions, d_input.normals, d_input.colors, d_input.numberOfPoints, offset);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
	CUDA_TE(Occupy_SDF)
}

HostPointCloud VoxelHashMap::Serialize()
{
	DevicePointCloud d_result;

	if (info.h_numberOfOccupiedVoxels == 0) return d_result;

	d_result.Intialize(info.h_numberOfOccupiedVoxels);

	LaunchKernel(Kernel_SerializeVoxelHashMap, info.h_numberOfOccupiedVoxels, info, d_result.positions, d_result.normals, d_result.colors);

	cudaDeviceSynchronize();

	HostPointCloud h_result(d_result);
	d_result.Terminate();

	return h_result;
}

HostPointCloud VoxelHashMap::Serialize_SDF()
{
	DevicePointCloud d_result;

	if (info.h_numberOfOccupiedVoxels == 0) return d_result;

	d_result.Intialize(info.h_numberOfOccupiedVoxels);

	LaunchKernel(Kernel_SerializeVoxelHashMap_SDF, info.h_numberOfOccupiedVoxels, info, d_result.positions, d_result.normals, d_result.colors);

	cudaDeviceSynchronize();

	HostPointCloud h_result(d_result);
	d_result.Terminate();

	return h_result;
}

void VoxelHashMap::SmoothSDF(float smoothingFactor, int iterations)
{
	for (int i = 0; i < iterations; ++i)
	{
		LaunchKernel(Kernel_SmoothSDF_VoxelHashMap, info.h_numberOfOccupiedVoxels, info, smoothingFactor);
		cudaDeviceSynchronize();
	}
}

void VoxelHashMap::FilterOppositeNormals()
{
	LaunchKernel(Kernel_FilterOppositeNormals, info.h_numberOfOccupiedVoxels, info, 0.8f);
	cudaDeviceSynchronize();
}

void VoxelHashMap::FilterByNormalGradient(float gradientThreshold, bool remove)
{
	//LaunchKernel(Kernel_FilterByNormalGradient, info.h_numberOfOccupiedVoxels, info, 1.5f);
	LaunchKernel(Kernel_FilterByNormalGradient, info.h_numberOfOccupiedVoxels, info, gradientThreshold, remove);
	cudaDeviceSynchronize();
}

__host__ __device__ inline uint64_t VoxelHashMap::expandBits(uint32_t v)
{
	uint64_t x = v & 0x1fffff; // 21 bits
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__host__ __device__ inline uint32_t VoxelHashMap::compactBits(uint64_t x)
{
	x &= 0x1249249249249249;
	x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
	x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
	x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
	x = (x ^ (x >> 16)) & 0x1f00000000ffff;
	x = (x ^ (x >> 32)) & 0x1fffff;
	return static_cast<uint32_t>(x);
}

__host__ __device__ inline VoxelKey VoxelHashMap::IndexToVoxelKey(const int3& coord)
{
	// To handle negative values
	const int OFFSET = 1 << 20; // 2^20 = 1048576
	return (expandBits(coord.z + OFFSET) << 2) |
		(expandBits(coord.y + OFFSET) << 1) |
		expandBits(coord.x + OFFSET);
}

__host__ __device__ inline int3 VoxelHashMap::VoxelKeyToIndex(VoxelKey key)
{
	const int OFFSET = 1 << 20;
	int x = static_cast<int>(compactBits(key));
	int y = static_cast<int>(compactBits(key >> 1));
	int z = static_cast<int>(compactBits(key >> 2));
	return make_int3(x - OFFSET, y - OFFSET, z - OFFSET);
}

__host__ inline int3 VoxelHashMap::PositionToIndex_host(float3 pos, float voxelSize)
{
	return make_int3(
		static_cast<int>(std::floor(pos.x / voxelSize)),
		static_cast<int>(std::floor(pos.y / voxelSize)),
		static_cast<int>(std::floor(pos.z / voxelSize)));
}

__device__ inline int3 VoxelHashMap::PositionToIndex_device(float3 pos, float voxelSize)
{
	return make_int3(
		__float2int_rd(pos.x / voxelSize),
		__float2int_rd(pos.y / voxelSize),
		__float2int_rd(pos.z / voxelSize));
}

__host__ __device__ inline int3 VoxelHashMap::PositionToIndex(float3 pos, float voxelSize)
{
#ifdef __CUDA_ARCH__
	return PositionToIndex_device(pos, voxelSize);
#else
	return PositionToIndex_host(pos, voxelSize);
#endif
}

__host__ __device__ inline float3 VoxelHashMap::IndexToPosition(int3 index, float voxelSize)
{
	return make_float3(
		(index.x + 0.5f) * voxelSize,
		(index.y + 0.5f) * voxelSize,
		(index.z + 0.5f) * voxelSize);
}

__host__ __device__ inline size_t VoxelHashMap::hash(VoxelKey key, size_t capacity)
{
	key ^= (key >> 33);
	key *= 0xff51afd7ed558ccd;
	key ^= (key >> 33);
	key *= 0xc4ceb9fe1a85ec53;
	key ^= (key >> 33);
	return static_cast<size_t>(key) % capacity;
}

__device__ bool VoxelHashMap::isZeroCrossing(VoxelHashMapInfo info, int3 index, float sdfCenter)
{
	const int3 dirs[6] = {
		make_int3(1, 0, 0), make_int3(-1, 0, 0),
		make_int3(0, 1, 0), make_int3(0, -1, 0),
		make_int3(0, 0, 1), make_int3(0, 0, -1),
	};

	for (int i = 0; i < 6; ++i)
	{
		int3 neighbor = index + dirs[i];
		VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
		size_t h = VoxelHashMap::hash(nkey, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			VoxelHashEntry entry = info.entries[slot];

			if (entry.key == nkey)
			{
				float sdfNeighbor = entry.voxel.sdfSum / max(1u, entry.voxel.count);
				if ((sdfCenter > 0 && sdfNeighbor < 0) || (sdfCenter < 0 && sdfNeighbor > 0))
				{
					return true;
				}
				break;
			}
		}
	}
	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_6_old(
	VoxelHashMapInfo info,
	int3 index,
	float sdfCenter,
	float3& outPosition,
	float3& outNormal,
	float3& outColor)
{
	const int3 dirs[6] = {
		make_int3(1, 0, 0), make_int3(-1, 0, 0),
		make_int3(0, 1, 0), make_int3(0, -1, 0),
		make_int3(0, 0, 1), make_int3(0, 0, -1),
	};

	float3 p1 = VoxelHashMap::IndexToPosition(index, info.voxelSize);
	Voxel voxel1;
	{
		VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
		size_t h = VoxelHashMap::hash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			if (info.entries[slot].key == key)
			{
				voxel1 = info.entries[slot].voxel;
				break;
			}
		}
	}

	for (int i = 0; i < 6; ++i)
	{
		int3 neighbor = make_int3(index.x + dirs[i].x, index.y + dirs[i].y, index.z + dirs[i].z);
		VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
		size_t h = VoxelHashMap::hash(nkey, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			VoxelHashEntry entry = info.entries[slot];

			if (entry.key == nkey)
			{
				float sdfNeighbor = entry.voxel.sdfSum / max(1u, entry.voxel.count);

				// 단방향 crossing 조건 (positive → negative)
				if (sdfCenter > 0.0f && sdfNeighbor < 0.0f)
				{
					float3 p2 = VoxelHashMap::IndexToPosition(neighbor, info.voxelSize);

					float alpha = sdfCenter / (sdfCenter - sdfNeighbor);  // [0,1]
					outPosition = p1 + alpha * (p2 - p1);
					outNormal = voxel1.normalSum / max(1u, voxel1.count);
					outColor = voxel1.colorSum / max(1u, voxel1.count);
					return true;
				}
				break;
			}
		}
	}

	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_6(
	VoxelHashMapInfo info,
	int3 index,
	float sdfCenter,
	float3& outPosition,
	float3& outNormal,
	float3& outColor)
{
	const int3 dirs[6] = {
		make_int3(1, 0, 0), make_int3(-1, 0, 0),
		make_int3(0, 1, 0), make_int3(0, -1, 0),
		make_int3(0, 0, 1), make_int3(0, 0, -1),
	};

	float3 p1 = VoxelHashMap::IndexToPosition(index, info.voxelSize);
	Voxel voxel1;
	{
		VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
		size_t h = VoxelHashMap::hash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			if (info.entries[slot].key == key)
			{
				voxel1 = info.entries[slot].voxel;
				break;
			}
		}
	}

	for (int i = 0; i < 6; ++i)
	{
		int3 neighbor = index + dirs[i];
		VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
		size_t h = VoxelHashMap::hash(nkey, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			VoxelHashEntry entry = info.entries[slot];

			if (entry.key == nkey)
			{
				float sdfNeighbor = entry.voxel.sdfSum / max(1u, entry.voxel.count);

				if (sdfCenter > 0.0f && sdfNeighbor < 0.0f)
				{
					float diff = sdfCenter - sdfNeighbor;
					if (fabsf(diff) < 0.01f)
						break; // very flat interface, likely noise

					float alpha = sdfCenter / diff;
					if (alpha < 0.01f || alpha > 0.99f)
						break; // avoid extrapolated/interpolated noise

					float3 p2 = VoxelHashMap::IndexToPosition(neighbor, info.voxelSize);
					outPosition = p1 + alpha * (p2 - p1);
					outNormal = voxel1.normalSum / max(1u, voxel1.count);
					outColor = voxel1.colorSum / max(1u, voxel1.count);
					return true;
				}
				break;
			}
		}
	}

	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_26_old(
	VoxelHashMapInfo info,
	int3 index,
	float sdfCenter,
	float3& outPosition,
	float3& outNormal,
	float3& outColor)
{
	float3 p1 = VoxelHashMap::IndexToPosition(index, info.voxelSize);
	Voxel voxel1;
	{
		VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
		size_t h = VoxelHashMap::hash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			if (info.entries[slot].key == key)
			{
				voxel1 = info.entries[slot].voxel;
				break;
			}
		}
	}

	// 26방향 오프셋 정의
	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue; // 중심은 건너뜀

				int3 neighbor = make_int3(index.x + dx, index.y + dy, index.z + dz);
				VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
				size_t h = VoxelHashMap::hash(nkey, info.capacity);

				for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
				{
					size_t slot = (h + probe) % info.capacity;
					VoxelHashEntry entry = info.entries[slot];

					if (entry.key == nkey)
					{
						float sdfNeighbor = entry.voxel.sdfSum / max(1u, entry.voxel.count);

						// 단방향 crossing: 양 → 음
						if (sdfCenter > 0.0f && sdfNeighbor < 0.0f)
						{
							float3 p2 = VoxelHashMap::IndexToPosition(neighbor, info.voxelSize);
							float alpha = sdfCenter / (sdfCenter - sdfNeighbor);
							outPosition = p1 + alpha * (p2 - p1);
							outNormal = voxel1.normalSum / max(1u, voxel1.count);
							outColor = voxel1.colorSum / max(1u, voxel1.count);
							return true;
						}
						break;
					}
				}
			}
		}
	}

	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_26(
	VoxelHashMapInfo info,
	int3 index,
	float sdfCenter,
	float3& outPosition,
	float3& outNormal,
	float3& outColor)
{
	float3 p1 = VoxelHashMap::IndexToPosition(index, info.voxelSize);
	Voxel voxel1;
	{
		VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
		size_t h = VoxelHashMap::hash(key, info.capacity);

		for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
		{
			size_t slot = (h + probe) % info.capacity;
			if (info.entries[slot].key == key)
			{
				voxel1 = info.entries[slot].voxel;
				break;
			}
		}
	}

	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;

				int3 neighbor = index + make_int3(dx, dy, dz);
				VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
				size_t h = VoxelHashMap::hash(nkey, info.capacity);

				for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
				{
					size_t slot = (h + probe) % info.capacity;
					VoxelHashEntry entry = info.entries[slot];

					if (entry.key == nkey)
					{
						float sdfNeighbor = entry.voxel.sdfSum / max(1u, entry.voxel.count);

						//if (sdfCenter > 0.0f && sdfNeighbor < 0.0f)
						if ((sdfCenter > 0.0f && sdfNeighbor < 0.0f) || (sdfCenter < 0.0f && sdfNeighbor > 0.0f))
						{
							float diff = sdfCenter - sdfNeighbor;
							if (fabsf(diff) < 0.01f)
								break;

							float alpha = sdfCenter / diff;
							if (alpha < 0.01f || alpha > 0.99f)
								break;

							float3 p2 = VoxelHashMap::IndexToPosition(neighbor, info.voxelSize);
							outPosition = p1 + alpha * (p2 - p1);
							outNormal = voxel1.normalSum / max(1u, voxel1.count);
							outColor = voxel1.colorSum / max(1u, voxel1.count);
							return true;
						}
						break;
					}
				}
			}
		}
	}

	return false;
}

__global__ void Kernel_ClearHashMap(VoxelHashMapInfo info)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= info.capacity) return;

	info.entries[idx].key = EMPTY_KEY;
	info.entries[idx].voxel = {};
}

__global__ void Kernel_OccupyVoxelHashMap(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors,
	unsigned int numberOfPoints)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numberOfPoints) return;

	float3 pos = positions[tid];
	int3 index = VoxelHashMap::PositionToIndex(pos, info.voxelSize);
	VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
	float3 n = normals ? normals[tid] : make_float3(0, 0, 0);
	float3 c = colors ? colors[tid] : make_float3(0, 0, 0);

	size_t h = VoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		VoxelHashEntry* entry = &info.entries[slot];

		VoxelKey old = atomicCAS(reinterpret_cast<unsigned long long*>(&entry->key),
			EMPTY_KEY, key);

		if (old == EMPTY_KEY || old == key)
		{
			atomicAdd(&entry->voxel.normalSum.x, n.x);
			atomicAdd(&entry->voxel.normalSum.y, n.y);
			atomicAdd(&entry->voxel.normalSum.z, n.z);

			atomicAdd(&entry->voxel.colorSum.x, c.x);
			atomicAdd(&entry->voxel.colorSum.y, c.y);
			atomicAdd(&entry->voxel.colorSum.z, c.z);

			unsigned int count = atomicAdd(&entry->voxel.count, 1u);
			if (count == 0)
			{
				unsigned int occindex = atomicAdd(info.d_numberOfOccupiedVoxels, 1);
				if (occindex < info.h_occupiedCapacity)
					info.d_occupiedVoxelIndices[occindex] = index;
			}

			entry->voxel.coordinate = index;

			return;
		}
	}
}

__global__ void Kernel_OccupySDF(
	VoxelHashMapInfo info,
	float3* positions,
	float3* normals,
	float3* colors,
	unsigned int numberOfPoints,
	int offset)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numberOfPoints) return;

	//printf("tid : %d\n", tid);

	float3 p = positions[tid];
	float3 n = normals[tid];
	float3 c = colors ? colors[tid] : make_float3(0, 0, 0);

	int3 baseIndex = VoxelHashMap::PositionToIndex(p, info.voxelSize);

	for (int dz = -offset; dz <= offset; ++dz)
	{
		for (int dy = -offset; dy <= offset; ++dy)
		{
			for (int dx = -offset; dx <= offset; ++dx)
			{
				int3 index = make_int3(baseIndex.x + dx, baseIndex.y + dy, baseIndex.z + dz);
				VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
				float3 center = VoxelHashMap::IndexToPosition(index, info.voxelSize);

				float3 dir = center - p;
				float dist = length(dir);
				if (dist < 1e-6f) continue;

				float sign = (dot(n, dir) >= 0.0f) ? 1.0f : -1.0f;
				float sdf = dist * sign;

				size_t h = VoxelHashMap::hash(key, info.capacity);

				for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
				{
					size_t slot = (h + probe) % info.capacity;
					VoxelHashEntry* entry = &info.entries[slot];

					VoxelKey old = atomicCAS(reinterpret_cast<VoxelKey*>(&entry->key), EMPTY_KEY, key);

					if (old == EMPTY_KEY || old == key)
					{
						atomicAdd(&entry->voxel.sdfSum, sdf);
						atomicAdd(&entry->voxel.normalSum.x, n.x);
						atomicAdd(&entry->voxel.normalSum.y, n.y);
						atomicAdd(&entry->voxel.normalSum.z, n.z);

						atomicAdd(&entry->voxel.colorSum.x, c.x);
						atomicAdd(&entry->voxel.colorSum.y, c.y);
						atomicAdd(&entry->voxel.colorSum.z, c.z);

						// Count 0 → 1일 때만 index 등록
						if (atomicCAS(&entry->voxel.count, 0u, 1u) == 0u)
						{
							unsigned int occindex = atomicAdd(info.d_numberOfOccupiedVoxels, 1);
							if (occindex < info.h_occupiedCapacity)
								info.d_occupiedVoxelIndices[occindex] = index;
						}
						else
						{
							// 이후 count 증가
							atomicAdd(&entry->voxel.count, 1u);
						}

						entry->voxel.coordinate = index;
						break;
					}
				}
			}
		}
	}
}

__global__ void Kernel_SerializeVoxelHashMap(
	VoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	VoxelKey voxelKey = VoxelHashMap::IndexToVoxelKey(voxelIndex);
	size_t h = VoxelHashMap::hash(voxelKey, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		VoxelHashEntry& entry = info.entries[slot];

		if (entry.key == voxelKey)
		{
			Voxel& voxel = entry.voxel;

			if (voxel.count == 0)
			{
				d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
				d_normals[tid] = make_float3(0, 0, 0);
				d_colors[tid] = make_float3(0, 0, 0);
				return;
			}
			else
			{
				auto x = voxelIndex.x * info.voxelSize;
				auto y = voxelIndex.y * info.voxelSize;
				auto z = voxelIndex.z * info.voxelSize;

				d_positions[tid] = make_float3(x, y, z);
				d_normals[tid] = voxel.normalSum / (float)voxel.count;
				d_colors[tid] = voxel.colorSum / (float)voxel.count;
				return;
			}
		}
	}

	// Not found fallback
	d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	d_normals[tid] = make_float3(0, 0, 0);
	d_colors[tid] = make_float3(0, 0, 0);
}

__global__ void Kernel_SerializeVoxelHashMap_SDF(
	VoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	VoxelKey voxelKey = VoxelHashMap::IndexToVoxelKey(voxelIndex);
	size_t h = VoxelHashMap::hash(voxelKey, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		VoxelHashEntry& entry = info.entries[slot];

		if (entry.key == voxelKey)
		{
			Voxel& voxel = entry.voxel;

			if (voxel.count == 0)
			{
				d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
				d_normals[tid] = make_float3(0, 0, 0);
				d_colors[tid] = make_float3(0, 0, 0);
				return;
			}

			float sdf = voxel.sdfSum / (float)voxel.count;

			float3 pos, normal, color;
			bool valid = VoxelHashMap::computeInterpolatedSurfacePoint_26(info, voxelIndex, sdf, pos, normal, color);

			if (!valid)
			{
				d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
				d_normals[tid] = make_float3(0, 0, 0);
				d_colors[tid] = make_float3(0, 0, 0);
				return;
			}

			d_positions[tid] = pos;
			d_normals[tid] = normal;
			d_colors[tid] = color;
			return;
		}
	}

	// Not found fallback
	d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	d_normals[tid] = make_float3(0, 0, 0);
	d_colors[tid] = make_float3(0, 0, 0);
}

__global__ void Kernel_SmoothSDF_VoxelHashMap(
	VoxelHashMapInfo info, float smoothingFactor) // 예: 1.0f 평균, 0.5f low-pass
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
	size_t h = VoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		VoxelHashEntry& centerEntry = info.entries[slot];

		if (centerEntry.key == key)
		{
			Voxel& voxel = centerEntry.voxel;
			if (voxel.count == 0) return;

			float sdfCenter = voxel.sdfSum / (float)voxel.count;
			float sdfSum = sdfCenter;
			int sdfCount = 1;

			const int3 dirs[6] = {
				make_int3(1, 0, 0), make_int3(-1, 0, 0),
				make_int3(0, 1, 0), make_int3(0, -1, 0),
				make_int3(0, 0, 1), make_int3(0, 0, -1),
			};

			for (int i = 0; i < 6; ++i)
			{
				int3 neighbor = index + dirs[i];
				VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(neighbor);
				size_t hn = VoxelHashMap::hash(nkey, info.capacity);

				for (unsigned int p = 0; p < info.maxProbe; ++p)
				{
					size_t slotn = (hn + p) % info.capacity;
					VoxelHashEntry& nEntry = info.entries[slotn];

					if (nEntry.key == nkey && nEntry.voxel.count > 0)
					{
						float sdfNeighbor = nEntry.voxel.sdfSum / (float)nEntry.voxel.count;
						sdfSum += sdfNeighbor;
						++sdfCount;
						break;
					}
				}
			}

			float sdfSmoothed = (1.0f - smoothingFactor) * sdfCenter + smoothingFactor * (sdfSum / sdfCount);
			centerEntry.voxel.sdfSum = sdfSmoothed * voxel.count; // 보존: sum 형태로 저장
			return;
		}
	}
}

__global__ void Kernel_FilterOppositeNormals(VoxelHashMapInfo info, float thresholdDotCos) // 예: 0.2f
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
	size_t h = VoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		Voxel& voxel = info.entries[slot].voxel;
		if (info.entries[slot].key != key || voxel.count == 0) continue;

		float3 n0 = normalize(voxel.normalSum / voxel.count);
		float3 nAvg = make_float3(0, 0, 0);
		int neighborCount = 0;

		for (int dz = -1; dz <= 1; ++dz)
			for (int dy = -1; dy <= 1; ++dy)
				for (int dx = -1; dx <= 1; ++dx)
				{
					if (dx == 0 && dy == 0 && dz == 0) continue;
					int3 neighbor = index + make_int3(dx, dy, dz);
					VoxelKey nk = VoxelHashMap::IndexToVoxelKey(neighbor);
					size_t hn = VoxelHashMap::hash(nk, info.capacity);

					for (unsigned int p = 0; p < info.maxProbe; ++p)
					{
						size_t slotn = (hn + p) % info.capacity;
						VoxelHashEntry& e = info.entries[slotn];
						if (e.key == nk && e.voxel.count > 0)
						{
							nAvg += e.voxel.normalSum;
							neighborCount += e.voxel.count;
							break;
						}
					}
				}

		if (neighborCount < 3) return;

		float3 nAvgNorm = normalize(nAvg / (float)neighborCount);
		float dotProduct = dot(n0, nAvgNorm);

		if (dotProduct < thresholdDotCos)
		{
			//voxel.count = 0;
			//voxel.sdfSum = 0;
			//voxel.normalSum = make_float3(0, 0, 0);
			//voxel.colorSum = make_float3(0, 0, 0);

			voxel.colorSum = make_float3((float)voxel.count, 0, 0);
		}

		return;
	}
}

__global__ void Kernel_FilterByNormalGradient(
	VoxelHashMapInfo info,
	float gradientThreshold,
	bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	VoxelKey key = VoxelHashMap::IndexToVoxelKey(index);
	size_t h = VoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		Voxel& voxel = info.entries[slot].voxel;
		if (info.entries[slot].key != key || voxel.count == 0) continue;

		float3 n_center = voxel.normalSum / voxel.count;

		float3 dnx = make_float3(0, 0, 0);
		float3 dny = make_float3(0, 0, 0);
		float3 dnz = make_float3(0, 0, 0);

		auto get_normal = [&](int3 offset) -> float3
		{
			int3 nidx = index + offset;
			VoxelKey nkey = VoxelHashMap::IndexToVoxelKey(nidx);
			size_t nh = VoxelHashMap::hash(nkey, info.capacity);

			for (unsigned int p = 0; p < info.maxProbe; ++p)
			{
				size_t nslot = (nh + p) % info.capacity;
				VoxelHashEntry& e = info.entries[nslot];
				if (e.key == nkey && e.voxel.count > 0)
					return e.voxel.normalSum / e.voxel.count;
			}
			return n_center; // fallback: self
		};

		float3 n_posx = get_normal(make_int3(1, 0, 0));
		float3 n_negx = get_normal(make_int3(-1, 0, 0));
		float3 n_posy = get_normal(make_int3(0, 1, 0));
		float3 n_negy = get_normal(make_int3(0, -1, 0));
		float3 n_posz = get_normal(make_int3(0, 0, 1));
		float3 n_negz = get_normal(make_int3(0, 0, -1));

		dnx = 0.5f * (n_posx - n_negx);
		dny = 0.5f * (n_posy - n_negy);
		dnz = 0.5f * (n_posz - n_negz);

		float gradMag = length(dnx) + length(dny) + length(dnz);

		if (gradMag > gradientThreshold)
		{
			if (remove)
			{
				voxel.count = 0;
				voxel.normalSum = make_float3(0, 0, 0);
				voxel.colorSum = make_float3(0, 0, 0);
				voxel.sdfSum = 0.0f;
			}
			else
			{
				voxel.colorSum = make_float3((float)voxel.count, 0, 0);
			}
		}

		return;
	}
}
