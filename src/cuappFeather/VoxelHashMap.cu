#include <VoxelHashMap.cuh>

void VoxelHashMap::Initialize(float voxelSize, size_t capacity, unsigned int maxProbe)
{
	info.voxelSize = voxelSize;
	info.capacity = capacity;
	info.maxProbe = maxProbe;
	cudaMalloc(&info.entries, sizeof(VoxelHashEntry) * info.capacity);

	LaunchKernel(Kernel_VoxelHashMap_Clear, (unsigned int)info.capacity, info);

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
	LaunchKernel(Kernel_VoxelHashMap_Occupy, numberOfPoints, info, d_positions, d_normals, d_colors, numberOfPoints);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy(const DevicePointCloud& d_input)
{
	CheckOccupiedIndicesLength(d_input.numberOfPoints);
	LaunchKernel(Kernel_VoxelHashMap_Occupy, d_input.numberOfPoints, info, d_input.positions, d_input.normals, d_input.colors, d_input.numberOfPoints);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy_SDF(float3* d_positions, float3* d_normals, float3* d_colors, unsigned int numberOfPoints, int offset)
{
	int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

	CheckOccupiedIndicesLength(numberOfPoints * count);
	LaunchKernel(Kernel_VoxelHashMap_Occupy_SDF, numberOfPoints, info, d_positions, d_normals, d_colors, numberOfPoints, offset);
	cudaMemcpy(&info.h_numberOfOccupiedVoxels, info.d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Number of occupied voxels : %u\n", info.h_numberOfOccupiedVoxels);
}

void VoxelHashMap::Occupy_SDF(const DevicePointCloud& d_input, int offset)
{
	CUDA_TS(Occupy_SDF)
		int count = (offset * 2 + 1) * (offset * 2 + 1) * (offset * 2 + 1);

	CheckOccupiedIndicesLength(d_input.numberOfPoints * count);
	LaunchKernel(Kernel_VoxelHashMap_Occupy_SDF, d_input.numberOfPoints, info, d_input.positions, d_input.normals, d_input.colors, d_input.numberOfPoints, offset);
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

	LaunchKernel(Kernel_VoxelHashMap_Serialize, info.h_numberOfOccupiedVoxels, info, d_result.positions, d_result.normals, d_result.colors);

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

	LaunchKernel(Kernel_VoxelHashMap_Serialize_SDF, info.h_numberOfOccupiedVoxels, info, d_result.positions, d_result.normals, d_result.colors);

	cudaDeviceSynchronize();

	HostPointCloud h_result(d_result);
	d_result.Terminate();

	return h_result;
}

HostPointCloud VoxelHashMap::Serialize_SDF_Tidy()
{
	DevicePointCloud d_result;

	if (info.h_numberOfOccupiedVoxels == 0) return d_result;

	d_result.Intialize(info.h_numberOfOccupiedVoxels);

	LaunchKernel(Kernel_VoxelHashMap_Serialize_SDF_Tidy, info.h_numberOfOccupiedVoxels, info, d_result.positions, d_result.normals, d_result.colors);

	cudaDeviceSynchronize();

	HostPointCloud h_result(d_result);
	d_result.Terminate();

	return h_result;
}

void VoxelHashMap::Dilation(int iterations, int step)
{
	for (int iter = 0; iter < iterations; ++iter)
	{
		unsigned int prevNumOccupied = info.h_numberOfOccupiedVoxels;

		int maxNew = prevNumOccupied * ((2 * step + 1) * (2 * step + 1) * (2 * step + 1) - 1);
		int3* d_newOccupiedIndices = nullptr;
		unsigned int* d_newOccupiedCount = nullptr;
		cudaMalloc(&d_newOccupiedIndices, sizeof(int3) * maxNew);
		cudaMalloc(&d_newOccupiedCount, sizeof(unsigned int));
		cudaMemset(d_newOccupiedCount, 0, sizeof(unsigned int));

		LaunchKernel(Kernel_VoxelHashMap_Dilation, prevNumOccupied, info,
			d_newOccupiedIndices, d_newOccupiedCount, step);

		// d_newOccupiedCount → host 복사
		unsigned int h_newOccupied = 0;
		cudaMemcpy(&h_newOccupied, d_newOccupiedCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		if (h_newOccupied > 0)
		{
			// 기존 occupiedIndices 확장
			CheckOccupiedIndicesLength(info.h_numberOfOccupiedVoxels + h_newOccupied);
			cudaMemcpy(
				info.d_occupiedVoxelIndices + info.h_numberOfOccupiedVoxels,
				d_newOccupiedIndices,
				sizeof(int3) * h_newOccupied, cudaMemcpyDeviceToDevice
			);
			info.h_numberOfOccupiedVoxels += h_newOccupied;
			cudaMemcpy(info.d_numberOfOccupiedVoxels, &info.h_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyHostToDevice);
		}

		cudaFree(d_newOccupiedIndices);
		cudaFree(d_newOccupiedCount);

		printf("Dilation iteration %d: prevNumOccupied: %u, new: %u, total: %u\n",
			iter, prevNumOccupied, h_newOccupied, info.h_numberOfOccupiedVoxels);
	}
}

bool VoxelHashMap::MarchingCubes(std::vector<float3>& outVertices, std::vector<float3>& outNormals, std::vector<float3>& outColors, std::vector<uint3>& outTriangles)
{
	if (info.h_numberOfOccupiedVoxels == 0)
		return false;

	const int maxTriangles = (info.h_numberOfOccupiedVoxels) * 5;
	const int maxVertices = maxTriangles * 3;

	float3* d_vertices = nullptr;
	float3* d_normals = nullptr;
	float3* d_colors = nullptr;
	uint3* d_indices = nullptr;
	uint64_t* d_edgeSlotKeys = nullptr;
	int* d_edgeSlotValues = nullptr;
	unsigned int* d_numVertices = nullptr;
	unsigned int* d_numTriangles = nullptr;

	int edgeSlotCapacity = maxVertices * 3;

	cudaMalloc(&d_vertices, sizeof(float3) * maxVertices);
	cudaMalloc(&d_normals, sizeof(float3) * maxVertices);
	cudaMalloc(&d_colors, sizeof(float3) * maxVertices);
	cudaMalloc(&d_indices, sizeof(uint3) * maxTriangles);
	cudaMalloc(&d_edgeSlotKeys, sizeof(uint64_t) * edgeSlotCapacity);
	cudaMalloc(&d_edgeSlotValues, sizeof(int) * edgeSlotCapacity);
	cudaMalloc(&d_numVertices, sizeof(unsigned int));
	cudaMalloc(&d_numTriangles, sizeof(unsigned int));

	cudaMemset(d_numVertices, 0, sizeof(unsigned int));
	cudaMemset(d_numTriangles, 0, sizeof(unsigned int));
#define INVALID_EDGE_KEY 0xffffffffffffffffull
	cudaMemset(d_edgeSlotKeys, 0xff, sizeof(uint64_t) * edgeSlotCapacity);
	cudaMemset(d_edgeSlotValues, -1, sizeof(int) * edgeSlotCapacity);

	LaunchKernel(Kernel_VoxelHashMap_MarchingCubes, info.h_numberOfOccupiedVoxels,
		info,
		d_vertices,
		d_normals,
		d_colors,
		d_indices,
		d_edgeSlotKeys,
		d_edgeSlotValues,
		edgeSlotCapacity,
		d_numVertices,
		d_numTriangles);

	cudaDeviceSynchronize();

	unsigned int h_numVertices = 0, h_numTriangles = 0;
	cudaMemcpy(&h_numVertices, d_numVertices, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_numTriangles, d_numTriangles, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	printf("h_numVertices : %d, h_numTriangles : %d\n", h_numVertices, h_numTriangles);

	outVertices.resize(h_numVertices);
	outNormals.resize(h_numVertices);
	outColors.resize(h_numVertices);
	outTriangles.resize(h_numTriangles);

	cudaMemcpy(outVertices.data(), d_vertices, sizeof(float3) * h_numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(outNormals.data(), d_normals, sizeof(float3) * h_numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(outColors.data(), d_colors, sizeof(float3) * h_numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(outTriangles.data(), d_indices, sizeof(uint3) * h_numTriangles, cudaMemcpyDeviceToHost);

	cudaFree(d_vertices);
	cudaFree(d_normals);
	cudaFree(d_colors);
	cudaFree(d_indices);
	cudaFree(d_edgeSlotKeys);
	cudaFree(d_edgeSlotValues);
	cudaFree(d_numVertices);
	cudaFree(d_numTriangles);

	return true;
}

void VoxelHashMap::FindOverlap(int step, bool remove)
{
	LaunchKernel(Kernel_VoxelHashMap_FindOverlap, info.h_numberOfOccupiedVoxels, info, step, remove);
	cudaDeviceSynchronize();
}

void VoxelHashMap::SmoothSDF(float smoothingFactor, int iterations)
{
	for (int i = 0; i < iterations; ++i)
	{
		LaunchKernel(Kernel_VoxelHashMap_SmoothSDF, info.h_numberOfOccupiedVoxels, info, smoothingFactor);
		cudaDeviceSynchronize();
	}
}

void VoxelHashMap::FilterOppositeNormals()
{
	LaunchKernel(Kernel_VoxelHashMap_FilterOppositeNormals, info.h_numberOfOccupiedVoxels, info, 0.8f);
	cudaDeviceSynchronize();
}

void VoxelHashMap::FilterByNormalGradient(float gradientThreshold, bool remove)
{
	//LaunchKernel(Kernel_VoxelHashMap_FilterByNormalGradient, info.h_numberOfOccupiedVoxels, info, 1.5f);
	LaunchKernel(Kernel_VoxelHashMap_FilterByNormalGradient, info.h_numberOfOccupiedVoxels, info, gradientThreshold, remove);
	cudaDeviceSynchronize();
}

void VoxelHashMap::FilterByNormalGradientWithOffset(int offset, float gradientThreshold, bool remove)
{
	//LaunchKernel(Kernel_VoxelHashMap_FilterByNormalGradient, info.h_numberOfOccupiedVoxels, info, 1.5f);
	LaunchKernel(Kernel_VoxelHashMap_FilterByNormalGradientWithOffset, info.h_numberOfOccupiedVoxels, info, offset, gradientThreshold, remove);
	cudaDeviceSynchronize();
}

void VoxelHashMap::FilterBySDFGradient(float sdfThreshold, bool remove)
{
	LaunchKernel(Kernel_VoxelHashMap_FilterBySDFGradient, info.h_numberOfOccupiedVoxels, info, sdfThreshold, remove);
	cudaDeviceSynchronize();
}

void VoxelHashMap::FilterBySDFGradientWithOffset(int offset, float sdfThreshold, bool remove)
{
	LaunchKernel(Kernel_VoxelHashMap_FilterBySDFGradient_26, info.h_numberOfOccupiedVoxels, info, offset, sdfThreshold, remove);
	cudaDeviceSynchronize();
}

__host__ __device__ uint64_t VoxelHashMap::expandBits(uint32_t v)
{
	uint64_t x = v & 0x1fffff; // 21 bits
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__host__ __device__ uint32_t VoxelHashMap::compactBits(uint64_t x)
{
	x &= 0x1249249249249249;
	x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
	x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
	x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
	x = (x ^ (x >> 16)) & 0x1f00000000ffff;
	x = (x ^ (x >> 32)) & 0x1fffff;
	return static_cast<uint32_t>(x);
}

__host__ __device__ VoxelKey VoxelHashMap::IndexToVoxelKey(const int3& coord)
{
	// To handle negative values
	const int OFFSET = 1 << 20; // 2^20 = 1048576
	return (expandBits(coord.z + OFFSET) << 2) |
		(expandBits(coord.y + OFFSET) << 1) |
		expandBits(coord.x + OFFSET);
}

__host__ __device__ int3 VoxelHashMap::VoxelKeyToIndex(VoxelKey key)
{
	const int OFFSET = 1 << 20;
	int x = static_cast<int>(compactBits(key));
	int y = static_cast<int>(compactBits(key >> 1));
	int z = static_cast<int>(compactBits(key >> 2));
	return make_int3(x - OFFSET, y - OFFSET, z - OFFSET);
}

__host__ int3 VoxelHashMap::PositionToIndex_host(float3 pos, float voxelSize)
{
	return make_int3(
		static_cast<int>(std::floor(pos.x / voxelSize)),
		static_cast<int>(std::floor(pos.y / voxelSize)),
		static_cast<int>(std::floor(pos.z / voxelSize)));
}

__device__ int3 VoxelHashMap::PositionToIndex_device(float3 pos, float voxelSize)
{
	return make_int3(
		__float2int_rd(pos.x / voxelSize),
		__float2int_rd(pos.y / voxelSize),
		__float2int_rd(pos.z / voxelSize));
}

__host__ __device__ int3 VoxelHashMap::PositionToIndex(float3 pos, float voxelSize)
{
#ifdef __CUDA_ARCH__
	return PositionToIndex_device(pos, voxelSize);
#else
	return PositionToIndex_host(pos, voxelSize);
#endif
}

__host__ __device__ float3 VoxelHashMap::IndexToPosition(int3 index, float voxelSize)
{
	return make_float3(
		(index.x + 0.5f) * voxelSize,
		(index.y + 0.5f) * voxelSize,
		(index.z + 0.5f) * voxelSize);
}

__host__ __device__ size_t VoxelHashMap::hash(VoxelKey key, size_t capacity)
{
	key ^= (key >> 33);
	key *= 0xff51afd7ed558ccd;
	key ^= (key >> 33);
	key *= 0xc4ceb9fe1a85ec53;
	key ^= (key >> 33);
	return static_cast<size_t>(key) % capacity;
}

__device__ Voxel* VoxelHashMap::GetVoxel(VoxelHashMapInfo& info, const int3& index)
{
	VoxelKey key = IndexToVoxelKey(index);
	size_t hashIdx = hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t idx = (hashIdx + probe) % info.capacity;
		VoxelHashEntry& entry = info.entries[idx];

		if (entry.key == key)
		{
			return &entry.voxel;
			//if (0 != entry.voxel.count) return &entry.voxel;
			//else return nullptr;
		}
		if (entry.key == EMPTY_KEY)
		{
			return nullptr;
		}
	}

	return nullptr;
}

__device__ bool VoxelHashMap::isZeroCrossing(VoxelHashMapInfo& info, int3 index, float sdfCenter)
{
	const int3 dirs[6] = {
		make_int3(1, 0, 0), make_int3(-1, 0, 0),
		make_int3(0, 1, 0), make_int3(0, -1, 0),
		make_int3(0, 0, 1), make_int3(0, 0, -1),
	};

	for (int i = 0; i < 6; ++i)
	{
		int3 neighbor = index + dirs[i];
		auto neighborVoxel = GetVoxel(info, neighbor);
		if (nullptr != neighborVoxel)
		{
			float sdfNeighbor = neighborVoxel->sdfSum / (float)max(1u, neighborVoxel->count);
			if ((sdfCenter > 0 && sdfNeighbor < 0) || (sdfCenter < 0 && sdfNeighbor > 0))
			{
				return true;
			}
			break;
		}
	}

	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_6(
	VoxelHashMapInfo& info,
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

	Voxel* voxel1 = GetVoxel(info, index);
	if (!voxel1 || voxel1->count == 0) return false;

	float3 p1 = IndexToPosition(index, info.voxelSize);

	for (int i = 0; i < 6; ++i)
	{
		int3 neighborIndex = index + dirs[i];
		auto neighborVoxel = GetVoxel(info, neighborIndex);
		if (!neighborVoxel || neighborVoxel->count == 0) return false;
		{
			float sdfNeighbor = neighborVoxel->sdfSum / (float)max(1u, neighborVoxel->count);

			if (sdfCenter > 0.0f && sdfNeighbor < 0.0f)
			{
				float diff = sdfCenter - sdfNeighbor;
				if (fabsf(diff) < 0.01f)
					break; // very flat interface, likely noise

				float alpha = sdfCenter / diff;
				if (alpha < 0.01f || alpha > 0.99f)
					break; // avoid extrapolated/interpolated noise

				float3 p2 = IndexToPosition(neighborIndex, info.voxelSize);
				outPosition = p1 + alpha * (p2 - p1);
				outNormal = voxel1->normalSum / (float)max(1u, voxel1->count);
				outColor = voxel1->colorSum / (float)max(1u, voxel1->count);
				return true;
			}
			break;
		}
	}

	return false;
}

__device__ bool VoxelHashMap::computeInterpolatedSurfacePoint_26(
	VoxelHashMapInfo& info,
	int3 index,
	float sdfCenter,
	float3& outPosition,
	float3& outNormal,
	float3& outColor)
{
	float3 p1 = IndexToPosition(index, info.voxelSize);
	Voxel* voxel1 = GetVoxel(info, index);
	if (!voxel1 || voxel1->count == 0) return false;

	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				int3 neighbor = index + make_int3(dx, dy, dz);
				Voxel* voxel2 = GetVoxel(info, neighbor);

				if (!voxel2 || voxel2->count == 0) continue;

				float sdfNeighbor = voxel2->sdfSum / (float)max(1u, voxel2->count);

				if ((sdfCenter > 0.0f && sdfNeighbor < 0.0f) || (sdfCenter < 0.0f && sdfNeighbor > 0.0f))
				{
					float diff = sdfCenter - sdfNeighbor;
					if (fabsf(diff) < 0.01f) continue;

					float alpha = sdfCenter / diff;
					if (alpha < 0.01f || alpha > 0.99f) continue;

					float3 p2 = IndexToPosition(neighbor, info.voxelSize);
					outPosition = p1 + alpha * (p2 - p1);
					outNormal = voxel1->normalSum / (float)max(1u, voxel1->count);
					outColor = voxel1->colorSum / (float)max(1u, voxel1->count);
					return true;
				}
			}
		}
	}

	return false;
}

__device__ bool VoxelHashMap::isSimpleVoxel(VoxelHashMapInfo& info, int3 index)
{
	const int dx[6] = { 1, -1, 0, 0, 0, 0 };
	const int dy[6] = { 0, 0, 1, -1, 0, 0 };
	const int dz[6] = { 0, 0, 0, 0, 1, -1 };

	int count = 0;

	for (int i = 0; i < 6; ++i)
	{
		int3 neighborIndex = make_int3(index.x + dx[i], index.y + dy[i], index.z + dz[i]);
		auto neighbor = GetVoxel(info, neighborIndex);
		if (nullptr != neighbor && 0 < neighbor->count)
		{
			++count;
		}
	}

	return (count >= 1); // can be made stricter
}

__device__ uint64_t VoxelHashMap::EncodeEdgeKey(int3 v0, int3 v1)
{
	if ((v1.x < v0.x) || (v1.x == v0.x && v1.y < v0.y) || (v1.x == v0.x && v1.y == v0.y && v1.z < v0.z))
	{
		int3 tmp = v0; v0 = v1; v1 = tmp;
	}
	return (uint64_t(v0.x & 0x1FFFFF) << 42) |
		(uint64_t(v0.y & 0x1FFFFF) << 21) |
		(uint64_t(v0.z & 0x1FFFFF));
}

__device__ float3 VoxelHashMap::Interpolate(float3 p1, float3 p2, float sdf1, float sdf2)
{
	float alpha = sdf1 / (sdf1 - sdf2 + 1e-6f);
	alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
	return p1 + alpha * (p2 - p1);
}

__device__ bool VoxelHashMap::FillMissingCorner(
	VoxelHashMapInfo& info, int3* cornerIndices, float* sdf, float3* normal, float3* color, bool* valid)
{
	bool anyFilled = false;
	for (int i = 0; i < 8; ++i)
	{
		if (!valid[i])
		{
			// 주변 코너 중 valid인 것의 평균
			float sumSdf = 0.0f;
			float3 sumNormal = make_float3(0, 0, 0);
			float3 sumColor = make_float3(0, 0, 0);
			int cnt = 0;
			for (int j = 0; j < 8; ++j)
			{
				if (valid[j])
				{
					sumSdf += sdf[j];
					sumNormal += normal[j];
					sumColor += color[j];
					cnt++;
				}
			}
			if (cnt > 0)
			{
				sdf[i] = sumSdf / cnt;
				normal[i] = sumNormal / (float)cnt;
				color[i] = sumColor / (float)cnt;
				valid[i] = true;
				anyFilled = true;
			}
		}
	}
	return anyFilled;
}

__device__ void VoxelHashMap::FillHardMissingCorner(float* sdf, float3* normal, float3* color, bool* valid)
{
	// 1. 찾기: valid 코너
	int firstValid = -1;
	for (int i = 0; i < 8; ++i) if (valid[i]) { firstValid = i; break; }
	if (firstValid < 0) return; // 전부 비었으면 return

	// 2. 할당
	for (int i = 0; i < 8; ++i)
	{
		if (!valid[i])
		{
			sdf[i] = sdf[firstValid];
			normal[i] = normal[firstValid];
			color[i] = color[firstValid];
			valid[i] = true;
		}
	}
}

__device__ bool VoxelHashMap::GetOrFillVoxelCorner(
	VoxelHashMapInfo& info, int3 corner, float& outSDF, float3& outNormal, float3& outColor)
{
	Voxel* voxel = VoxelHashMap::GetVoxel(info, corner);
	if (voxel && voxel->count > 0)
	{
		outSDF = voxel->sdfSum / (float)max(1u, voxel->count);
		outNormal = voxel->normalSum / (float)max(1u, voxel->count);
		outColor = voxel->colorSum / (float)max(1u, voxel->count);
		return true;
	}

	// 26방향 이웃에서 채우기
	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue; // 자기 자신 제외

				int3 n = corner + make_int3(dx, dy, dz);
				Voxel* nvoxel = VoxelHashMap::GetVoxel(info, n);
				if (nvoxel && nvoxel->count > 0)
				{
					outSDF = nvoxel->sdfSum / (float)max(1u, nvoxel->count);
					outNormal = nvoxel->normalSum / (float)max(1u, nvoxel->count);
					outColor = nvoxel->colorSum / (float)max(1u, nvoxel->count);
					return true;
				}
			}
		}
	}
	outSDF = 0.0f;
	outNormal = make_float3(0, 0, 0);
	outColor = make_float3(0, 0, 0);
	return false;
}

__device__ void VoxelHashMap::FillMissingCornersWithNearest(
	int3 baseIndex,
	float* sdf, float3* normal, float3* color, bool* cornerValid)
{
	// 8개 중 하나라도 있으면 없는 코너를 가장 가까운 코너로 채움
	for (int i = 0; i < 8; ++i)
	{
		if (!cornerValid[i])
		{
			float minDist = 1e9f;
			int bestJ = -1;
			int3 ci = baseIndex + MC_CORNERS[i];
			for (int j = 0; j < 8; ++j)
			{
				if (!cornerValid[j]) continue;
				int3 cj = baseIndex + MC_CORNERS[j];
				float d = length(make_float3(
					float(ci.x - cj.x),
					float(ci.y - cj.y),
					float(ci.z - cj.z)));
				if (d < minDist)
				{
					minDist = d;
					bestJ = j;
				}
			}
			if (bestJ >= 0)
			{
				//float bias = 1e-3f;
				float bias = 0.25f;
				sdf[i] = sdf[bestJ] + ((sdf[bestJ] >= 0) ? bias : -bias);
				normal[i] = normal[bestJ];
				color[i] = color[bestJ];
				cornerValid[i] = true;
			}
		}
	}
}

__global__ void Kernel_VoxelHashMap_Clear(VoxelHashMapInfo info)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= info.capacity) return;

	info.entries[idx].key = EMPTY_KEY;
	info.entries[idx].voxel = {};
}

__global__ void Kernel_VoxelHashMap_Occupy(
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

	float3 center = VoxelHashMap::IndexToPosition(index, info.voxelSize);

	float3 dir = center - pos;
	float dist = length(dir);

	float sign = (dot(n, dir) >= 0.0f) ? 1.0f : -1.0f;
	float sdf = dist * sign;

	size_t h = VoxelHashMap::hash(key, info.capacity);

	for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
	{
		size_t slot = (h + probe) % info.capacity;
		VoxelHashEntry* entry = &info.entries[slot];

		VoxelKey old = atomicCAS(reinterpret_cast<unsigned long long*>(&entry->key), EMPTY_KEY, key);

		if (old == EMPTY_KEY || old == key)
		{
			atomicAdd(&entry->voxel.sdfSum, sdf);
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

__global__ void Kernel_VoxelHashMap_Occupy_SDF(
	VoxelHashMapInfo info,
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
						float prev = atomicCAS(reinterpret_cast<int*>(&entry->voxel.sdfSum),
							__float_as_int(FLT_MAX),
							__float_as_int(sdf));

						if (__int_as_float(prev) != FLT_MAX)
						{
							atomicAdd(&entry->voxel.sdfSum, sdf);
						}
						atomicAdd(&entry->voxel.normalSum.x, n.x);
						atomicAdd(&entry->voxel.normalSum.y, n.y);
						atomicAdd(&entry->voxel.normalSum.z, n.z);

						atomicAdd(&entry->voxel.colorSum.x, c.x);
						atomicAdd(&entry->voxel.colorSum.y, c.y);
						atomicAdd(&entry->voxel.colorSum.z, c.z);

						if (atomicCAS(&entry->voxel.count, 0u, 1u) == 0u)
						{
							unsigned int occindex = atomicAdd(info.d_numberOfOccupiedVoxels, 1);
							if (occindex < info.h_occupiedCapacity)
								info.d_occupiedVoxelIndices[occindex] = index;
						}
						else
						{
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

__global__ void Kernel_VoxelHashMap_Serialize(
	VoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = VoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	float sdf = voxel->sdfSum / (float)max(1u, voxel->count);

	auto x = voxelIndex.x * info.voxelSize;
	auto y = voxelIndex.y * info.voxelSize;
	auto z = voxelIndex.z * info.voxelSize;

	d_positions[tid] = make_float3(x, y, z);
	d_normals[tid] = voxel->normalSum / (float)max(1u, voxel->count);
	d_colors[tid] = voxel->colorSum / (float)max(1u, voxel->count);

	if (voxel->colorSum == make_float3(0.0f, 0.0f, 0.0f))
	{
		printf("[%d, %d, %d] count : %d\n", voxel->coordinate.x, voxel->coordinate.y, voxel->coordinate.z, voxel->count);
	}

	if (0 < sdf)
	{
		d_colors[tid] = make_float3(1.0f, 0.0f, 0.0f);
	}
	else if (0 > sdf)
	{
		d_colors[tid] = make_float3(0.0f, 0.0f, 1.0f);
	}
}

__global__ void Kernel_VoxelHashMap_Serialize_SDF(
	VoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = VoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	float sdf = voxel->sdfSum / (float)max(1u, voxel->count);

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
}

__global__ void Kernel_VoxelHashMap_Serialize_SDF_Tidy(
	VoxelHashMapInfo info,
	float3* d_positions,
	float3* d_normals,
	float3* d_colors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	auto voxel = VoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	float sdf = voxel->sdfSum / (float)max(1u, voxel->count);

	bool zeroCrossing = false;
	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;
				int3 neighborIndex = voxelIndex + make_int3(dx, dy, dz);
				Voxel* nvoxel = VoxelHashMap::GetVoxel(info, neighborIndex);
				if (nvoxel && nvoxel->count > 0)
				{
					float nsdf = nvoxel->sdfSum / (float)max(1u, nvoxel->count);
					if ((0 > sdf && 0 < nsdf) || (0 < sdf && 0 > nsdf))
					{
						auto diff = sdf + nsdf;
						if (info.voxelSize * 0.5f > diff)
						{
							zeroCrossing = true;
						}
					}
				}
			}
		}
	}

	if (!zeroCrossing)
	{
		d_positions[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		d_normals[tid] = make_float3(0, 0, 0);
		d_colors[tid] = make_float3(0, 0, 0);
		return;
	}

	d_positions[tid] = VoxelHashMap::IndexToPosition(voxelIndex, info.voxelSize);
	d_normals[tid] = voxel->normalSum / (float)max(1u, voxel->count);
	d_colors[tid] = voxel->colorSum / (float)max(1u, voxel->count);
}

__global__ void Kernel_VoxelHashMap_FindOverlap(VoxelHashMapInfo info, int step, bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 voxelIndex = info.d_occupiedVoxelIndices[tid];
	float3 voxelPosition = VoxelHashMap::IndexToPosition(voxelIndex, info.voxelSize);
	Voxel* voxel = VoxelHashMap::GetVoxel(info, voxelIndex);
	if (!voxel || voxel->count == 0) return;

	for (int i = 0; i < step; i++)
	{
		float3 upperPosition = voxelPosition + normalize(voxel->normalSum / (float)max(1u, voxel->count)) * info.voxelSize * (float)(i + 1);
		auto upperIndex = VoxelHashMap::PositionToIndex(upperPosition, info.voxelSize);
		if (voxelIndex == upperIndex) continue;

		auto upperVoxel = VoxelHashMap::GetVoxel(info, upperIndex);
		if (!upperVoxel || upperVoxel->count == 0) continue;

		if (false == remove)
		{
			upperVoxel->colorSum = make_float3(1.0f, 0.0f, 0.0f) * (float)upperVoxel->count;
		}
		else
		{
			//upperVoxel->coordinate = make_int3(INT32_MAX, INT32_MAX, INT32_MAX);
			upperVoxel->normalSum = make_float3(0, 0, 0);
			upperVoxel->colorSum = make_float3(0, 0, 0);
			upperVoxel->sdfSum = FLT_MAX;
			upperVoxel->count = 0;
		}
	}
}

__global__ void Kernel_VoxelHashMap_SmoothSDF(
	VoxelHashMapInfo info,
	float smoothingFactor)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	float sdfCenter = voxel->sdfSum / (float)max(1u, voxel->count);;
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
		Voxel* neighborVoxel = VoxelHashMap::GetVoxel(info, neighbor);
		if (neighborVoxel && neighborVoxel->count > 0)
		{
			float sdfNeighbor = neighborVoxel->sdfSum / (float)max(1u, neighborVoxel->count);
			sdfSum += sdfNeighbor;
			++sdfCount;
		}
	}

	float sdfSmoothed = (1.0f - smoothingFactor) * sdfCenter + smoothingFactor * (sdfSum / sdfCount);
	voxel->sdfSum = sdfSmoothed * voxel->count;
}

__global__ void Kernel_VoxelHashMap_FilterOppositeNormals(VoxelHashMapInfo info, float thresholdDotCos)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	float3 n0 = normalize(voxel->normalSum / (float)max(1u, voxel->count));
	float3 nAvg = make_float3(0, 0, 0);
	int neighborCount = 0;

	for (int dz = -1; dz <= 1; ++dz)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;
				int3 neighbor = index + make_int3(dx, dy, dz);
				Voxel* nvoxel = VoxelHashMap::GetVoxel(info, neighbor);
				if (nvoxel && nvoxel->count > 0)
				{
					nAvg += nvoxel->normalSum;
					neighborCount += nvoxel->count;
				}
			}
		}
	}

	if (neighborCount < 3) return;

	float3 nAvgNorm = normalize(nAvg / (float)max(1u, neighborCount));
	float dotProduct = dot(n0, nAvgNorm);

	if (dotProduct < thresholdDotCos)
	{
		voxel->colorSum = make_float3((float)voxel->count, 0, 0);
	}
}

__global__ void Kernel_VoxelHashMap_FilterByNormalGradient(
	VoxelHashMapInfo info,
	float gradientThreshold,
	bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	float3 n_center = voxel->normalSum / (float)max(1u, voxel->count);

	auto get_normal = [&](int3 offset) -> float3
	{
		int3 nidx = index + offset;
		Voxel* nvoxel = VoxelHashMap::GetVoxel(info, nidx);
		if (nvoxel && nvoxel->count > 0)
			return nvoxel->normalSum / (float)max(1u, nvoxel->count);
		return n_center; // fallback
	};

	float3 dnx = 0.5f * (get_normal(make_int3(1, 0, 0)) - get_normal(make_int3(-1, 0, 0)));
	float3 dny = 0.5f * (get_normal(make_int3(0, 1, 0)) - get_normal(make_int3(0, -1, 0)));
	float3 dnz = 0.5f * (get_normal(make_int3(0, 0, 1)) - get_normal(make_int3(0, 0, -1)));

	float gradMag = length(dnx + dny + dnz);

	if (gradMag > gradientThreshold)
	{
		if (remove)
		{
			voxel->normalSum = make_float3(0, 0, 0);
			voxel->colorSum = make_float3(0, 0, 0);
			voxel->sdfSum = FLT_MAX;
			voxel->count = 0;
		}
		else
		{
			voxel->colorSum = make_float3((float)voxel->count, 0, 0);
		}
	}
}

__global__ void Kernel_VoxelHashMap_FilterByNormalGradientWithOffset(
	VoxelHashMapInfo info,
	int offset,
	float gradientThreshold,
	bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	float3 n_center = voxel->normalSum / (float)max(1u, voxel->count);

	float maxGradient = 0.0f;
	int neighborCount = 0;

	for (int dz = -offset; dz <= offset; ++dz)
	{
		for (int dy = -offset; dy <= offset; ++dy)
		{
			for (int dx = -offset; dx <= offset; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				int3 nidx = index + make_int3(dx, dy, dz);
				Voxel* nvoxel = VoxelHashMap::GetVoxel(info, nidx);
				if (!nvoxel || nvoxel->count == 0) continue;

				float3 n_neighbor = nvoxel->normalSum / (float)max(1u, nvoxel->count);
				float3 diff = n_neighbor - n_center;
				float gradMag = length(diff);

				maxGradient = fmaxf(maxGradient, gradMag);
				++neighborCount;
			}
		}
	}

	if (maxGradient > gradientThreshold)
	{
		if (remove)
		{
			voxel->normalSum = make_float3(0, 0, 0);
			voxel->colorSum = make_float3(0, 0, 0);
			voxel->sdfSum = FLT_MAX;
			voxel->count = 0;
		}
		else
		{
			voxel->colorSum = make_float3((float)voxel->count, 0, 0);
		}
	}
}

__global__ void Kernel_VoxelHashMap_FilterBySDFGradient(
	VoxelHashMapInfo info,
	float gradientThreshold,
	bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	auto get_sdf = [&](int3 offset) -> float
	{
		int3 nidx = index + offset;
		Voxel* nvoxel = VoxelHashMap::GetVoxel(info, nidx);
		if (!nvoxel || nvoxel->count == 0) return 0.0f;
		return nvoxel->sdfSum / (float)max(1u, nvoxel->count);
	};

	float sdf_x1 = get_sdf(make_int3(1, 0, 0));
	float sdf_x2 = get_sdf(make_int3(-1, 0, 0));
	float sdf_y1 = get_sdf(make_int3(0, 1, 0));
	float sdf_y2 = get_sdf(make_int3(0, -1, 0));
	float sdf_z1 = get_sdf(make_int3(0, 0, 1));
	float sdf_z2 = get_sdf(make_int3(0, 0, -1));

	float3 grad;
	grad.x = 0.5f * (sdf_x1 - sdf_x2) / info.voxelSize;
	grad.y = 0.5f * (sdf_y1 - sdf_y2) / info.voxelSize;
	grad.z = 0.5f * (sdf_z1 - sdf_z2) / info.voxelSize;

	float gradMag = length(grad);

	if (gradMag > gradientThreshold)
	{
		if (remove)
		{
			voxel->normalSum = make_float3(0, 0, 0);
			voxel->colorSum = make_float3(0, 0, 0);
			voxel->sdfSum = FLT_MAX;
			voxel->count = 0;
		}
		else
		{
			voxel->colorSum = make_float3((float)voxel->count, 0, 0);
		}
	}
}

__global__ void Kernel_VoxelHashMap_FilterBySDFGradient_26(
	VoxelHashMapInfo info,
	int offset,
	float gradientThreshold,
	bool remove)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 index = info.d_occupiedVoxelIndices[tid];
	Voxel* voxel = VoxelHashMap::GetVoxel(info, index);
	if (!voxel || voxel->count == 0) return;

	float3 grad = make_float3(0, 0, 0);
	int validCount = 0;

	for (int dz = -offset; dz <= offset; ++dz)
	{
		for (int dy = -offset; dy <= offset; ++dy)
		{
			for (int dx = -offset; dx <= offset; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				int3 dir = make_int3(dx, dy, dz);
				int3 i1 = index + dir;
				int3 i2 = index - dir;

				Voxel* v1 = VoxelHashMap::GetVoxel(info, i1);
				Voxel* v2 = VoxelHashMap::GetVoxel(info, i2);
				if (!v1 || !v2 || v1->count == 0 || v2->count == 0) continue;

				float sdf1 = v1->sdfSum / (float)max(1u, v1->count);
				float sdf2 = v2->sdfSum / (float)max(1u, v2->count);

				float3 unitDir = normalize(make_float3((float)dir.x, (float)dir.y, (float)dir.z) * info.voxelSize);
				float delta = (sdf1 - sdf2) / (2.0f * info.voxelSize);

				grad += unitDir * delta;
				++validCount;
			}
		}
	}

	if (validCount == 0) return;

	grad /= (float)validCount;
	float gradMag = length(grad);

	if (gradMag > gradientThreshold)
	{
		if (remove)
		{
			voxel->normalSum = make_float3(0, 0, 0);
			voxel->colorSum = make_float3(0, 0, 0);
			voxel->sdfSum = FLT_MAX;
			voxel->count = 0;
		}
		else
		{
			voxel->colorSum = make_float3((float)voxel->count, 0, 0);
		}
	}
}

__global__ void Kernel_VoxelHashMap_MarchingCubes(
	VoxelHashMapInfo info,
	float3* d_vertices,
	float3* d_normals,
	float3* d_colors,
	uint3* d_indices,
	uint64_t* d_edgeSlotKeys,
	int* d_edgeSlotValues,
	int edgeSlotCapacity,
	unsigned int* d_vertexCounter,
	unsigned int* d_triangleCounter)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels) return;

	int3 baseIndex = info.d_occupiedVoxelIndices[tid];

	float sdf[8];
	float3 normal[8], color[8];
	bool cornerValid[8] = { false };
	bool hasAnyCorner = false;

	for (int i = 0; i < 8; ++i)
	{
		int3 corner = baseIndex + MC_CORNERS[i];
		cornerValid[i] = VoxelHashMap::GetOrFillVoxelCorner(info, corner, sdf[i], normal[i], color[i]);
		if (cornerValid[i]) hasAnyCorner = true;
	}
	if (false == hasAnyCorner)
	{
		VoxelHashMap::FillMissingCornersWithNearest(baseIndex, sdf, normal, color, cornerValid);
		VoxelHashMap::FillHardMissingCorner(sdf, normal, color, cornerValid);
		for (int i = 0; i < 8; ++i)
		{
			int3 corner = baseIndex + MC_CORNERS[i];
			if (cornerValid[i]) hasAnyCorner = true;
		}
	}
	if (false == hasAnyCorner) return;

	int cubeIndex = 0;
	for (int i = 0; i < 8; ++i)
		if (sdf[i] < 0) cubeIndex |= (1 << i);

	int edgeMask = MC_EDGE_TABLE[cubeIndex];
	if (edgeMask == 0) return;

	int vertexIndex[12];
#pragma unroll
	for (int e = 0; e < 12; ++e)
	{
		vertexIndex[e] = -1;
		if (!(edgeMask & (1 << e))) continue;
		int edge0 = MC_EDGE_CONNECTIONS[e][0];
		int edge1 = MC_EDGE_CONNECTIONS[e][1];
		int3 i0 = baseIndex + MC_CORNERS[edge0];
		int3 i1 = baseIndex + MC_CORNERS[edge1];

		float sdf0 = sdf[edge0];
		float sdf1 = sdf[edge1];
		float3 n0 = normal[edge0];
		float3 n1 = normal[edge1];
		float3 c0 = color[edge0];
		float3 c1 = color[edge1];

		float3 p0 = VoxelHashMap::IndexToPosition(i0, info.voxelSize);
		float3 p1 = VoxelHashMap::IndexToPosition(i1, info.voxelSize);

		float alpha = sdf0 / (sdf0 - sdf1 + 1e-6f);
		alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
		float3 interpolatedPosition = p0 + alpha * (p1 - p0);
		float3 interpolatedNormal = normalize(n0 + alpha * (n1 - n0));
		float3 interpolatedColor = c0 + alpha * (c1 - c0);

		uint64_t edgeKey = VoxelHashMap::EncodeEdgeKey(i0, i1);
		int hash = (edgeKey ^ (edgeKey >> 32)) % edgeSlotCapacity;

		for (int probe = 0; probe < 8; ++probe)
		{
			int slot = (hash + probe) % edgeSlotCapacity;
			uint64_t old = atomicCAS(&d_edgeSlotKeys[slot], UINT64_MAX, edgeKey);
			int vtxIdx;
			if (old == UINT64_MAX)
			{
				vtxIdx = atomicAdd(d_vertexCounter, 1);
				d_vertices[vtxIdx] = interpolatedPosition;
				d_normals[vtxIdx] = interpolatedNormal;
				d_colors[vtxIdx] = interpolatedColor;
				d_edgeSlotValues[slot] = vtxIdx;
			}
			else if (old == edgeKey)
			{
				vtxIdx = d_edgeSlotValues[slot];
			}
			else continue;
			vertexIndex[e] = vtxIdx;
			break;
		}
	}

	for (int i = 0; MC_TRI_TABLE[cubeIndex][i] != -1; i += 3)
	{
		int a = vertexIndex[MC_TRI_TABLE[cubeIndex][i + 0]];
		int b = vertexIndex[MC_TRI_TABLE[cubeIndex][i + 1]];
		int c = vertexIndex[MC_TRI_TABLE[cubeIndex][i + 2]];
		if (a < 0 || b < 0 || c < 0) continue;
		if (a == b || b == c || c == a) continue;
		int t = atomicAdd(d_triangleCounter, 1);
		d_indices[t] = make_uint3(a, b, c);
	}
}

__global__ void Kernel_VoxelHashMap_Dilation(
	VoxelHashMapInfo info,
	int3* d_newOccupiedIndices,
	unsigned int* d_newOccupiedCount,
	int dilationStep)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *info.d_numberOfOccupiedVoxels)
		return;

	int3 base = info.d_occupiedVoxelIndices[tid];
	Voxel* baseVoxel = VoxelHashMap::GetVoxel(info, base);
	if (!baseVoxel || baseVoxel->count == 0)
		return;

	float sdf = baseVoxel->sdfSum / (float)max(1u, baseVoxel->count);
	const float threshold = info.voxelSize * 0.7f;

	for (int dz = -dilationStep; dz <= dilationStep; ++dz)
	{
		for (int dy = -dilationStep; dy <= dilationStep; ++dy)
		{
			for (int dx = -dilationStep; dx <= dilationStep; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;

				int3 neighbor = make_int3(base.x + dx, base.y + dy, base.z + dz);
				Voxel* neighborVoxel = VoxelHashMap::GetVoxel(info, neighbor);
				if (!neighborVoxel || neighborVoxel->count == 0)
				{
					VoxelKey key = VoxelHashMap::IndexToVoxelKey(neighbor);
					size_t hashIdx = VoxelHashMap::hash(key, info.capacity);

					for (unsigned int probe = 0; probe < info.maxProbe; ++probe)
					{
						size_t idx = (hashIdx + probe) % info.capacity;
						VoxelHashEntry* entry = &info.entries[idx];

						VoxelKey old = atomicCAS(
							reinterpret_cast<unsigned long long*>(&entry->key),
							EMPTY_KEY, key
						);

						if (old == EMPTY_KEY)
						{
							if (fabsf(sdf) < threshold)
							{
								entry->voxel.sdfSum = -sdf * baseVoxel->count;
								entry->voxel.normalSum = -baseVoxel->normalSum;
								entry->voxel.colorSum = baseVoxel->colorSum;
								entry->voxel.count = baseVoxel->count;
							}
							else
							{
								// 그냥 복사
								entry->voxel.sdfSum = baseVoxel->sdfSum;
								entry->voxel.normalSum = baseVoxel->normalSum;
								entry->voxel.colorSum = baseVoxel->colorSum;
								entry->voxel.count = baseVoxel->count;
							}

							entry->voxel.coordinate = neighbor;

							unsigned int occIdx = atomicAdd(d_newOccupiedCount, 1u);
							d_newOccupiedIndices[occIdx] = neighbor;
							break;
						}
						else if (old == key)
						{
							break;
						}
					}
				}
			}
		}
	}
}
