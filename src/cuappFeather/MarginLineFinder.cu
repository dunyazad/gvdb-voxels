#include <MarginLineFinder.cuh>

void MarginLineFinder::Initialize(float voxelSize, size_t capacity, uint8_t maxProbe)
{
	this->voxelSize = voxelSize;
	pointMap.Initialize(capacity, maxProbe);
	countMap.Initialize(capacity, maxProbe);
}

void MarginLineFinder::Terminate()
{
	pointMap.Terminate();
	countMap.Terminate();
}

void MarginLineFinder::Clear()
{
	pointMap.Clear();
	countMap.Clear();
}

void MarginLineFinder::InsertPoints(const std::vector<float3>& points, uint64_t tag)
{
	CUDA_TS(InsertPoints_Host);

	float3* d_points = nullptr;
	int numberOfPoints = static_cast<int>(points.size());
	if (numberOfPoints == 0)
		return;
	CUDA_CHECK(CUDA_MALLOC(&d_points, sizeof(float3) * numberOfPoints));
	CUDA_CHECK(CUDA_COPY_H2D(d_points, points.data(), sizeof(float3) * numberOfPoints));

	InsertPoints(d_points, numberOfPoints, tag);
	
	CUDA_SAFE_FREE(d_points);

	CUDA_TE(InsertPoints_Host);
}

__global__ void Kernel_MarginLineFinder_InsertPoints(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	const float3* points, int numberOfPoints, uint64_t tag, float voxelSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numberOfPoints) return;
	float3 p = points[idx];
	uint64_t key = MarginLineFinder::ToKey(p, voxelSize);

	//float3 rp = MarginLineFinder::FromKey(key, voxelSize);

	//if (length(p - rp) > voxelSize * 1.732f * 0.5f)
	//{
	//	printf("Warning: Point too far from voxel center! (%f, %f, %f) -> (%f, %f, %f), dist = %f\n",
	//		p.x, p.y, p.z, rp.x, rp.y, rp.z, length(p - rp));
	//	return;
	//}

	SimpleHashMap<uint64_t, uint64_t>::insert(info, key, tag);
}

void MarginLineFinder::InsertPoints(const float3* d_points, int numberOfPoints, uint64_t tag)
{
	if (numberOfPoints == 0)
		return;

	CUDA_TS(InsertPoints_Device);

	LaunchKernel(Kernel_MarginLineFinder_InsertPoints, numberOfPoints,
		pointMap.info, d_points, numberOfPoints, tag, voxelSize);

	CUDA_TE(InsertPoints_Device);
}

void MarginLineFinder::Dump(std::vector<float3>& resultPositions, std::vector<uint64_t>& resultTags)
{
	unsigned int* d_numberOfResultPositions = nullptr;
	CUDA_MALLOC(&d_numberOfResultPositions, sizeof(unsigned int));
	CUDA_MEMSET(d_numberOfResultPositions, 0, sizeof(unsigned int));

	float3* d_resultPositions = nullptr;
	unsigned int allocatedSize = pointMap.info.capacity;
	CUDA_MALLOC(&d_resultPositions, sizeof(float3) * allocatedSize);

	uint64_t* d_resultTags = nullptr;
	CUDA_MALLOC(&d_resultTags, sizeof(uint64_t) * allocatedSize);

	Dump(d_resultPositions, d_resultTags, d_numberOfResultPositions);

	unsigned int numberOfResultPositions = 0;
	CUDA_COPY_D2H(&numberOfResultPositions, d_numberOfResultPositions, sizeof(unsigned int));

	printf("Number of Result Positions: %u\n", numberOfResultPositions);

	resultPositions.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultPositions.data(), d_resultPositions, sizeof(float3) * numberOfResultPositions);

	resultTags.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultTags.data(), d_resultTags, sizeof(uint64_t) * numberOfResultPositions);

	CUDA_SAFE_FREE(d_resultPositions);
	CUDA_SAFE_FREE(d_resultTags);

	CUDA_SYNC();
}

__device__ uint64_t MarginLineFinder::FindRootVoxel(SimpleHashMapInfo<uint64_t, uint64_t> info, uint64_t key)
{
	while (true)
	{
		uint64_t parent = UINT64_MAX;
		SimpleHashMap<uint64_t, uint64_t>::find(info, key, &parent);
		if (parent == key | UINT64_MAX == parent) break;
		uint64_t grand = UINT64_MAX;
		SimpleHashMap<uint64_t, uint64_t>::find(info, parent, &grand);
		if (UINT64_MAX == grand) break;
		if (parent != grand) SimpleHashMap<uint64_t, uint64_t>::insert(info, key, grand);
		key = parent;
	}
	return key;
}

__device__ void MarginLineFinder::UnionVoxel(SimpleHashMapInfo<uint64_t, uint64_t> info, uint64_t a, uint64_t b)
{
	uint64_t rootA = FindRootVoxel(info, a);
	uint64_t rootB = FindRootVoxel(info, b);

	if (rootA != rootB)
	{
		if (rootA < rootB)
		{
			uint64_t vb = UINT64_MAX;
			SimpleHashMap<uint64_t, uint64_t>::find(info, rootB, &vb);
			if (rootA < vb)
				SimpleHashMap<uint64_t, uint64_t>::insert(info, rootB, rootA);
		}
		else
		{
			uint64_t va = UINT64_MAX;
			SimpleHashMap<uint64_t, uint64_t>::find(info, rootA, &va);
			if (rootB < va)
				SimpleHashMap<uint64_t, uint64_t>::insert(info, rootA, rootB);
		}
	}
}



















__global__ void Kernel_MarginLineFinder_Dump(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	float3* resultPositions, uint64_t* resultTags, unsigned int* numberOfResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= info.capacity) return;

	auto& entry = info.entries[tid];
	if (entry.key == UINT64_MAX)
		return;

	float3 p = MarginLineFinder::FromKey(entry.key, voxelSize);
	unsigned int index = atomicAdd(numberOfResultPositions, 1);
	resultPositions[index] = p;
	
	uint64_t cv = entry.value;
	resultTags[index] = cv;
}

void MarginLineFinder::Dump(
	float3* d_resultPositions, uint64_t* d_resultTags, unsigned int* d_numberOfResultPositions)
{
	LaunchKernel(Kernel_MarginLineFinder_Dump, pointMap.info.capacity,
		pointMap.info, d_resultPositions, d_resultTags , d_numberOfResultPositions, voxelSize);
}

void MarginLineFinder::FindMarginLinePoints(std::vector<float3>& result)
{
	unsigned int* d_numberOfResultPoints = nullptr;
	CUDA_MALLOC(&d_numberOfResultPoints, sizeof(unsigned int));
	CUDA_MEMSET(d_numberOfResultPoints, 0, sizeof(unsigned int));

	float3* d_resultPoints = nullptr;
	unsigned int allocatedSize = pointMap.info.capacity;
	CUDA_MALLOC(&d_resultPoints, sizeof(float3) * allocatedSize);

	FindMarginLinePoints(d_resultPoints, d_numberOfResultPoints);

	unsigned int numberOfResultPoints = 0;
	CUDA_COPY_D2H(&numberOfResultPoints, d_numberOfResultPoints, sizeof(unsigned int));

	printf("Number of Result Points: %u\n", numberOfResultPoints);

	result.resize(numberOfResultPoints);
	CUDA_COPY_D2H(result.data(), d_resultPoints, sizeof(float3) * numberOfResultPoints);

	CUDA_SAFE_FREE(d_resultPoints);

	CUDA_SYNC();
}

__global__ void Kernel_MarginLineFinder_FindMarginLinePoints(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	float3* resultPoints, unsigned int* numberOfResultPoints, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= info.capacity) return;

	auto& entry = info.entries[tid];
	if (entry.key == UINT64_MAX)
		return;

	auto cv = entry.value;

	if (cv != 2)
	{
		return;
	}

	auto ci = MarginLineFinder::KeyToIndex(entry.key);

	int offset = 2;
	bool found = false;
#pragma unroll
	for (int dz = -offset; dz <= offset; ++dz)
	{
#pragma unroll
		for (int dy = -offset; dy <= offset; ++dy)
		{
#pragma unroll
			for (int dx = -offset; dx <= offset; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;
		
				int3 ni = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
				uint64_t nkey = MarginLineFinder::ToKey(ni);

				uint64_t nv = 0;
				if (SimpleHashMap<uint64_t, uint64_t>::find(info, nkey, &nv))
				{
					if(1 == nv)
					{
						found = true;
						break;
					}
				}
			}
			if (found)
			{
				break;
			}
		}
		if (found)
		{
			break;
		}
	}

	if (!found)
		return;

	float3 p = MarginLineFinder::FromKey(entry.key, voxelSize);
	unsigned int index = atomicAdd(numberOfResultPoints, 1);
	resultPoints[index] = p;
}

void MarginLineFinder::FindMarginLinePoints(float3* d_resultPoints, unsigned int* d_numberofResultPoints)
{
	LaunchKernel(Kernel_MarginLineFinder_FindMarginLinePoints, pointMap.info.capacity,
		pointMap.info, d_resultPoints, d_numberofResultPoints, voxelSize);
}

void MarginLineFinder::MarginLineNoiseRemoval(std::vector<float3>& result)
{
	unsigned int* d_numberOfResultPoints = nullptr;
	CUDA_MALLOC(&d_numberOfResultPoints, sizeof(unsigned int));
	CUDA_MEMSET(d_numberOfResultPoints, 0, sizeof(unsigned int));

	float3* d_resultPoints = nullptr;
	unsigned int allocatedSize = pointMap.info.capacity;
	CUDA_MALLOC(&d_resultPoints, sizeof(float3) * allocatedSize);

	MarginLineNoiseRemoval(d_resultPoints, d_numberOfResultPoints);

	unsigned int numberOfResultPoints = 0;
	CUDA_COPY_D2H(&numberOfResultPoints, d_numberOfResultPoints, sizeof(unsigned int));

	printf("Number of Result Points: %u\n", numberOfResultPoints);

	result.resize(numberOfResultPoints);
	CUDA_COPY_D2H(result.data(), d_resultPoints, sizeof(float3) * numberOfResultPoints);

	CUDA_SAFE_FREE(d_resultPoints);

	CUDA_SYNC();
}

__global__ void Kernel_MarginLineFinder_NoiseRemoval(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	float3* resultPoints, unsigned int* numberOfResultPoints, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= info.capacity) return;
	auto& entry = info.entries[tid];
	if (entry.key == UINT64_MAX)
		return;

	auto ci = MarginLineFinder::KeyToIndex(entry.key);

	int offset = 2;
	unsigned int count = 0;
#pragma unroll
	for (int dz = -offset; dz <= offset; ++dz)
	{
#pragma unroll
		for (int dy = -offset; dy <= offset; ++dy)
		{
#pragma unroll
			for (int dx = -offset; dx <= offset; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;

				int3 ni = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
				uint64_t nkey = MarginLineFinder::ToKey(ni);

				uint64_t nv = 0;
				if (SimpleHashMap<uint64_t, uint64_t>::find(info, nkey, &nv))
				{
					count++;
				}
			}
		}
	}

	if (count < 10)
		return;

	float3 p = MarginLineFinder::FromKey(entry.key, voxelSize);
	unsigned int index = atomicAdd(numberOfResultPoints, 1);
	resultPoints[index] = p;
}

void MarginLineFinder::MarginLineNoiseRemoval(float3* d_resultPoints, unsigned int* d_numberofResultPoints)
{
	LaunchKernel(Kernel_MarginLineFinder_NoiseRemoval, pointMap.info.capacity,
		pointMap.info, d_resultPoints, d_numberofResultPoints, voxelSize);
}

void MarginLineFinder::Clustering(std::vector<float3>& resultPositions, std::vector<uint64_t>& resultTags, std::vector<uint64_t>& resultTagCounts)
{
	unsigned int* d_numberOfResultPositions = nullptr;
	CUDA_MALLOC(&d_numberOfResultPositions, sizeof(unsigned int));
	CUDA_MEMSET(d_numberOfResultPositions, 0, sizeof(unsigned int));

	float3* d_resultPositions = nullptr;
	unsigned int allocatedSize = pointMap.info.capacity;
	CUDA_MALLOC(&d_resultPositions, sizeof(float3) * allocatedSize);

	uint64_t* d_resultTags = nullptr;
	CUDA_MALLOC(&d_resultTags, sizeof(uint64_t) * allocatedSize);

	uint64_t* d_resultTagCounts = nullptr;
	CUDA_MALLOC(&d_resultTagCounts, sizeof(uint64_t) * allocatedSize);

	Clustering(d_resultPositions, d_resultTags, d_resultTagCounts, d_numberOfResultPositions);

	unsigned int numberOfResultPositions = 0;
	CUDA_COPY_D2H(&numberOfResultPositions, d_numberOfResultPositions, sizeof(unsigned int));

	printf("Number of Result Positions: %u\n", numberOfResultPositions);

	resultPositions.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultPositions.data(), d_resultPositions, sizeof(float3) * numberOfResultPositions);

	resultTags.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultTags.data(), d_resultTags, sizeof(uint64_t) * numberOfResultPositions);

	resultTagCounts.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultTagCounts.data(), d_resultTagCounts, sizeof(uint64_t) * numberOfResultPositions);

	CUDA_SAFE_FREE(d_resultPositions);
	CUDA_SAFE_FREE(d_resultTags);
	CUDA_SAFE_FREE(d_resultTagCounts);

	CUDA_SYNC();
}

__global__ void Kernel_MarginLineFinder_Prepare_Clustering(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= info.capacity) return;

	auto& entry = info.entries[tid];
	if (entry.key == UINT64_MAX)
		return;

	entry.value = entry.key;

	float3 p = MarginLineFinder::FromKey(entry.key, voxelSize);
	unsigned int index = atomicAdd(d_numberofResultPositions, 1);
	d_resultPositions[index] = p;
	d_resultTags[index] = UINT64_MAX;
	d_resultTagCounts[index] = 0;
}

__global__ void Kernel_MarginLineFinder_Clustering(
	SimpleHashMapInfo<uint64_t, uint64_t> info,
	float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *d_numberofResultPositions) return;
	float3 p = d_resultPositions[tid];
	
	uint64_t key = MarginLineFinder::ToKey(p, voxelSize);
	int offset = 1;
#pragma unroll
	for (int dz = -offset; dz <= offset; ++dz)
	{
#pragma unroll
		for (int dy = -offset; dy <= offset; ++dy)
		{
#pragma unroll
			for (int dx = -offset; dx <= offset; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;
				int3 ni = MarginLineFinder::KeyToIndex(key);
				ni.x += dx;
				ni.y += dy;
				ni.z += dz;
				uint64_t nkey = MarginLineFinder::ToKey(ni);
				uint64_t nv = 0;
				if (SimpleHashMap<uint64_t, uint64_t>::find(info, nkey, &nv))
				{
					MarginLineFinder::UnionVoxel(info, key, nkey);
				}
			}
		}
	}

	uint64_t value = UINT64_MAX;
	SimpleHashMap<uint64_t, uint64_t>::find(info, key, &value);
	d_resultTags[tid] = value;
}

__global__ void Kernel_MarginLineFinder_Count_Tags(
	SimpleHashMapInfo<uint64_t, uint64_t> info_countMap,
	float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *d_numberofResultPositions) return;
	auto tag = d_resultTags[tid];
	if (tag == UINT64_MAX)
		return;

	if (false == SimpleHashMap<uint64_t, uint64_t>::increase(info_countMap, tag))
	{
		printf("Error: Failed to increase count for tag %llu\n", tag);
	}
}

__global__ void Kernel_MarginLineFinder_Fill_TagCounts(
	SimpleHashMapInfo<uint64_t, uint64_t> info_countMap,
	float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *d_numberofResultPositions) return;
	auto tag = d_resultTags[tid];
	if (tag == UINT64_MAX)
		return;

	uint64_t count = 0;
	SimpleHashMap<uint64_t, uint64_t>::find(info_countMap, tag, &count);
	d_resultTagCounts[tid] = count;
}

void MarginLineFinder::Clustering(float3* d_resultPositions, uint64_t* d_resultTags, uint64_t* d_resultTagCounts, unsigned int* d_numberofResultPositions)
{
	LaunchKernel(Kernel_MarginLineFinder_Prepare_Clustering, pointMap.info.capacity,
		pointMap.info, d_resultPositions, d_resultTags, d_resultTagCounts, d_numberofResultPositions, voxelSize);

	unsigned int numberOfResultPoints = 0;
	CUDA_COPY_D2H(&numberOfResultPoints, d_numberofResultPositions, sizeof(unsigned int));
	if (numberOfResultPoints == 0)
		return;

	printf("Number of Points before Clustering: %u\n", numberOfResultPoints);

	LaunchKernel(Kernel_MarginLineFinder_Clustering, numberOfResultPoints,
		pointMap.info, d_resultPositions, d_resultTags, d_resultTagCounts, d_numberofResultPositions, voxelSize);

	LaunchKernel(Kernel_MarginLineFinder_Count_Tags, numberOfResultPoints,
		countMap.info, d_resultPositions, d_resultTags, d_resultTagCounts, d_numberofResultPositions, voxelSize);

	LaunchKernel(Kernel_MarginLineFinder_Fill_TagCounts, numberOfResultPoints,
		countMap.info, d_resultPositions, d_resultTags, d_resultTagCounts, d_numberofResultPositions, voxelSize);
}
