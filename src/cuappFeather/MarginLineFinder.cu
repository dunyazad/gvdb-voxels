#include <MarginLineFinder.cuh>

void MarginLineFinder::Initialize(float voxelSize, size_t capacity, uint8_t maxProbe)
{
	this->voxelSize = voxelSize;
	pointMap.Initialize(capacity, maxProbe);
}

void MarginLineFinder::Terminate()
{
	pointMap.Terminate();
}

void MarginLineFinder::Clear()
{
	pointMap.Clear();
}

void MarginLineFinder::InsertPoints(const std::vector<float3>& points, unsigned char tag)
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
	HashMapInfo<uint64_t, unsigned char> info,
	const float3* points, int numberOfPoints, unsigned char tag, float voxelSize)
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

	HashMap<uint64_t, unsigned char>::insert(info, key, static_cast<unsigned char>(tag));
}

void MarginLineFinder::InsertPoints(const float3* d_points, int numberOfPoints, unsigned char tag)
{
	if (numberOfPoints == 0)
		return;

	CUDA_TS(InsertPoints_Device);

	LaunchKernel(Kernel_MarginLineFinder_InsertPoints, numberOfPoints,
		pointMap.info, d_points, numberOfPoints, tag, voxelSize);

	CUDA_TE(InsertPoints_Device);
}

void MarginLineFinder::Dump(std::vector<float3>& resultPositions, std::vector<unsigned int>& resultTags)
{
	unsigned int* d_numberOfResultPositions = nullptr;
	CUDA_MALLOC(&d_numberOfResultPositions, sizeof(unsigned int));
	CUDA_MEMSET(d_numberOfResultPositions, 0, sizeof(unsigned int));

	float3* d_resultPositions = nullptr;
	unsigned int allocatedSize = pointMap.info.capacity;
	CUDA_MALLOC(&d_resultPositions, sizeof(float3) * allocatedSize);

	unsigned int* d_resultTags = nullptr;
	CUDA_MALLOC(&d_resultTags, sizeof(unsigned int) * allocatedSize);

	Dump(d_resultPositions, d_resultTags, d_numberOfResultPositions);

	unsigned int numberOfResultPositions = 0;
	CUDA_COPY_D2H(&numberOfResultPositions, d_numberOfResultPositions, sizeof(unsigned int));

	printf("Number of Result Positions: %u\n", numberOfResultPositions);

	resultPositions.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultPositions.data(), d_resultPositions, sizeof(float3) * numberOfResultPositions);

	resultTags.resize(numberOfResultPositions);
	CUDA_COPY_D2H(resultTags.data(), d_resultTags, sizeof(unsigned int) * numberOfResultPositions);

	CUDA_SAFE_FREE(d_resultPositions);
	CUDA_SAFE_FREE(d_resultTags);

	CUDA_SYNC();
}

__global__ void Kernel_MarginLineFinder_Dump(
	HashMapInfo<uint64_t, unsigned char> info,
	float3* resultPositions, unsigned int* resultTags, unsigned int* numberOfResultPositions, float voxelSize)
{
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= info.capacity) return;

	auto& entry = info.entries[tid];
	if (entry.key == UINT64_MAX)
		return;

	float3 p = MarginLineFinder::FromKey(entry.key, voxelSize);
	unsigned int index = atomicAdd(numberOfResultPositions, 1);
	resultPositions[index] = p;
	
	auto cv = entry.value;
	resultTags[index] = static_cast<unsigned int>(cv);
}

void MarginLineFinder::Dump(
	float3* d_resultPositions, unsigned int* d_resultTags, unsigned int* d_numberOfResultPositions)
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
	HashMapInfo<uint64_t, unsigned char> info,
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

				unsigned char nv = 0;
				if (HashMap<uint64_t, unsigned char>::find(info, nkey, &nv))
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
	HashMapInfo<uint64_t, unsigned char> info,
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

				unsigned char nv = 0;
				if (HashMap<uint64_t, unsigned char>::find(info, nkey, &nv))
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
