#include <HPOctree.cuh>

__global__ void Kernel_HPOctree_Occupy(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	float3* d_positions,
	unsigned int numberOfPositions,
	unsigned int maxDepth,
	cuAABB aabb,
	float domain_length)
{
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= numberOfPositions) return;

	const float3& p = d_positions[tid];

	HPOctreeKey leaf_key;
	leaf_key.FromPosition(p, aabb, domain_length, maxDepth);

	for (int d = leaf_key.d; d >= 0; --d)
	{
		HPOctreeKey ancestor_key = leaf_key.GetAncestorKey(d);
		uint64_t ancestor_code = ancestor_key.ToCode();

		unsigned int dummy_value;
		if (SimpleHashMap<uint64_t, unsigned int>::find(info, ancestor_code, &dummy_value))
		{
			break;
		}

		SimpleHashMap<uint64_t, unsigned int>::insert(info, ancestor_code, 1);
	}
}

void HPOctree::Initialize(const vector<float3>& positions, const cuAABB& aabb, unsigned int maxDepth)
{
	originalAABB = aabb;
	domainAABB = HPOctreeKey::GetDomainAABB(aabb);
	this->domain_length = HPOctreeKey::GetDomainLength(domainAABB);

	numberOfPositions = positions.size();

	CUDA_MALLOC(&d_positions, sizeof(float3) * numberOfPositions);
	CUDA_COPY_H2D(d_positions, positions.data(), sizeof(float3) * numberOfPositions);
	CUDA_SYNC();

	// [수정] 최대 깊이를 HPOctreeKey의 상수로부터 직접 계산합니다.
	// 포인트 하나가 생성할 수 있는 최대 키의 개수입니다.
	const unsigned int max_keys_per_point = (1u << HPOctreeKey::DEPTH_BITS); // 2^7 = 128

	// 필요한 총 용량을 계산합니다.
	const size_t required_capacity = (size_t)numberOfPositions * max_keys_per_point;

	// 해시맵의 로드 팩터(load factor)를 고려하여 여유 공간을 줍니다 (예: 2배).
	// 너무 빡빡하면 충돌이 많아져 maxProbe를 초과하고 insert가 실패할 수 있습니다.
	const size_t capacity_with_margin = required_capacity * 2;

	// 계산된 용량으로 해시맵을 초기화합니다.
	keys.Initialize(capacity_with_margin, 64);

	CUDA_TS(OctreeInitialiize);

	LaunchKernel(Kernel_HPOctree_Occupy, numberOfPositions,
		keys.info, d_positions, numberOfPositions, maxDepth, domainAABB, domain_length);

	CUDA_TE(OctreeInitialiize);
}

void HPOctree::Initialize(float3* h_positions, unsigned int numberOfPositions, const cuAABB& aabb, unsigned int maxDepth)
{
	originalAABB = aabb;
	domainAABB = HPOctreeKey::GetDomainAABB(aabb);
	this->domain_length = HPOctreeKey::GetDomainLength(aabb);

	this->numberOfPositions = numberOfPositions;

	CUDA_MALLOC(&d_positions, sizeof(float3) * numberOfPositions);
	CUDA_COPY_H2D(d_positions, h_positions, sizeof(float3) * numberOfPositions);
	CUDA_SYNC();

	// [수정] 최대 깊이를 HPOctreeKey의 상수로부터 직접 계산합니다.
	// 포인트 하나가 생성할 수 있는 최대 키의 개수입니다.
	const unsigned int max_keys_per_point = (1u << HPOctreeKey::DEPTH_BITS); // 2^7 = 128

	// 필요한 총 용량을 계산합니다.
	const size_t required_capacity = (size_t)numberOfPositions * max_keys_per_point;

	// 해시맵의 로드 팩터(load factor)를 고려하여 여유 공간을 줍니다 (예: 2배).
	// 너무 빡빡하면 충돌이 많아져 maxProbe를 초과하고 insert가 실패할 수 있습니다.
	const size_t capacity_with_margin = required_capacity * 2;

	// 계산된 용량으로 해시맵을 초기화합니다.
	keys.Initialize(capacity_with_margin, 64);

	CUDA_TS(OctreeInitialiize);

	LaunchKernel(Kernel_HPOctree_Occupy, numberOfPositions,
		keys.info, d_positions, numberOfPositions, maxDepth, domainAABB, domain_length);

	CUDA_TE(OctreeInitialiize);
}

void HPOctree::Terminate()
{
	CUDA_SAFE_FREE(d_positions);
}

__global__ void Kernel_CompactOccupiedKeys(
	const SimpleHashMapEntry<uint64_t, unsigned int>* hash_entries,
	size_t capacity,
	uint64_t* compacted_keys,
	unsigned int* counter)
{
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= capacity) return;

	const uint64_t empty_val = empty_key<uint64_t>();
	const uint64_t key_code = hash_entries[tid].key;

	// 현재 슬롯이 비어있지 않다면
	if (key_code != empty_val)
	{
		// 원자적 덧셈을 이용해 이 키를 저장할 고유한 인덱스를 얻습니다.
		unsigned int write_idx = atomicAdd(counter, 1);

		// 압축된 배열에 키 코드를 기록합니다.
		compacted_keys[write_idx] = key_code;
	}
}

vector<HPOctreeKey> HPOctree::Dump()
{
	// 1. 점유된 엔트리의 총 개수를 GPU에서 가져옵니다.
	unsigned int num_occupied_entries = 0;
	// (m_hashMap은 HPOctree 클래스의 멤버 변수라고 가정)
	cudaMemcpy(&num_occupied_entries, keys.info.numberOfEntries, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// 점유된 셀이 없으면 빈 벡터를 반환합니다.
	if (num_occupied_entries == 0)
	{
		return {};
	}

	// 2. 압축된 키 코드를 저장할 GPU 메모리를 할당합니다.
	uint64_t* d_compacted_keys = nullptr;
	cudaMalloc(&d_compacted_keys, sizeof(uint64_t) * num_occupied_entries);

	// 3. 압축 커널에서 사용할 원자적 카운터를 할당하고 0으로 초기화합니다.
	unsigned int* d_compaction_counter = nullptr;
	cudaMalloc(&d_compaction_counter, sizeof(unsigned int));
	cudaMemset(d_compaction_counter, 0, sizeof(unsigned int));

	// 4. 압축 커널을 실행하여 유효한 키들을 d_compacted_keys로 모읍니다.
	const size_t capacity = keys.info.capacity;
	const int threads_per_block = 256;
	const int blocks = (capacity + threads_per_block - 1) / threads_per_block;

	Kernel_CompactOccupiedKeys << <blocks, threads_per_block >> > (
		keys.info.entries,
		capacity,
		d_compacted_keys,
		d_compaction_counter
		);
	cudaDeviceSynchronize(); // 커널 실행이 끝날 때까지 동기화

	// 5. 압축된 키 코드들을 GPU에서 호스트(CPU)로 복사합니다.
	std::vector<uint64_t> h_compacted_codes(num_occupied_entries);
	cudaMemcpy(
		h_compacted_codes.data(),
		d_compacted_keys,
		sizeof(uint64_t) * num_occupied_entries,
		cudaMemcpyDeviceToHost
	);

	// 6. 커널 실행을 위해 할당했던 임시 GPU 메모리를 해제합니다.
	cudaFree(d_compacted_keys);
	cudaFree(d_compaction_counter);

	// 7. 호스트로 가져온 uint64_t 코드들을 HPOctreeKey 구조체로 변환합니다.
	std::vector<HPOctreeKey> result_keys;
	result_keys.reserve(num_occupied_entries); // 벡터 재할당 방지를 위해 메모리 미리 확보

	for (uint64_t code : h_compacted_codes)
	{
		HPOctreeKey key;
		key.FromCode(code);
		result_keys.push_back(key);
	}

	// 8. 최종 결과를 반환합니다.
	return result_keys;
}





__device__ NNS_Result FindNearestNeighbor(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	const float3& query_pos,
	const float3* d_positions,
	unsigned int numPositions,
	cuAABB domainAABB,
	float domain_length)
{
	constexpr int STACK_SIZE = 64;

	NNS_Result best_result;

	HPOctreeKey stack[STACK_SIZE];
	int stack_ptr = 0;

	// 탐색을 루트 노드(depth=0)에서 시작
	HPOctreeKey root_key;
	root_key.d = 0; root_key.x = 0; root_key.y = 0; root_key.z = 0;
	stack[stack_ptr++] = root_key;

	while (stack_ptr > 0)
	{
		HPOctreeKey current_key = stack[--stack_ptr];

		cuAABB current_aabb = HPOctreeKey::GetAABB(current_key, domainAABB, domain_length);

		// *** 가지치기(Pruning) ***
		if (detail::PointToAABBDistanceSq(query_pos, current_aabb) >= best_result.distance_sq)
		{
			continue;
		}

		if (current_key.d == 15)
		{
			const uint64_t code = current_key.ToCode();
			unsigned int pointIndex = 0;
			if (SimpleHashMap<uint64_t, unsigned int>::find(info, code, &pointIndex))
			{
				if (pointIndex < numPositions)
				{
					float dist_sq = detail::DistanceSq(query_pos, d_positions[pointIndex]);
					if (dist_sq < best_result.distance_sq)
					{
						best_result.distance_sq = dist_sq;
						best_result.index = pointIndex;
					}
				}
			}
		}
		else
		{
			if (stack_ptr + 8 > STACK_SIZE)
			{
				// 스택 오버플로우 발생 가능. 실제로는 깊이가 15로 제한되어 거의 발생하지 않음.
				// 필요시 에러 처리 추가 가능 (e.g., printf from device)
				continue;
			}

			unsigned int child_depth = current_key.d + 1;
			unsigned int child_x_base = current_key.x << 1;
			unsigned int child_y_base = current_key.y << 1;
			unsigned int child_z_base = current_key.z << 1;

			for (int i = 0; i < 8; ++i)
			{
				HPOctreeKey child_key;
				child_key.d = child_depth;
				child_key.x = child_x_base + ((i & 4) >> 2);
				child_key.y = child_y_base + ((i & 2) >> 1);
				child_key.z = child_z_base + (i & 1);
				stack[stack_ptr++] = child_key;
			}
		}
	}

	return best_result;
}

void HPOctree::SearchOnDevice(const std::vector<float3>& h_query_points, std::vector<NNS_Result>& h_results)
{
	if (h_query_points.empty())
	{
		h_results.clear();
		return;
	}

	unsigned int numQueries = h_query_points.size();
	h_results.resize(numQueries);

	// Device에 쿼리 포인트와 결과 버퍼 할당
	float3* d_query_points;
	NNS_Result* d_results;
	cudaMalloc(&d_query_points, sizeof(float3) * numQueries);
	cudaMalloc(&d_results, sizeof(NNS_Result) * numQueries);

	// 쿼리 포인트 복사 (Host -> Device)
	cudaMemcpy(d_query_points, h_query_points.data(), sizeof(float3) * numQueries, cudaMemcpyHostToDevice);

	LaunchKernel(FindNearestNeighbor_kernel, numQueries,
		keys.info, d_query_points, d_results, d_positions, numQueries, numberOfPositions, domainAABB, domain_length);

	// 결과 복사 (Device -> Host)
	cudaMemcpy(h_results.data(), d_results, sizeof(NNS_Result) * numQueries, cudaMemcpyDeviceToHost);

	// 임시 Device 메모리 해제
	cudaFree(d_query_points);
	cudaFree(d_results);
}

__global__ void FindNearestNeighbor_kernel(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	const float3* d_query_points,
	NNS_Result* d_results,
	const float3* d_positions,
	unsigned int numQueries,
	unsigned int numPositions,
	cuAABB domainAABB,
	float domain_length)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numQueries) return;

	float3 query_pos = d_query_points[idx];

	d_results[idx] = FindNearestNeighbor(info, query_pos, d_positions, numPositions, domainAABB, domain_length);
}