#include <HPOctree.cuh>

__global__ void Kernel_CreateLeafMap(
	SimpleHashMapInfo<uint64_t, unsigned int> leaf_map_info,
	const float3* d_positions,
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
	uint64_t leaf_code = leaf_key.ToCode();

	SimpleHashMap<uint64_t, unsigned int>::insert(leaf_map_info, leaf_code, tid);
}

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

__global__ void Kernel_CompactOccupiedKeys(
	SimpleHashMapInfo<uint64_t, unsigned int> info,
	uint64_t* compacted_keys,
	unsigned int* counter)
{
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= info.capacity) return;

	const uint64_t empty_val = empty_key<uint64_t>();
	const uint64_t key_code = info.entries[tid].key;

	if (key_code != empty_val)
	{
		unsigned int write_idx = atomicAdd(counter, 1);

		compacted_keys[write_idx] = key_code;
	}
}

__device__ inline unsigned int lower_bound_device(const uint64_t* arr, unsigned int arr_size, uint64_t value)
{
	unsigned int low = 0;
	unsigned int high = arr_size;
	while (low < high)
	{
		unsigned int mid = low + (high - low) / 2;
		if (arr[mid] < value)
		{
			low = mid + 1;
		}
		else
		{
			high = mid;
		}
	}
	return low;
}

__global__ void Kernel_BuildOctreeNodes(
	SimpleHashMapInfo<uint64_t, unsigned int> leaf_map_info,
	HPOctreeNode* nodes,
	const uint64_t* d_sorted_keys,
	unsigned int numNodes,
	unsigned int maxDepth)
{
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= numNodes) return;

	const uint64_t my_key_code = d_sorted_keys[tid];
	HPOctreeKey my_key;
	my_key.FromCode(my_key_code);

	uint32_t parent_idx = UINT32_MAX;
	if (my_key.d > 0)
	{
		HPOctreeKey parent_key = my_key.GetParentKey();
		uint64_t parent_key_code = parent_key.ToCode();
		parent_idx = lower_bound_device(d_sorted_keys, numNodes, parent_key_code);
		if (parent_idx >= numNodes || d_sorted_keys[parent_idx] != parent_key_code)
		{
			parent_idx = UINT32_MAX;
		}
	}

	unsigned int point_idx = UINT32_MAX; // 기본값은 '없음'
	if (my_key.d == maxDepth) // 현재 노드가 리프 노드라면
	{
		// leaf_map에서 현재 리프 키로 pointIndex를 찾는다.
		SimpleHashMap<uint64_t, unsigned int>::find(leaf_map_info, my_key_code, &point_idx);
	}

	uint32_t first_child_idx = UINT32_MAX;
	uint32_t child_count = 0;
	if (my_key.d < maxDepth)
	{
		// 현재 노드의 Morton 코드 부분만 추출 (상위 깊이 비트 제거)
		const uint64_t my_morton = my_key_code & HPOctreeKey::COORD_MASK;

		// 자식 노드의 깊이에 해당하는 비트 마스크를 계산
		const uint64_t child_depth_bits = (uint64_t)(my_key.d + 1) << HPOctreeKey::DEPTH_SHIFT;

		// **하한선**: 첫 번째 자식(000)의 전체 키(Key) 계산
		// 부모 Morton 코드를 3비트 왼쪽으로 밀면 첫 자식의 Morton 코드가 됨
		const uint64_t first_child_morton = my_morton << 3;
		const uint64_t lower_bound_code = child_depth_bits | first_child_morton;

		// **상한선**: 마지막 자식(111) 바로 다음 키 계산
		// 부모의 다음 형제 노드의 첫 번째 자식에 해당함
		const uint64_t after_last_child_morton = (my_morton + 1) << 3;
		const uint64_t upper_bound_code = child_depth_bits | after_last_child_morton;

		// 이진 탐색으로 자식 그룹의 시작과 끝 인덱스를 찾음
		const uint32_t start_idx = lower_bound_device(d_sorted_keys, numNodes, lower_bound_code);
		const uint32_t end_idx = lower_bound_device(d_sorted_keys, numNodes, upper_bound_code);

		child_count = end_idx - start_idx;
		if (child_count > 0)
		{
			first_child_idx = start_idx;
		}
	}

	nodes[tid].parentIndex = parent_idx;
	nodes[tid].firstChildIndex = first_child_idx;
	nodes[tid].childCount = child_count;
	nodes[tid].key = my_key;
	nodes[tid].pointIndex = point_idx;
}

__global__ void Kernel_NNSearch(
	NNS_Result* d_results,
	const float3* d_query_points,
	unsigned int numQueries,
	const HPOctreeNode* d_nodes,
	unsigned int numNodes,
	const float3* d_positions, // 원본 데이터 포인트 배열
	cuAABB domain_aabb,
	float domain_length,
	unsigned int maxDepth)
{
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= numQueries) return;

	const float3 query_pos = d_query_points[tid];

	// 1. 각 스레드(쿼리 포인트)를 위한 변수 초기화
	float best_dist_sq = FLT_MAX;
	unsigned int best_index = UINT32_MAX;

	// GPU 커널 내에서 사용할 작은 로컬 메모리 스택
	unsigned int stack[64]; // 옥트리 최대 깊이보다 충분히 큰 크기
	int stack_ptr = 0;

	// 2. 탐색 시작: 스택에 루트 노드(인덱스 0)를 넣는다.
	stack[stack_ptr++] = 0;

	int count = 0;
	// 3. 스택이 빌 때까지 깊이 우선 탐색(DFS) 수행
	while (stack_ptr > 0)
	{
		unsigned int nodeIndex = stack[--stack_ptr]; // 스택에서 노드를 꺼냄(Pop)
		const HPOctreeNode& node = d_nodes[nodeIndex];

		// **가지치기(Pruning) 검사**
		cuAABB aabb = node.key.GetAABB(domain_aabb, domain_length);
		if (detail::PointToAABBDistanceSq(query_pos, aabb) >= best_dist_sq)
		{
			continue; // 이 노드와 모든 자식은 탐색할 필요 없음
		}

		// 4. 노드 타입에 따라 처리
		if (node.childCount > 0) // 내부 노드(Internal Node)인 경우
		{
			// 자식들을 스택에 넣는다 (Push)
			// (최적화: 쿼리 지점과 가까운 자식부터 스택에 넣으면 더 좋음)
			for (unsigned int i = 0; i < node.childCount; ++i)
			{
				if (stack_ptr < 64) // 스택 오버플로우 방지
				{
					stack[stack_ptr++] = node.firstChildIndex + i;
				}
			}
		}
		else // 리프 노드(Leaf Node)인 경우
		{
			// 노드에 저장된 pointIndex를 이용해 실제 데이터 포인트와 거리 비교
			unsigned int point_idx = node.pointIndex;
			if (UINT32_MAX != point_idx)
			{
				float dist_sq = detail::DistanceSq(query_pos, d_positions[point_idx]);

				// 더 가까운 점을 찾으면 업데이트
				if (dist_sq < best_dist_sq)
				{
					best_dist_sq = dist_sq;
					best_index = point_idx;
				}
			}
		}
	}

	// 5. 최종 결과를 전역 메모리에 기록
	d_results[tid].index = best_index;
	d_results[tid].distance_sq = best_dist_sq;
}

void HPOctree::Initialize(const std::vector<float3>& positions, const cuAABB& aabb, unsigned int maxDepth)
{
	Initialize(positions.data(), positions.size(), aabb, maxDepth);
}

void HPOctree::Initialize(const float3* h_positions, unsigned int numberOfPositions, const cuAABB& aabb, unsigned int maxDepth)
{
	originalAABB = aabb;
	domainAABB = HPOctreeKey::GetDomainAABB(aabb);
	this->domain_length = HPOctreeKey::GetDomainLength(domainAABB);

	this->maxDepth = maxDepth;
	if (0 == this->maxDepth) this->maxDepth = (1u << HPOctreeKey::DEPTH_BITS) - 1;
	
	this->numberOfPositions = numberOfPositions;

	CUDA_MALLOC(&d_positions, sizeof(float3) * numberOfPositions);
	CUDA_COPY_H2D(d_positions, h_positions, sizeof(float3) * numberOfPositions);
	CUDA_SYNC();

	const unsigned int max_keys_per_point = this->maxDepth + 1;
	const size_t required_capacity = (size_t)numberOfPositions * max_keys_per_point;
	const size_t capacity_with_margin = required_capacity * 2;
	
	keys.Initialize(capacity_with_margin, 64);
	leaf_map.Initialize(capacity_with_margin, 64);

	CUDA_TS(OctreeInitialiize);

	LaunchKernel(Kernel_CreateLeafMap, numberOfPositions,
		leaf_map.info, d_positions, numberOfPositions, this->maxDepth, domainAABB, domain_length);

	LaunchKernel(Kernel_HPOctree_Occupy, numberOfPositions,
		keys.info, d_positions, numberOfPositions, this->maxDepth, domainAABB, domain_length);

	CUDA_COPY_D2H(&numberOfNodes, keys.info.numberOfEntries, sizeof(unsigned int));

	uint64_t* d_compacted_keys = nullptr;
	CUDA_MALLOC(&d_compacted_keys, sizeof(uint64_t) * numberOfNodes);

	unsigned int* d_compaction_counter = nullptr;
	CUDA_MALLOC(&d_compaction_counter, sizeof(unsigned int));
	CUDA_MEMSET(d_compaction_counter, 0, sizeof(unsigned int));

	LaunchKernel(Kernel_CompactOccupiedKeys, keys.info.capacity,
		keys.info, d_compacted_keys, d_compaction_counter);

	thrust::device_ptr<uint64_t> d_sorted_keys(d_compacted_keys);

	thrust::sort(d_sorted_keys, d_sorted_keys + numberOfNodes);

	CUDA_MALLOC(&d_nodes, sizeof(HPOctreeNode) * numberOfNodes);
	CUDA_MEMSET(d_nodes, 0xFF, sizeof(HPOctreeNode) * numberOfNodes);

	LaunchKernel(Kernel_BuildOctreeNodes, numberOfNodes,
		leaf_map.info, d_nodes, thrust::raw_pointer_cast(d_sorted_keys), numberOfNodes, maxDepth);

	CUDA_SAFE_FREE(d_compacted_keys);
	CUDA_SAFE_FREE(d_compaction_counter);

	CUDA_TE(OctreeInitialiize);
}

void HPOctree::Terminate()
{
	keys.Terminate();
	leaf_map.Terminate();

	CUDA_SAFE_FREE(d_positions);
	numberOfPositions = 0;

	CUDA_SAFE_FREE(d_nodes);
	numberOfNodes = 0;
}

std::vector<NNS_Result> HPOctree::Search(const std::vector<float3>& h_queries)
{
	unsigned int numQueries = h_queries.size();
	std::vector<NNS_Result> h_results(numQueries);
	if (numQueries == 0) return h_results;

	float3* d_query_points;
	NNS_Result* d_results;

	CUDA_MALLOC(&d_query_points, sizeof(float3) * numQueries);
	CUDA_MALLOC(&d_results, sizeof(NNS_Result) * numQueries);

	CUDA_COPY_H2D(d_query_points, h_queries.data(), sizeof(float3) * numQueries);

	// NN 탐색 커널 실행!
	LaunchKernel(Kernel_NNSearch, numQueries,
		d_results,
		d_query_points,
		numQueries,
		d_nodes,
		numberOfNodes,
		d_positions, // 원본 데이터 포인트의 GPU 포인터
		domainAABB,
		domain_length,
		maxDepth
	);
	CUDA_SYNC();

	CUDA_COPY_D2H(h_results.data(), d_results, sizeof(NNS_Result) * numQueries);

	// h_results에 담긴 결과 확인...
	// ...

	cudaFree(d_query_points);
	cudaFree(d_results);

	return h_results;
}

std::vector<HPOctreeKey> HPOctree::Dump()
{
	unsigned int num_occupied_entries = 0;
	cudaMemcpy(&num_occupied_entries, keys.info.numberOfEntries, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if (num_occupied_entries == 0)
	{
		return {};
	}

	uint64_t* d_compacted_keys = nullptr;
	cudaMalloc(&d_compacted_keys, sizeof(uint64_t) * num_occupied_entries);

	unsigned int* d_compaction_counter = nullptr;
	cudaMalloc(&d_compaction_counter, sizeof(unsigned int));
	cudaMemset(d_compaction_counter, 0, sizeof(unsigned int));

	const size_t capacity = keys.info.capacity;
	const int threads_per_block = 256;
	const int blocks = (capacity + threads_per_block - 1) / threads_per_block;

	Kernel_CompactOccupiedKeys << <blocks, threads_per_block >> > (
		keys.info,
		d_compacted_keys,
		d_compaction_counter
		);
	cudaDeviceSynchronize();

	std::vector<uint64_t> h_compacted_codes(num_occupied_entries);
	cudaMemcpy(
		h_compacted_codes.data(),
		d_compacted_keys,
		sizeof(uint64_t) * num_occupied_entries,
		cudaMemcpyDeviceToHost
	);

	cudaFree(d_compacted_keys);
	cudaFree(d_compaction_counter);

	std::vector<HPOctreeKey> result_keys;
	result_keys.reserve(num_occupied_entries);

	for (uint64_t code : h_compacted_codes)
	{
		HPOctreeKey key;
		key.FromCode(code);
		result_keys.push_back(key);
	}

	return result_keys;
}
