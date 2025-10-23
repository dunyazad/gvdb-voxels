#include <Octree.cuh>
#include <IVisualDebugging.h>
#include <cassert>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/merge.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <string>
#include <algorithm>

using VD = IVisualDebugging;

template <int BLOCK_SIZE>
__global__ void computeAABB_reduction_kernel(const float3* inputPoints, int numElements, cuAABB* outputAABBs)
{
    extern __shared__ cuAABB s_aabbs[];
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    cuAABB thread_aabb;
    thread_aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    thread_aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = global_tid; i < numElements; i += grid_stride)
        thread_aabb.expand(inputPoints[i]);

    s_aabbs[tid] = thread_aabb;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            s_aabbs[tid].expand(s_aabbs[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        outputAABBs[blockIdx.x] = s_aabbs[0];
}

__host__ __device__
uint64_t expandBits(uint32_t v)
{
    uint64_t x = v & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__host__ __device__
uint32_t compactBits(uint64_t v)
{
    uint64_t x = v & 0x1249249249249249;
    x = (x ^ x >> 2) & 0x30c30c30c30c30c3;
    x = (x ^ x >> 4) & 0xf00f00f00f00f00f;
    x = (x ^ x >> 8) & 0xff0000ff0000ff;
    x = (x ^ x >> 16) & 0xffff00000000ffff;
    x = (x ^ x >> 32) & 0x1fffff;
    return (uint32_t)x;
}

__host__ __device__
uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    return (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
}

__host__ __device__
uint3 decodeMorton3D_Integer(uint64_t code)
{
    return { compactBits(code >> 2), compactBits(code >> 1), compactBits(code) };
}

__host__ __device__
uint3 WorldToInteger(float3 p, cuAABB aabb, int depth)
{
    if (depth <= 0) return { 0,0,0 };
    const float cells = float(1u << depth);
    const uint32_t max_idx = (1u << depth) - 1u;
    const float eps = 1e-7f;

    float3 size = { (aabb.max.x - aabb.min.x) + eps,
                    (aabb.max.y - aabb.min.y) + eps,
                    (aabb.max.z - aabb.min.z) + eps };

    float nx = (p.x - aabb.min.x) / size.x;
    float ny = (p.y - aabb.min.y) / size.y;
    float nz = (p.z - aabb.min.z) / size.z;

    nx = fminf(fmaxf(nx, 0.0f), 1.0f - 1e-8f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f - 1e-8f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f - 1e-8f);

    uint32_t ix = (uint32_t)floorf(nx * cells);
    uint32_t iy = (uint32_t)floorf(ny * cells);
    uint32_t iz = (uint32_t)floorf(nz * cells);

    ix = min(ix, max_idx);
    iy = min(iy, max_idx);
    iz = min(iz, max_idx);

    return { ix, iy, iz };
}

__host__ __device__
float3 IntegerToWorld(uint3 coords, cuAABB aabb, int depth)
{
    float3 pos;
    const float scale = float(1u << depth);
    float3 s = { aabb.max.x - aabb.min.x, aabb.max.y - aabb.min.y, aabb.max.z - aabb.min.z };
    float3 c = { (coords.x + 0.5f) / scale, (coords.y + 0.5f) / scale, (coords.z + 0.5f) / scale };
    pos.x = aabb.min.x + c.x * s.x;
    pos.y = aabb.min.y + c.y * s.y;
    pos.z = aabb.min.z + c.z * s.z;
    return pos;
}

struct PointToMortonFunctor {
    cuAABB aabb;
    float3 size;
    unsigned depth;
    float cells;
    uint32_t max_idx;

    __host__ __device__
        PointToMortonFunctor(const cuAABB& A, unsigned d)
        : aabb(A), depth(d),
        cells(float(1u << d)), max_idx((1u << d) - 1u),
        size({ (A.max.x - A.min.x) + 1e-7f,
               (A.max.y - A.min.y) + 1e-7f,
               (A.max.z - A.min.z) + 1e-7f }) {
    }

    __device__ uint64_t operator()(const float3& p) const {
        float nx = (p.x - aabb.min.x) / size.x;
        float ny = (p.y - aabb.min.y) / size.y;
        float nz = (p.z - aabb.min.z) / size.z;
        nx = fminf(fmaxf(nx, 0.0f), 1.0f - 1e-8f);
        ny = fminf(fmaxf(ny, 0.0f), 1.0f - 1e-8f);
        nz = fminf(fmaxf(nz, 0.0f), 1.0f - 1e-8f);

        uint32_t ix = (uint32_t)floorf(nx * cells);
        uint32_t iy = (uint32_t)floorf(ny * cells);
        uint32_t iz = (uint32_t)floorf(nz * cells);
        return morton3D(min(ix, max_idx), min(iy, max_idx), min(iz, max_idx));
    }
};

struct PointToAABB {
    __host__ __device__ cuAABB operator()(const float3& p) const { return { p,p }; }
};

struct MergeAABB {
    __host__ __device__ cuAABB operator()(const cuAABB& a, const cuAABB& b) const {
        cuAABB merged = a; merged.expand(b); return merged;
    }
};

struct select_first_functor {
    template <typename T>
    __host__ __device__ T operator()(const T& a, const T&) const { return a; }
};

__host__ __device__
cuAABB IntegerToAABB(uint64_t code, const cuAABB& world_aabb, int max_depth)
{
    int depth = 0;

#if defined(__CUDA_ARCH__)
    // GPU 코드
    if (code > 0)
        depth = (63 - __clzll(code)) / 3 + 1;
#else
    // CPU 코드
    if (code > 0)
    {
        unsigned long msb_index = 0;
#if defined(_MSC_VER)
        _BitScanReverse64(&msb_index, code);
#else
        msb_index = 63 - __builtin_clzll(code);
#endif
        depth = (msb_index / 3) + 1;
    }
#endif

    int shift = 3 * (max_depth - depth);
    uint64_t min_leaf_code = code << shift;

    uint3 min_coords = decodeMorton3D_Integer(min_leaf_code);
    uint32_t span = 1u << (max_depth - depth);
    uint3 max_coords = {
        min_coords.x + span,
        min_coords.y + span,
        min_coords.z + span
    };

    const float scale = float(1u << max_depth);
    float3 ws = {
        world_aabb.max.x - world_aabb.min.x,
        world_aabb.max.y - world_aabb.min.y,
        world_aabb.max.z - world_aabb.min.z
    };

    cuAABB box;
    box.min.x = world_aabb.min.x + (float)min_coords.x / scale * ws.x;
    box.min.y = world_aabb.min.y + (float)min_coords.y / scale * ws.y;
    box.min.z = world_aabb.min.z + (float)min_coords.z / scale * ws.z;

    box.max.x = world_aabb.min.x + (float)max_coords.x / scale * ws.x;
    box.max.y = world_aabb.min.y + (float)max_coords.y / scale * ws.y;
    box.max.z = world_aabb.min.z + (float)max_coords.z / scale * ws.z;

    const float eps = 1e-6f;
    box.max.x -= eps * (ws.x / scale);
    box.max.y -= eps * (ws.y / scale);
    box.max.z -= eps * (ws.z / scale);

    return box;
}

// === Internal node generation ===
__global__ void generateInternalNodes_kernel(
    const uint64_t* d_leaf_codes, int num_leaves, int max_depth, uint64_t* d_all_parent_codes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    uint64_t code = d_leaf_codes[tid];
    int offset = tid * max_depth;

    for (int d = 0; d < max_depth; ++d)
    {
        uint64_t parent = code >> 3;
        d_all_parent_codes[offset + d] = parent;
        code = parent;
        if (code == 0)
        {
            for (int j = d + 1; j < max_depth; ++j)
                d_all_parent_codes[offset + j] = 0;
            break;
        }
    }
}

__global__ void linkTree_kernel(
    const uint64_t* d_all_node_codes, int num_all_nodes,
    const uint64_t* d_leaf_codes, int num_leaves,
    const uint32_t* d_leaf_point_indices, OctreeNode* d_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_all_nodes) return;

    uint64_t my_code = d_all_node_codes[tid];
    const uint64_t* leaf_it = thrust::lower_bound(thrust::seq, d_leaf_codes, d_leaf_codes + num_leaves, my_code);
    int leaf_idx = leaf_it - d_leaf_codes;

    if (leaf_it != d_leaf_codes + num_leaves && *leaf_it == my_code)
    {
        d_nodes[tid].child_mask = 0;
        d_nodes[tid].data_or_child_idx = d_leaf_point_indices[leaf_idx];
    }
    else
    {
        uint8_t mask = 0; uint32_t first_child = (uint32_t)-1;
        for (int k = 0; k < 8; ++k)
        {
            uint64_t child_code = (my_code << 3) | k;
            const uint64_t* it = thrust::lower_bound(thrust::seq, d_all_node_codes, d_all_node_codes + num_all_nodes, child_code);
            int idx = it - d_all_node_codes;
            if (it != d_all_node_codes + num_all_nodes && *it == child_code)
            {
                mask |= (1 << k);
                if (first_child == (uint32_t)-1) first_child = idx;
            }
        }
        d_nodes[tid].child_mask = mask;
        d_nodes[tid].data_or_child_idx = first_child;
    }
}

void Octree::Build(std::vector<float3> h_points, unsigned int MAX_DEPTH)
{
    VD::ClearAll();
    assert(MAX_DEPTH <= 21 && "Morton code supports up to 21 bits!");

    int N = static_cast<int>(h_points.size());
    thrust::device_vector<float3> d_points = h_points;

    // --- [1] 전체 AABB 계산 ---
    cuAABB aabb = thrust::transform_reduce(
        thrust::device,
        d_points.begin(), d_points.end(),
        PointToAABB(),
        cuAABB(),
        MergeAABB()
    );

    // AABB를 그대로 사용, cubic 변환 제거
    cuAABB world_aabb = aabb;

    // 혹시라도 floating precision 여유 확보
    const float epsilon = 1e-5f;
    world_aabb.min.x -= epsilon;
    world_aabb.min.y -= epsilon;
    world_aabb.min.z -= epsilon;
    world_aabb.max.x += epsilon;
    world_aabb.max.y += epsilon;
    world_aabb.max.z += epsilon;

    // --- [4] Morton 코드 계산 ---
    thrust::device_vector<uint64_t> codes(N);
    thrust::device_vector<uint32_t> idx(N);
    thrust::sequence(idx.begin(), idx.end());

    PointToMortonFunctor f(world_aabb, MAX_DEPTH);
    thrust::transform(d_points.begin(), d_points.end(), codes.begin(), f);
    thrust::sort_by_key(codes.begin(), codes.end(), idx.begin());

    // --- [5] 고유 리프 찾기 ---
    thrust::device_vector<uint64_t> uniq_codes(N);
    thrust::device_vector<uint32_t> uniq_idx(N);
    auto new_end = thrust::reduce_by_key(
        codes.begin(), codes.end(),
        idx.begin(),
        uniq_codes.begin(),
        uniq_idx.begin(),
        thrust::equal_to<uint64_t>(),
        select_first_functor()
    );

    int num_leaves = new_end.first - uniq_codes.begin();
    uniq_codes.resize(num_leaves);
    uniq_idx.resize(num_leaves);

    std::cout << "Unique leaf count: " << num_leaves << std::endl;

    // --- [6] 시각화: 리프 노드 (청록색) ---
    thrust::host_vector<uint64_t> h_codes = uniq_codes;
    for (int i = 0; i < h_codes.size(); ++i)
    {
        cuAABB leaf_box = IntegerToAABB(h_codes[i], world_aabb, MAX_DEPTH);
        VD::AddWiredBox("leaf", leaf_box, Color::cyan());
    }

    // --- [7] 전체 cubic 시각화 (빨간 박스) ---
    VD::AddWiredBox("world_aabb", world_aabb, Color::red());

    return;

    // ======================================================================
    // [8] 내부 노드 생성
    // ======================================================================
    thrust::device_vector<uint64_t> d_parents(num_leaves * MAX_DEPTH);
    int TPB = 256;
    int BPG = (num_leaves + TPB - 1) / TPB;

    generateInternalNodes_kernel << <BPG, TPB >> > (
        thrust::raw_pointer_cast(uniq_codes.data()),
        num_leaves,
        MAX_DEPTH,
        thrust::raw_pointer_cast(d_parents.data())
        );

    thrust::sort(d_parents.begin(), d_parents.end());
    auto pend = thrust::unique(d_parents.begin(), d_parents.end());
    d_parents.resize(pend - d_parents.begin());
    int num_internal = d_parents.size();
    std::cout << "Internal nodes: " << num_internal << std::endl;

    // --- [9] 내부 노드 시각화 (노란색) ---
    thrust::host_vector<uint64_t> h_int = d_parents;
    for (int i = 0; i < h_int.size(); ++i)
    {
        cuAABB node_box = IntegerToAABB(h_int[i], world_aabb, MAX_DEPTH);
        VD::AddWiredBox("internal_" + std::to_string(i), node_box, Color::yellow());
    }

    // ======================================================================
    // [10] 리프 + 내부 노드 병합
    // ======================================================================
    this->d_node_codes.resize(num_leaves + num_internal);
    auto mend = thrust::merge(
        uniq_codes.begin(), uniq_codes.end(),
        d_parents.begin(), d_parents.end(),
        this->d_node_codes.begin()
    );
    auto uend = thrust::unique(this->d_node_codes.begin(), mend);
    this->d_node_codes.resize(uend - this->d_node_codes.begin());
    int num_total = this->d_node_codes.size();

    // ======================================================================
    // [11] 트리 연결
    // ======================================================================
    this->d_nodes.resize(num_total);
    BPG = (num_total + TPB - 1) / TPB;

    linkTree_kernel << <BPG, TPB >> > (
        thrust::raw_pointer_cast(this->d_node_codes.data()), num_total,
        thrust::raw_pointer_cast(uniq_codes.data()), num_leaves,
        thrust::raw_pointer_cast(uniq_idx.data()),
        thrust::raw_pointer_cast(this->d_nodes.data())
        );
    cudaDeviceSynchronize();

    std::cout << "Total nodes (internal + leaf): " << num_total << std::endl;
    std::cout << "Octree build complete.\n";
}
