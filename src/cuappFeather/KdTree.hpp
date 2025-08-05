#include <cuda_common.cuh>

struct KdNode
{
    float3 min, max;    // bounding box
    int left, right;    // 자식 노드 인덱스 (-1: 없음)
    int start, end;     // points 범위 (indices 배열)
    int axis;           // 분할축 (0,1,2)
    float splitValue;   // 분할값
    bool isLeaf;        // 리프 여부
};


// axis별 value 얻는 헬퍼
__host__ __device__
float get_by_axis(const float3& p, int axis)
{
    if (axis == 0) return p.x;
    if (axis == 1) return p.y;
    return p.z;
}

__device__ void compute_aabb(float3* positions, int* indices, int start, int end, float3& out_min, float3& out_max)
{
    out_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    out_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = start; i < end; ++i)
    {
        float3 p = positions[indices[i]];
        out_min.x = min(out_min.x, p.x);
        out_min.y = min(out_min.y, p.y);
        out_min.z = min(out_min.z, p.z);
        out_max.x = max(out_max.x, p.x);
        out_max.y = max(out_max.y, p.y);
        out_max.z = max(out_max.z, p.z);
    }
}

__global__ void cuda_kdtree_build(
    float3* positions, int* indices, KdNode* nodes,
    int N, int* node_counter)
{
    // Single thread에서 큐로 트리 빌드 (미니멀/단순 예제)
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    struct StackEntry { int start, end, parent, isLeft; };
    StackEntry stack[64];
    int sp = 0;
    stack[sp++] = { 0, N, -1, 0 };

    int curr_node = 0;

    while (sp > 0)
    {
        StackEntry entry = stack[--sp];
        int start = entry.start, end = entry.end;
        int count = end - start;

        // AABB
        float3 min, max;
        compute_aabb(positions, indices, start, end, min, max);

        // axis 선택
        float3 extent = make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
        int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 : (extent.y > extent.z ? 1 : 2);

        bool isLeaf = (count <= 8);

        // 노드 작성
        int this_idx = atomicAdd(node_counter, 1);
        nodes[this_idx].min = min;
        nodes[this_idx].max = max;
        nodes[this_idx].start = start;
        nodes[this_idx].end = end;
        nodes[this_idx].axis = axis;
        nodes[this_idx].isLeaf = isLeaf;
        nodes[this_idx].left = nodes[this_idx].right = -1;

        // parent 연결
        if (entry.parent >= 0)
        {
            if (entry.isLeft)
                nodes[entry.parent].left = this_idx;
            else
                nodes[entry.parent].right = this_idx;
        }

        if (!isLeaf)
        {
            // axis 기준 median 찾기: selection sort (미니멀, 실제론 thrust로 대체 권장)
            for (int i = start; i < end - 1; ++i)
            {
                int min_j = i;
                float min_v = get_by_axis(positions[indices[i]], axis);
                for (int j = i + 1; j < end; ++j)
                {
                    float v = get_by_axis(positions[indices[j]], axis);
                    if (v < min_v)
                    {
                        min_j = j;
                        min_v = v;
                    }
                }
                if (min_j != i)
                {
                    int tmp = indices[i];
                    indices[i] = indices[min_j];
                    indices[min_j] = tmp;
                }
            }
            int mid = start + count / 2;
            nodes[this_idx].splitValue = get_by_axis(positions[indices[mid]], axis);

            // 스택 push
            stack[sp++] = { start, mid, this_idx, 1 };   // left
            stack[sp++] = { mid, end, this_idx, 0 };     // right
        }
    }
}
