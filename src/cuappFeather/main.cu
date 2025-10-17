#pragma warning(disable : 4819)
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>
#pragma comment(lib, "nvapi64.lib")

#include <cuda_common.cuh>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>

#include <Serialization.hpp>

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"

#include "ThrustHashMap.hpp"

CUDAInstance::CUDAInstance()
{
}

CUDAInstance::~CUDAInstance()
{
    vhm.Terminate();
    d_input.Terminate();
    h_mesh.Terminate();
    d_mesh.Terminate();
}

void CUDAInstance::ProcessHalfEdgeMesh(const std::string& filename)
{
    h_mesh.DeserializePLY(filename);
}

//HostPointCloud<PointCloudProperty> CUDAInstance::ProcessPointCloud(const HostPointCloud<PointCloudProperty>& input, float voxelSize, unsigned int occupyOffset)
//{
//    CUDA_TS(ProcessPointCloud);
//
//    h_input = input;
//    d_input = h_input;
//
//    d_input.Count();
//
//    vhm.Initialize(voxelSize, d_input.numberOfPoints * 8, 128);
//
//    vhm.Occupy(d_input, occupyOffset);
//
//    auto result = vhm.Serialize();
//    result.CompactValidPoints();
//
//    CUDA_TS(MarchingCubes);
//    vhm.MarchingCubes(d_mesh);
//    CUDA_TE(MarchingCubes);
//
//    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);
//
//    d_mesh.RemoveIsolatedVertices();
//
//    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);
//
//    h_mesh.CopyFromDevice(d_mesh);
//
//    //h_mesh.SerializePLY("../../res/3D/host_mesh.ply");
//
//    printf("h_mesh.numberOfPoints : %d\n", h_mesh.numberOfPoints);
//
//    vhm.Terminate();
//
//    for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
//    {
//        if (UINT32_MAX == h_mesh.vertexToHalfEdge[i])
//        {
//            printf("Vertex : %llu has no half-edge.\n", i);
//        }
//    }
//
//    CUDA_TE(ProcessPointCloud);
//
//    //ThrustHostHalfEdgeMesh th_mesh;
//    //th_mesh.DeserializePLY("../../res/3D/HostHalfEdgeMesh_Normal.ply");
//
//    //ThrustDeviceHalfEdgeMesh td_mesh;
//    //td_mesh.CopyFromHost(th_mesh);
//
//    //td_mesh.BuildHalfEdges();
//
//    //th_mesh.CopyFromDevice(td_mesh);
//
//    //th_mesh.SerializePLY("../../res/3D/Result.ply");
//
//    return result;
//}

void CUDAInstance::ProcessPointCloud(float voxelSize, unsigned int occupyOffset)
{
    d_input = h_input;

    vhm.Initialize(voxelSize, d_input.numberOfPoints * 32, 32);

    CUDA_TS(ProcessPointCloud);
    vhm.Occupy(d_input, occupyOffset);

    CUDA_TS(MarchingCubes);
    vhm.MarchingCubes(d_mesh);
    CUDA_TE(MarchingCubes);

    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    d_mesh.RemoveIsolatedVertices();



    CUDA_TE(ProcessPointCloud);
    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    h_mesh.CopyFromDevice(d_mesh);

    printf("h_mesh.numberOfPoints : %d\n", h_mesh.numberOfPoints);

    vhm.Terminate();

}
//
//__global__ void FindIntersectionPointsKernel(
//	const float3* d_points,
//	int numPoints,
//	float3 planePosition,
//	float3 planeNormal, // Assumed to be a unit std::vector
//	float distanceThreshold,
//	float3* d_intersectionPoints,
//	int* d_counter)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (idx >= numPoints)
//	{
//		return;
//	}
//
//	// Get the current point
//	float3 point = d_points[idx];
//
//	// Calculate std::vector from a point on the plane to the current point
//	float3 vec = make_float3(point.x - planePosition.x,
//		point.y - planePosition.y,
//		point.z - planePosition.z);
//
//	// Calculate the signed perpendicular distance to the plane
//	// distance = dot(vec, planeNormal)
//	float distance = vec.x * planeNormal.x + vec.y * planeNormal.y + vec.z * planeNormal.z;
//
//	// Check if the absolute distance is within the threshold
//	if (fabsf(distance) <= distanceThreshold)
//	{
//		// Atomically increment the counter to get a unique write index
//		int output_idx = atomicAdd(d_counter, 1);
//		// Store the point in the output buffer
//		d_intersectionPoints[output_idx] = point;
//	}
//}
//
//std::vector<float3> CUDAInstance::FindIntersectionPoints(
//    HostPointCloud<PointCloudProperty>& h_pointCloud,
//    float3 planePosition,
//    float3 planeNormal,
//    float distanceThreshold)
//{
//	// 1. Get host data and handle empty input
//	const int numPoints = h_pointCloud.numberOfPoints;
//	if (numPoints == 0)
//	{
//		return {};
//	}
//	const float3* h_points = h_pointCloud.positions;
//
//	// 2. Allocate memory on the GPU device
//	float3* d_points = nullptr;
//	float3* d_intersectionPoints = nullptr;
//	int* d_counter = nullptr;
//
//	CUDA_CHECK(cudaMalloc(&d_points, numPoints * sizeof(float3)));
//	// Allocate output buffer for the worst case (all points intersect)
//	CUDA_CHECK(cudaMalloc(&d_intersectionPoints, numPoints * sizeof(float3)));
//	CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
//
//	// 3. Copy data from host to device
//	CUDA_CHECK(cudaMemcpy(d_points, h_points, numPoints * sizeof(float3), cudaMemcpyHostToDevice));
//	CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
//
//	// Normalize the plane normal on the host once for efficiency
//	float3 unitNormal = normalize(planeNormal);
//
//	CUDA_TS(FindIntersectionPointsKernel);
//
//	LaunchKernel(FindIntersectionPointsKernel, numPoints,
//		d_points,
//		numPoints,
//		planePosition,
//		unitNormal,
//		distanceThreshold,
//		d_intersectionPoints,
//		d_counter);
//
//	CUDA_TE(FindIntersectionPointsKernel);
//
//	// Check for any errors during kernel launch
//	CUDA_CHECK(cudaGetLastError());
//	// Wait for the kernel to finish execution before proceeding
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	// 5. Copy results from device back to host
//	int intersectionCount = 0;
//	CUDA_CHECK(cudaMemcpy(&intersectionCount, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
//
//	std::vector<float3> h_intersectionPoints;
//	if (intersectionCount > 0)
//	{
//		h_intersectionPoints.resize(intersectionCount);
//		CUDA_CHECK(cudaMemcpy(h_intersectionPoints.data(), d_intersectionPoints,
//			intersectionCount * sizeof(float3), cudaMemcpyDeviceToHost));
//	}
//
//	// 6. Free all allocated device memory
//	CUDA_CHECK(cudaFree(d_points));
//	CUDA_CHECK(cudaFree(d_intersectionPoints));
//	CUDA_CHECK(cudaFree(d_counter));
//
//	// 7. Return the resulting std::vector of points
//	return h_intersectionPoints;
//}




struct PointInfo
{
    float3 point;
    float distance;
};

__global__ void ProjectIntersectionPoints_Kernel(
    const float3* d_points,
    int numPoints,
    float3 planePosition,
    float3 planeNormal,
    float distanceThreshold,
    float3* d_projectedPoints,
    int* d_counter)
{
    extern __shared__ PointInfo s_cache[];

    __shared__ int s_local_hit_count;
    __shared__ int s_global_base_idx;

    int const tid = threadIdx.x;
    int const gid = blockIdx.x * blockDim.x + tid;
    int const block_size = blockDim.x;

    if (tid == 0)
    {
        s_local_hit_count = 0;
    }
    __syncthreads();

    if (gid < numPoints)
    {
        float3 point = d_points[gid];
        float3 vec = make_float3(point.x - planePosition.x, point.y - planePosition.y, point.z - planePosition.z);
        float distance = dot(vec, planeNormal);

        if (fabsf(distance) <= distanceThreshold)
        {
            int local_idx = atomicAdd(&s_local_hit_count, 1);
            if (local_idx < block_size)
            {
                s_cache[local_idx].point = point;
                s_cache[local_idx].distance = distance;
            }
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        int hits_in_this_block = std::min(s_local_hit_count, block_size);
        if (hits_in_this_block > 0)
        {
            s_global_base_idx = atomicAdd(d_counter, hits_in_this_block);
        }
    }
    __syncthreads();

    int const hits_to_copy = std::min(s_local_hit_count, block_size);

    for (int i = tid; i < hits_to_copy; i += block_size)
    {
        PointInfo info = s_cache[i];

        float3 projected_point = make_float3(
            info.point.x - info.distance * planeNormal.x,
            info.point.y - info.distance * planeNormal.y,
            info.point.z - info.distance * planeNormal.z
        );

        d_projectedPoints[s_global_base_idx + i] = projected_point;
    }
}

std::vector<float3> CUDAInstance::FindIntersectionPoints(
    HostPointCloud<PointCloudProperty>& h_pointCloud,
    float3 planePosition,
    float3 planeNormal,
    float distanceThreshold)
{
    const int numPoints = h_pointCloud.numberOfPoints;
    if (numPoints == 0) { return {}; }
    const float3* h_points = h_pointCloud.positions;

    try
    {
        thrust::device_vector<float3> d_points(h_points, h_points + numPoints);
        thrust::device_vector<float3> d_intersectionPoints(numPoints);
        thrust::device_vector<int> d_counter(1, 0);

        float3 unitNormal = normalize(planeNormal);

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedMemSize = threadsPerBlock * sizeof(float3);

        CUDA_TS(Find);
        nvtxRangePushA("FindIntersectionPoints_FinalKernel");
        ProjectIntersectionPoints_Kernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (
            thrust::raw_pointer_cast(d_points.data()),
            numPoints,
            planePosition,
            unitNormal,
            distanceThreshold,
            thrust::raw_pointer_cast(d_intersectionPoints.data()),
            thrust::raw_pointer_cast(d_counter.data())
            );
        CUDA_SYNC();
        nvtxRangePop();
        CUDA_TE(Find);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
            return {};
        }
        cudaDeviceSynchronize();

        int intersectionCount = d_counter[0];
        std::vector<float3> h_intersectionPoints;
        if (intersectionCount > 0 && intersectionCount <= numPoints)
        {
            h_intersectionPoints.resize(intersectionCount);
            thrust::copy(d_intersectionPoints.begin(), d_intersectionPoints.begin() + intersectionCount, h_intersectionPoints.begin());
        }

        return h_intersectionPoints;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "Exception: %s\n", e.what());
        return {};
    }
}


double benchmarkInsert(HashMap<double, int>& map, const std::vector<double>& keys, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();
    map.insert(keys, values);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    return keys.size() / std::chrono::duration<double>(end - start).count();
}

double benchmarkFind(HashMap<double, int>& map, const std::vector<double>& keys) {
    std::vector<int> results;
    auto start = std::chrono::high_resolution_clock::now();
    map.find(keys, results);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    return keys.size() / std::chrono::duration<double>(end - start).count();
}

double benchmarkRemove(HashMap<double, int>& map, const std::vector<double>& keys) {
    auto start = std::chrono::high_resolution_clock::now();
    map.remove(keys);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    return keys.size() / std::chrono::duration<double>(end - start).count();
}

void printBenchmarkGraph(double insert_d, double find_d, double remove_d) {
    struct Op {
        std::string name;
        double value;
    };

    std::vector<Op> ops = {
        {"Insert", insert_d},
        {"Find  ", find_d},
        {"Remove", remove_d}
    };

    // 그래프 스케일 결정 (막대 최대 길이 50)
    double max_val = std::max({ insert_d, find_d, remove_d });
    double scale = max_val / 50.0;

    std::cout << "\nHashMap Benchmark (Throughput ops/sec)\n";
    std::cout << "-------------------------------------\n";

    for (auto& op : ops) {
        int bar_len = static_cast<int>(op.value / scale);
        std::cout << op.name << ": ";
        for (int i = 0; i < bar_len; ++i) std::cout << "#";
        std::cout << " " << op.value << "\n";
    }

    std::cout << "-------------------------------------\n";
}

void CUDAInstance::Test()
{
    {
        const size_t N = 10'000'000; // 1천만 elements
        std::vector<double> keys(N), keys_float(N);
        std::vector<int> values(N);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(0.0, 1e6);

        for (size_t i = 0; i < N; i++) {
            keys[i] = dist(rng); // double Key
            keys_float[i] = static_cast<float>(keys[i]); // float Key
            values[i] = static_cast<int>(i);
        }

        // --- double Key 벤치마크 ---
        HashMap<double, int> map_double(1 << 24);
        double insert_d = benchmarkInsert(map_double, keys, values);
        double find_d = benchmarkFind(map_double, keys);
        double remove_d = benchmarkRemove(map_double, keys);

        std::cout << "Double Key results:\n";
        std::cout << "Insert: " << insert_d << " ops/sec\n";
        std::cout << "Find  : " << find_d << " ops/sec\n";
        std::cout << "Remove: " << remove_d << " ops/sec\n";

        printBenchmarkGraph(insert_d, find_d, remove_d);
    }

    {
        PLYFormat ply;
        ply.Deserialize("D:\\Debug\\PLY\\input.ply");

        auto N = ply.GetPoints().size() / 3;
        thrust::host_vector<float3> h_positions(N);
        memcpy(h_positions.data(), ply.GetPoints().data(), sizeof(float3) * N);
        thrust::device_vector<float3> d_positions(h_positions);


    }
}
