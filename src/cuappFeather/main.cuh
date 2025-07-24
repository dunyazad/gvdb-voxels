#include <iostream>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

#include <cuda_common.cuh>

#include <PointCloud.cuh>
#include <HashMap.hpp>
#include <BitVolume.cuh>
#include <DenseGrid.hpp>
#include <VoxelHashMap.cuh>
#include <SCVoxelHashMap.cuh>
#include <HalfEdgeMesh.cuh>

#include <Eigen/Core>
#include <Eigen/Dense>
namespace Eigen {
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

#define alog(...) printf("\033[38;5;1m\033[48;5;15m(^(OO)^) /V/\033[0m\t" __VA_ARGS__)
#define alogt(tag, ...) printf("\033[38;5;1m\033[48;5;15m [%d] (^(OO)^) /V/\033[0m\t" tag, __VA_ARGS__)


#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)

#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);

#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[%s] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);


__host__ __device__ float3 rgb_to_hsv(uchar3 rgb);
__host__ __device__ uchar3 hsv_to_rgb(float3 hsv);

__host__ __device__ uint64_t IndexToKey(uint3 index);
__host__ __device__ uint3 KeyToIndex(uint64_t key);

__device__ uint64_t D_ToKey(float3 p, float resolution = 0.0001f);
__device__ float3 D_FromKey(uint64_t key, float resolution = 0.0001f);
__host__ uint64_t H_ToKey(const float3& p, float resolution = 0.0001f);
__host__ float3 H_FromKey(uint64_t key, float resolution = 0.0001f);

class Texture;

class CUDAInstance
{
public:
    CUDAInstance();
    ~CUDAInstance();

    DevicePointCloud d_input;
    SCVoxelHashMap vhm;
    HostHalfEdgeMesh h_mesh;
    DeviceHalfEdgeMesh d_mesh;

    HostPointCloud ProcessPointCloud(const HostPointCloud& h_input);
    void ProcessHalfEdgeMesh(const string& filename);
};
