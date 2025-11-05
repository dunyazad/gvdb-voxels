#pragma once

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//#include <device_functions.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <cuda_vector_math.cuh>
#include <marching_cubes_constants.cuh>
#include <surface_functions.h>

// Vector containers
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Algorithms
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>


//#include <cublas_v2.h>

#include <Serialization.hpp>

#include <TypeDefinitions.h>

#include "gvdb.h"

#define alog(...) printf("\033[38;5;1m\033[48;5;15m(^(OO)^) /V/\033[0m\t" __VA_ARGS__)
#define alogt(tag, ...) printf("\033[38;5;1m\033[48;5;15m [%d] (^(OO)^) /V/\033[0m\t" tag, __VA_ARGS__)

typedef char i8;
typedef short i16;
typedef int i32;
typedef long i64;

typedef unsigned char ui8;
typedef unsigned short ui16;
typedef unsigned int ui32;
typedef unsigned long ui64;

typedef float f32;
typedef double f64;

#ifndef MIN_MAX_DEFINITIONS
#define MIN_MAX_DEFINITIONS
#define i8_max  (INT8_MAX)
#define i8_min  (INT8_MIN)
#define i16_max (INT16_MAX)
#define i16_min (INT16_MIN)
#define i32_max (INT32_MAX)
#define i32_min (INT32_MIN)
#define i64_max (INT64_MAX)
#define i64_min (INT64_MIN)

#define ui8_max (UINT8_MAX)
#define ui16_max (UINT16_MAX)
#define ui32_max (UINT32_MAX)
#define ui64_max (UINT64_MAX)

#define f32_max (FLT_MAX)
#define f32_min (FLT_MIN)
#define f64_max (DBL_MAX)
#define f64_min (DBL_MIN)
#endif

#ifndef __CUSTOM_DEFINITIONS_FOR_CUDA__
#define __CUSTOM_DEFINITIONS_FOR_CUDA__
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA error %s (%d): %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            assert(false); \
        } \
    } while (0)

#ifndef LaunchKernel
#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)

#define LaunchKernel2D_16x16(KERNEL, NX, NY, ...) { \
    nvtxRangePushA(#KERNEL); \
    dim3 NOT(16, 16); \
    dim3 NOB((NX + NOT.x - 1) / NOT.x, (NY + NOT.y - 1) / NOT.y); \
    KERNEL<<<NOB, NOT>>>(__VA_ARGS__); \
    nvtxRangePop(); \
}
#define LaunchKernel2D(KERNEL, NX, NY, ...) LaunchKernel2D_16x16(KERNEL, NX, NY, __VA_ARGS__)

#define LaunchLambdaKernel_256(LAMBDA_KERNEL, NOE, ...) { nvtxRangePushA(#LAMBDA_KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; LaunchLambdaKernelTemplate<<<NOB, NOT>>>(LAMBDA_KERNEL, NOE, __VA_ARGS__); nvtxRangePop(); }
#define LaunchLambdaKernel_512(LAMBDA_KERNEL, NOE, ...) { nvtxRangePushA(#LAMBDA_KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; LaunchLambdaKernelTemplate<<<NOB, NOT>>>(LAMBDA_KERNEL, NOE, __VA_ARGS__); nvtxRangePop(); }
#define LaunchLambdaKernel(LAMBDA_KERNEL, NOE, ...) LaunchLambdaKernel_512(LAMBDA_KERNEL, NOE, __VA_ARGS__)

#define LaunchLambdaKernel2D_16x16(LAMBDA_KERNEL, NX, NY, ...) { \
    nvtxRangePushA(#LAMBDA_KERNEL); \
    dim3 NOT(16, 16); \
    dim3 NOB((NX + NOT.x - 1) / NOT.x, (NY + NOT.y - 1) / NOT.y); \
    LaunchLambdaKernelTemplate2D<<<NOB, NOT>>>(LAMBDA_KERNEL, NX, NY, __VA_ARGS__); \
    nvtxRangePop(); \
}
#define LaunchLambdaKernel2D(LAMBDA_KERNEL, NX, NY, ...) LaunchLambdaKernel2D_16x16(LAMBDA_KERNEL, NX, NY, __VA_ARGS__)
#endif

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[%s] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

#ifndef CUDA_MALLOC
#define CUDA_MALLOC(ptr, size) cudaMalloc(ptr, size);
#endif

#ifndef CUDA_FREE
#define CUDA_FREE(ptr) cudaFree(ptr);
#endif

#ifndef CUDA_SAFE_FREE
#define CUDA_SAFE_FREE(ptr) { if(ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; } }
//#define CUDA_SAFE_FREE(ptr) \
//    do { \
//        if (ptr) { \
//            cudaFree(ptr); \
//            ptr = nullptr; \
//        } \
//    } while(0)
#endif

#ifndef CUDA_MEMSET
#define CUDA_MEMSET(ptr, value, size) cudaMemset(ptr, value, size);
#endif

#ifndef CUDA_COPY_D2D
#define CUDA_COPY_D2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice);
#endif

#ifndef CUDA_COPY_D2H
#define CUDA_COPY_D2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
#endif

#ifndef CUDA_COPY_H2D
#define CUDA_COPY_H2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
#endif

#ifndef CUDA_COPY_H2H
#define CUDA_COPY_H2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToHost);
#endif

#ifndef CUDA_SYNC
#define CUDA_SYNC() cudaDeviceSynchronize();
#endif

#ifndef RAW_PTR
#define RAW_PTR(x) (thrust::raw_pointer_cast((x).data()))
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef DEG2RAD
#define DEG2RAD (PI/180)
#endif

#ifndef RAD2DEG
#define RAD2DEG (180/PI)
#endif

#ifndef XYZ
#define XYZ(v) (v).x, (v).y, (v).z
#endif
#ifndef XYZW
#define XYZW(v) (v).x, (v).y, (v).z, (v).w
#endif
#endif

struct cuAABB
{
    float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    __host__ __device__ __forceinline__
    cuAABB GetDomainAABB()
    {
        auto delta = max - min;
        float hl = fmaxf(fmaxf(delta.x, delta.y), delta.z);
        auto center = (min + max) * 0.5f;
        float half_length = hl * 0.5f;
        return {
            {center.x - half_length, center.y - half_length, center.z - half_length},
            {center.x + half_length, center.y + half_length, center.z + half_length}
        };
    }

    __host__ __device__ __forceinline__
    bool Contains(const float3& p) const
    {
        return
            p.x >= min.x && p.x <= max.x &&
            p.y >= min.y && p.y <= max.y &&
            p.z >= min.z && p.z <= max.z;
    }

    __host__ __device__ __forceinline__
        void expand(const float3& p)
    {
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);

        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    __host__ __device__ __forceinline__
        void expand(const cuAABB& b)
    {
        min.x = fminf(min.x, b.min.x);
        min.y = fminf(min.y, b.min.y);
        min.z = fminf(min.z, b.min.z);

        max.x = fmaxf(max.x, b.max.x);
        max.y = fmaxf(max.y, b.max.y);
        max.z = fmaxf(max.z, b.max.z);
    }

    __host__ __device__ __forceinline__
        bool contains(const float3& p) const
    {
        return (p.x >= min.x && p.x <= max.x &&
            p.y >= min.y && p.y <= max.y &&
            p.z >= min.z && p.z <= max.z);
    }

    __host__ __device__ __forceinline__
        float3 center() const
    {
        return make_float3((min.x + max.x) * 0.5f,
            (min.y + max.y) * 0.5f,
            (min.z + max.z) * 0.5f);
    }

    __host__ __device__ float3 extent() const
    {
        return make_float3(
            max.x - min.x,
            max.y - min.y,
            max.z - min.z);
    }

    __host__ __device__ float volume() const
    {
        float3 e = extent();
        return fabsf(e.x * e.y * e.z);
    }

    __host__ __device__ __forceinline__
    bool intersects(const cuAABB& other) const
    {
        return
            (min.x <= other.max.x && max.x >= other.min.x) &&
            (min.y <= other.max.y && max.y >= other.min.y) &&
            (min.z <= other.max.z && max.z >= other.min.z);
    }

    __host__ __device__ __forceinline__ static
    cuAABB merge(const cuAABB& a, const cuAABB& b)
    {
        cuAABB out;
        out.min = make_float3(fminf(a.min.x, b.min.x),
            fminf(a.min.y, b.min.y),
            fminf(a.min.z, b.min.z));
        out.max = make_float3(fmaxf(a.max.x, b.max.x),
            fmaxf(a.max.y, b.max.y),
            fmaxf(a.max.z, b.max.z));
        return out;
    }

    __host__ __device__ __forceinline__ static float Distance2(const cuAABB& aabb, const float3& p)
    {
        float3 clamped;
        clamped.x = fmaxf(aabb.min.x, fminf(p.x, aabb.max.x));
        clamped.y = fmaxf(aabb.min.y, fminf(p.y, aabb.max.y));
        clamped.z = fmaxf(aabb.min.z, fminf(p.z, aabb.max.z));
        return length2(p - clamped);
    }

    __host__ __device__ __forceinline__
        operator AABB() const
    {
        AABB aabb;
        aabb.min = glm::vec3(min.x, min.y, min.z);
        aabb.max = glm::vec3(max.x, max.y, max.z);
        return aabb;
    }
};

template <typename Functor, typename... Args>
__global__ void LaunchLambdaKernelTemplate(Functor func, unsigned int NOE, Args... args)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NOE) return;

    func(tid, args...);
}

template <typename Kernel, typename... Args>
__global__ void LaunchLambdaKernelTemplate2D(Kernel kernel, unsigned int nx, unsigned int ny, Args... args)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny)
    {
        kernel(i, j, args...);
    }
}

#ifdef __CUDACC__
__device__ inline void atomicMinF(float* addr, float val)
{
    int* addr_as_i = reinterpret_cast<int*>(addr);
    int old = *addr_as_i, assumed;
    do
    {
        assumed = old;
        float f = __int_as_float(assumed);
        if (val >= f) break;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (old != assumed);
}

__device__ inline void atomicMinF(float* addr, float val, int* idx, int myIdx)
{
    int* intAddr = reinterpret_cast<int*>(addr);
    int old = *intAddr, assumed;

    do
    {
        assumed = old;
        float oldVal = __int_as_float(assumed);

        if (val >= oldVal) break;

        old = atomicCAS(intAddr, assumed, __float_as_int(val));

        if (old == assumed)
        {
            atomicExch(idx, myIdx);
            break;
        }
    } while (true);
}

__device__ inline void atomicMaxF(float* addr, float val)
{
    int* addr_as_i = reinterpret_cast<int*>(addr);
    int old = *addr_as_i, assumed;
    do
    {
        assumed = old;
        float f = __int_as_float(assumed);
        if (val <= f) break;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (old != assumed);
}

__device__ inline void atomicMaxF(float* addr, float val, int* idx, int myIdx)
{
    int* intAddr = reinterpret_cast<int*>(addr);
    int old = *intAddr, assumed;

    do
    {
        assumed = old;
        float oldVal = __int_as_float(assumed);

        if (val <= oldVal) break;

        old = atomicCAS(intAddr, assumed, __float_as_int(val));

        if (old == assumed)
        {
            atomicExch(idx, myIdx);
            break;
        }
    } while (true);
}

__device__ __forceinline__ void atomicMinFloat3(float3& addr, const float3& val)
{
    atomicMinF(&addr.x, val.x);
    atomicMinF(&addr.y, val.y);
    atomicMinF(&addr.z, val.z);
}

__device__ __forceinline__ void atomicMaxFloat3(float3& addr, const float3& val)
{
    atomicMaxF(&addr.x, val.x);
    atomicMaxF(&addr.y, val.y);
    atomicMaxF(&addr.z, val.z);
}
#endif

__host__ __device__ inline float PointTriangleDistance2(const float3& p, const float3& a, const float3& b, const float3& c)
{
    // From "Real-Time Collision Detection" by Christer Ericson
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);

    if (d1 <= 0.0f && d2 <= 0.0f) return dot(ap, ap); // barycentric (1,0,0)

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return dot(bp, bp); // barycentric (0,1,0)

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        float3 proj = a + v * ab;
        return dot(p - proj, p - proj);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return dot(cp, cp); // barycentric (0,0,1)

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        float3 proj = a + w * ac;
        return dot(p - proj, p - proj);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 proj = b + w * (c - b);
        return dot(p - proj, p - proj);
    }

    // Inside face region
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 proj = a + ab * v + ac * w;
    return dot(p - proj, p - proj);
}

__host__ __device__ inline float3 ClosestPointOnTriangle(const float3& p, const float3& a, const float3& b, const float3& c)
{
    // From "Real-Time Collision Detection" by Christer Ericson
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);

    if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric (1,0,0)

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b; // barycentric (0,1,0)

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        return a + v * ab; // barycentric (1-v, v, 0)
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c; // barycentric (0,0,1)

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        return a + w * ac; // barycentric (1-w, 0, w)
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // barycentric (0, 1-w, w)
    }

    // Inside face region
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

__host__ __device__ inline bool RayTriangleIntersect(
    const float3& orig, const float3& dir,
    const float3& v0, const float3& v1, const float3& v2,
    float& t, float& u, float& v)
{
    const float EPSILON = 1e-6f;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (fabs(a) < EPSILON) return false;
    float f = 1.0f / a;
    float3 s = orig - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross(s, edge1);
    v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * dot(edge2, q);
    return t > EPSILON;
}

#pragma region Morton Code
__host__ __device__ inline uint64_t Part1By2(uint32_t n)
{
    uint64_t x = n & 0x1FFFFF;
    x = (x | x << 32) & 0x1F00000000FFFF;
    x = (x | x << 16) & 0x1F0000FF0000FF;
    x = (x | x << 8) & 0x100F00F00F00F00F;
    x = (x | x << 4) & 0x10C30C30C30C30C3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__host__ __device__ inline uint64_t EncodeMorton3D_21(uint32_t x, uint32_t y, uint32_t z)
{
    return (Part1By2(z) << 2) | (Part1By2(y) << 1) | (Part1By2(x));
}

__host__ __device__ inline uint32_t Compact1By2(uint64_t n)
{
    n &= 0x1249249249249249;
    n = (n ^ (n >> 2)) & 0x10C30C30C30C30C3;
    n = (n ^ (n >> 4)) & 0x100F00F00F00F00F;
    n = (n ^ (n >> 8)) & 0x1F0000FF0000FF;
    n = (n ^ (n >> 16)) & 0x1F00000000FFFF;
    n = (n ^ (n >> 32)) & 0x1FFFFF;
    return static_cast<uint32_t>(n);
}

__host__ __device__ inline void DecodeMorton3D_21(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z)
{
    x = Compact1By2(code >> 0);
    y = Compact1By2(code >> 1);
    z = Compact1By2(code >> 2);
}

__host__ __device__ inline uint64_t Float3ToMorton64(const float3& p, const float3& min_corner, float voxel_size)
{
    int3 idx;
    idx.x = static_cast<int>((p.x - min_corner.x) / voxel_size);
    idx.y = static_cast<int>((p.y - min_corner.y) / voxel_size);
    idx.z = static_cast<int>((p.z - min_corner.z) / voxel_size);

    return EncodeMorton3D_21(idx.x & 0x1FFFFF, idx.y & 0x1FFFFF, idx.z & 0x1FFFFF);
}

__host__ __device__ inline float3 Morton64ToFloat3(uint64_t code, const float3& min_corner, float voxel_size)
{
    uint32_t x, y, z;
    DecodeMorton3D_21(code, x, y, z);
    float3 p;
    p.x = min_corner.x + x * voxel_size;
    p.y = min_corner.y + y * voxel_size;
    p.z = min_corner.z + z * voxel_size;
    return p;
}
#pragma endregion

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

#ifdef __NVOPTIMUSENABLEMENT__
#define __NVOPTIMUSENABLEMENT__
extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001
#endif

inline bool ForceGPUPerformance()
{
    NvAPI_Status status;

    status = NvAPI_Initialize();
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSSessionHandle hSession = 0;
    status = NvAPI_DRS_CreateSession(&hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (2) load all the system settings into the session
    status = NvAPI_DRS_LoadSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSProfileHandle hProfile = 0;
    status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NVDRS_SETTING drsGet = { 0, };
    drsGet.version = NVDRS_SETTING_VER;
    status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
    if (status != NVAPI_OK)
    {
        return false;
    }
    auto m_gpu_performance = drsGet.u32CurrentValue;

    NVDRS_SETTING drsSetting = { 0, };
    drsSetting.version = NVDRS_SETTING_VER;
    drsSetting.settingId = PREFERRED_PSTATE_ID;
    drsSetting.settingType = NVDRS_DWORD_TYPE;
    drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;

    status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
    if (status != NVAPI_OK)
    {
        return false;
    }

    status = NvAPI_DRS_SaveSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (6) We clean up. This is analogous to doing a free()
    NvAPI_DRS_DestroySession(hSession);
    hSession = 0;

    return true;
}
