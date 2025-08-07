#pragma once

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//#include <device_functions.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <cuda_vector_math.cuh>
#include <marching_cubes_constants.cuh>

// Vector containers
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Algorithms
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/binary_search.h>



#include <Serialization.hpp>

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

#ifdef MIN_MAX_DEFINITIONS
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

#ifndef LaunchKernel
#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)
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
