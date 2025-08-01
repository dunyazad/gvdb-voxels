#pragma once

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <cuda_vector_math.cuh>
#include <marching_cubes_constants.cuh>

#include <Serialization.hpp>

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

#define i8_max (INT8_MAX)
#define i8_min (-INT8_MAX)
#define i16_max (INT16_MAX)
#define i16_min (-INT16_MAX)
#define i32_max (INT32_MAX)
#define i32_min (-INT32_MAX)
#define i64_max (INT64_MAX)
#define i64_min (-INT64_MAX)

#define ui8_max (UINT8_MAX)
#define ui16_max (UINT16_MAX)
#define ui32_max (UINT32_MAX)
#define ui64_max (UINT64_MAX)

#define f32_max (FLT_MAX)
#define f32_min (-FLT_MAX)
#define f64_max (DBL_MAX)
#define f64_min (-DBL_MAX)


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

#ifndef PI
#define PI 3.14159265358979323846
#endif