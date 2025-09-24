#pragma once

#include <cuda_common.cuh>

struct SOMarhcingCubesVoxel
{
	float sdf = FLT_MAX;
	float normalX = FLT_MAX;
	float normalY = FLT_MAX;
	float normalZ = FLT_MAX;

	float weight = FLT_MAX;
	float colorR = FLT_MAX;
	float colorG = FLT_MAX;
	float colorB = FLT_MAX;
};

struct SOMarhcingCubes
{
	dim3 dimensions = dim3(300, 300, 300);

	cudaArray* d_array1 = nullptr;
	cudaArray* d_array2 = nullptr;
	cudaSurfaceObject_t surf_obj1 = 0;
	cudaSurfaceObject_t surf_obj2 = 0;

	void Initialize();
	void Terminate();
	void Clear();
};
