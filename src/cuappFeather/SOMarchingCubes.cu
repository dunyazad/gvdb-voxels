#include <SOMarchingCubes.cuh>

__global__ void Kernel_SOMarhcingCubes_Clear(cudaSurfaceObject_t surf1, cudaSurfaceObject_t surf2, dim3 dims);

void SOMarhcingCubes::Initialize()
{
	CUDA_TS(SOMarhcingCubes_Initialize);

	// Surface가 사용할 float4 채널 포맷을 정의
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

	// 2개의 3D cudaArray를 할당
	cudaExtent extent = make_cudaExtent(XYZ(dimensions));
	CUDA_CHECK(cudaMalloc3DArray(&d_array1, &channel_desc, extent, cudaArraySurfaceLoadStore));
	CUDA_CHECK(cudaMalloc3DArray(&d_array2, &channel_desc, extent, cudaArraySurfaceLoadStore));

	// 2개의 Surface Object를 생성
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	res_desc.res.array.array = d_array1;
	CUDA_CHECK(cudaCreateSurfaceObject(&surf_obj1, &res_desc));
	res_desc.res.array.array = d_array2;
	CUDA_CHECK(cudaCreateSurfaceObject(&surf_obj2, &res_desc));

	CUDA_TE(SOMarhcingCubes_Initialize);
}

void SOMarhcingCubes::Terminate()
{
	CUDA_TS(SOMarhcingCubes_Terminate);

	CUDA_CHECK(cudaDestroySurfaceObject(surf_obj1));
	CUDA_CHECK(cudaDestroySurfaceObject(surf_obj2));
	CUDA_CHECK(cudaFreeArray(d_array1));
	CUDA_CHECK(cudaFreeArray(d_array2));

	CUDA_TE(SOMarhcingCubes_Terminate);
}

void SOMarhcingCubes::Clear()
{
	CUDA_TS(SOMarhcingCubes_Clear);

	dim3 block_dim(8, 8, 8);
	dim3 grid_dim(
		(dimensions.x + block_dim.x - 1) / block_dim.x,
		(dimensions.y + block_dim.y - 1) / block_dim.y,
		(dimensions.z + block_dim.z - 1) / block_dim.z
	);

	Kernel_SOMarhcingCubes_Clear<<<grid_dim, block_dim >>>(surf_obj1, surf_obj2, dimensions);

	CUDA_CHECK(cudaGetLastError());
	CUDA_SYNC();

	CUDA_TE(SOMarhcingCubes_Clear);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Kernel_SOMarhcingCubes_Clear(cudaSurfaceObject_t surf1, cudaSurfaceObject_t surf2, dim3 dims)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dims.x || y >= dims.y || z >= dims.z) {
		return;
	}

	// 초기화할 기본값 (모든 멤버가 FLT_MAX)
	const float4 default_value = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

	// 두 개의 Surface에 기본값을 쓴다.
	//surf3Dwrite(default_value, surf1, x * sizeof(float4), y, z);
	//surf3Dwrite(default_value, surf2, x * sizeof(float4), y, z);
}
