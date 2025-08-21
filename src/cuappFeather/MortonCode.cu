#include <MortonCode.cuh>

/*
__global__ void Kernel_MortonCode_Encode(const uint3* in, uint64_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    int x = in[i].x;
    int y = in[i].y;
    int z = in[i].z;

    // 사용 환경에 따라 범위 보장 필요: x,y,z ∈ [-2^20, 2^20-1]
    out[i] = MortonCode::morton3D_encode_biased(x, y, z);
}

__global__ void Kernel_MortonCode_Decode(const uint64_t* codes, uint3* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    int x, y, z;
    MortonCode::morton3D_decode_biased(codes[i], x, y, z);
    out[i].x = x;
    out[i].y = y;
    out[i].z = z;
}
*/
