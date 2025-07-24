#pragma once

#include <cuda_common.cuh>

#include <Eigen/Core>
#include <Eigen/Dense>
namespace Eigen {
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

struct BitVolumeInfo
{
    dim3 dimensions = dim3(1000, 1000, 1000);
    unsigned int numberOfVoxels = 1000 * 1000 * 1000;

    dim3 allocated_dimensions = dim3(0, 0, 0);
    unsigned int numberOfAllocatedWords = 0;

    float voxelSize = 0.1f;
    float3 volumeMin;
    uint32_t* d_volume = nullptr;

    bool* d_simplePointLUT = nullptr;
};

__global__ void Kernel_BitVolumeInfo_OccupyFromPoints(BitVolumeInfo info, const float3* points, int numPoints);
__global__ void Kernel_BitVolumeInfo_OccupyFromEigenPoints(BitVolumeInfo info, const Eigen::Vector3f* points, int numPoints);
__global__ void Kernel_BitVolumeInfo_SerializeToFloat3(BitVolumeInfo info, float3* outPoints, unsigned int* outCount);
__global__ void Kernel_BitVolumeInfo_MarchingCubes(BitVolumeInfo info, float3* vertices, int3* faces, unsigned int* vCount, unsigned int* fCount);

struct BitVolume
{
    BitVolumeInfo info;

    void Initialize(dim3 dimensions = dim3(1000, 1000, 1000), float voxelSize = 0.1f);
    void Terminate();
    void Clear();

    __device__ static bool isOccupied(const BitVolumeInfo& info, dim3 voxelIndex);
    __device__ static void setOccupied(BitVolumeInfo& info, dim3 voxelIndex);
    __device__ static void setNotOccupied(BitVolumeInfo& info, dim3 voxelIndex);
    
    void OccupyFromPoints(float3 volumeMin, const float3* d_points, int numPoints);
    void OccupyFromEigenPoints(float3 volumeMin, const Eigen::Vector3f* d_points, int numPoints);
    void SerializeToFloat3(float3* d_output, unsigned int* d_outputCount);

    __device__ static float getVoxelCornerValue(const BitVolumeInfo& info, dim3 base, int dx, int dy, int dz);
    __device__ static float3 getEdgeVertexPosition(const BitVolumeInfo& info, dim3 base, int edge);

    void MarchingCubes(std::vector<float3>& vertices, std::vector<int3>& faces);
};
