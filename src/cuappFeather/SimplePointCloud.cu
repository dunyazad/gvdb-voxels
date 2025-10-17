#include <SimplePointCloud.cuh>

__global__ void DevicePointCloud_generateBoxes(
    cuBQL::box3f* boxForBuilder,
    const float3* positions,
    int numberOfPositions)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numberOfPositions) return;

    auto p = positions[tid];
    auto box = cuBQL::box3f();
    box.lower.x = p.x;
    box.lower.y = p.y;
    box.lower.z = p.z;
    box.upper.x = p.x;
    box.upper.y = p.y;
    box.upper.z = p.z;
    boxForBuilder[tid] = box;

    //printf("Box %d: Min(%f, %f, %f) Max(%f, %f, %f)\n", tid,
    //    boxForBuilder[tid].lower.x, boxForBuilder[tid].lower.y, boxForBuilder[tid].lower.z,
    //    boxForBuilder[tid].upper.x, boxForBuilder[tid].upper.y, boxForBuilder[tid].upper.z);
}

void Call_DevicePointCloud_generateBoxes(
    cuBQL::box3f* boxForBuilder,
    const float3* positions,
    int numberOfPositions)
{
    LaunchKernel(DevicePointCloud_generateBoxes, numberOfPositions,
        boxForBuilder, positions, numberOfPositions);
}

void Call_DevicePointCloud_gpuBuilder(
    cuBQL::bvh3f* bvh,
    cuBQL::box3f* boxes,
    unsigned int numberOfPoints)
{
    cuBQL::gpuBuilder(*bvh, boxes, numberOfPoints, cuBQL::BuildConfig());
}
