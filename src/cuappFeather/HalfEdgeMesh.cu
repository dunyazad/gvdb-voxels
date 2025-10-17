#include <HalfEdgeMesh.cuh>

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"

__global__ void generateCentroids(
    float3* positions,
    uint3* faces,
    unsigned int numberOfFaces,
    cuBQL::Triangle* triangles)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numberOfFaces) return;

    auto face = faces[tid];
    triangles[tid].a = cuBQL::vec3f(XYZ(positions[face.x]));
    triangles[tid].b = cuBQL::vec3f(XYZ(positions[face.y]));
    triangles[tid].c = cuBQL::vec3f(XYZ(positions[face.z]));

    //  printf("Triangle %d: A(%f, %f, %f) B(%f, %f, %f) C(%f, %f, %f)\n", tid,
    //      triangles[tid].a.x, triangles[tid].a.y, triangles[tid].a.z,
    //      triangles[tid].b.x, triangles[tid].b.y, triangles[tid].b.z,
    // 	  triangles[tid].c.x, triangles[tid].c.y, triangles[tid].c.z);
}

__global__ void generateBoxes(
    cuBQL::box3f* boxForBuilder,
    const cuBQL::Triangle* triangles,
    int numTriangles)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numTriangles) return;

    auto triangle = triangles[tid];
    boxForBuilder[tid] = triangle.bounds();

    //printf("Box %d: Min(%f, %f, %f) Max(%f, %f, %f)\n", tid,
    //    boxForBuilder[tid].lower.x, boxForBuilder[tid].lower.y, boxForBuilder[tid].lower.z,
    //    boxForBuilder[tid].upper.x, boxForBuilder[tid].upper.y, boxForBuilder[tid].upper.z);
}

void Call_generateCentroids(
    float3* positions,
    uint3* faces,
    unsigned int numberOfFaces,
    cuBQL::Triangle* triangles)
{
    LaunchKernel(generateCentroids, numberOfFaces,
        positions, faces, numberOfFaces, triangles);
}

void Call_generateBoxes(
    cuBQL::box3f* boxes,
    const cuBQL::Triangle* triangles,
    int numTriangles)
{
    LaunchKernel(generateBoxes, numTriangles,
        boxes, triangles, numTriangles);
}

void Call_gpuBuilder(
    cuBQL::bvh3f* bvh,
    cuBQL::box3f* boxes,
    unsigned int numberOfFaces)
{
    cuBQL::gpuBuilder(*bvh, boxes, numberOfFaces, cuBQL::BuildConfig());
}

__global__ void Kernel_DeviceHalfEdgeMesh_FindNearestPoints(
    const cuBQL::Triangle* __restrict__ triangles,
    cuBQL::bvh3f bvh,
    const float3* __restrict__ queries,
    float3* __restrict__ resultsPositions,
    int* __restrict__ resultTriangleIndices,
    size_t numberOfQueries)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numberOfQueries) return;

    auto& query = queries[index];

    //printf("Thread %zu processing query point (%.3f, %.3f, %.3f)\n", index, query.x, query.y, query.z);

    cuBQL::vec3f queryPoint(query.x, query.y, query.z);
    cuBQL::triangles::CPAT cpat;
    cpat.runQuery(triangles, bvh, queryPoint);

    resultsPositions[index].x = cpat.P.x;
    resultsPositions[index].y = cpat.P.y;
    resultsPositions[index].z = cpat.P.z;

    resultTriangleIndices[index] = cpat.triangleIdx;

    //  printf("Query[%zu]: (%.3f, %.3f, %.3f) -> Nearest: (%.3f, %.3f, %.3f), dist=%.6f\n",
    //      index, query.x, query.y, query.z,
    //      results[index].x, results[index].y, results[index].z,
          //cuBQL::length(cpat.P - queryPoint));
}

void Call_Kernel_FindNearestPoints(
    const cuBQL::Triangle* __restrict__ triangles,
    cuBQL::bvh3f bvh,
    const float3* __restrict__ d_queries,
    float3* __restrict__ d_resultsPositions,
    int* __restrict__ d_resultTriangleIndices,
    size_t numberOfQueries)
{
    LaunchKernel(Kernel_DeviceHalfEdgeMesh_FindNearestPoints, (unsigned int)numberOfQueries,
        triangles, bvh, d_queries, d_resultsPositions, d_resultTriangleIndices, numberOfQueries);
}