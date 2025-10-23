#pragma warning(disable : 4819)
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>
#pragma comment(lib, "nvapi64.lib")

#include <cuda_common.cuh>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>

#include <Serialization.hpp>

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"

#include "ThrustHashMap.hpp"

#include <Octree.cuh>

#include <IVisualDebugging.h>
using VD = IVisualDebugging;

CUDAInstance::CUDAInstance()
{
}

CUDAInstance::~CUDAInstance()
{
    vhm.Terminate();
    d_input.Terminate();
    h_mesh.Terminate();
    d_mesh.Terminate();
}

void CUDAInstance::ProcessHalfEdgeMesh(const std::string& filename)
{
    h_mesh.DeserializePLY(filename);
}

//HostPointCloud<PointCloudProperty> CUDAInstance::ProcessPointCloud(const HostPointCloud<PointCloudProperty>& input, float voxelSize, unsigned int occupyOffset)
//{
//    CUDA_TS(ProcessPointCloud);
//
//    h_input = input;
//    d_input = h_input;
//
//    d_input.Count();
//
//    vhm.Initialize(voxelSize, d_input.numberOfPoints * 8, 128);
//
//    vhm.Occupy(d_input, occupyOffset);
//
//    auto result = vhm.Serialize();
//    result.CompactValidPoints();
//
//    CUDA_TS(MarchingCubes);
//    vhm.MarchingCubes(d_mesh);
//    CUDA_TE(MarchingCubes);
//
//    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);
//
//    d_mesh.RemoveIsolatedVertices();
//
//    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);
//
//    h_mesh.CopyFromDevice(d_mesh);
//
//    //h_mesh.SerializePLY("../../res/3D/host_mesh.ply");
//
//    printf("h_mesh.numberOfPoints : %d\n", h_mesh.numberOfPoints);
//
//    vhm.Terminate();
//
//    for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
//    {
//        if (UINT32_MAX == h_mesh.vertexToHalfEdge[i])
//        {
//            printf("Vertex : %llu has no half-edge.\n", i);
//        }
//    }
//
//    CUDA_TE(ProcessPointCloud);
//
//    //ThrustHostHalfEdgeMesh th_mesh;
//    //th_mesh.DeserializePLY("../../res/3D/HostHalfEdgeMesh_Normal.ply");
//
//    //ThrustDeviceHalfEdgeMesh td_mesh;
//    //td_mesh.CopyFromHost(th_mesh);
//
//    //td_mesh.BuildHalfEdges();
//
//    //th_mesh.CopyFromDevice(td_mesh);
//
//    //th_mesh.SerializePLY("../../res/3D/Result.ply");
//
//    return result;
//}

void CUDAInstance::ProcessPointCloud(float voxelSize, unsigned int occupyOffset)
{
    d_input = h_input;

    vhm.Initialize(voxelSize, d_input.numberOfPoints * 32, 32);

    CUDA_TS(ProcessPointCloud);
    vhm.Occupy(d_input, occupyOffset);

    CUDA_TS(MarchingCubes);
    vhm.MarchingCubes(d_mesh);
    CUDA_TE(MarchingCubes);

    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    d_mesh.RemoveIsolatedVertices();



    CUDA_TE(ProcessPointCloud);
    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    h_mesh.CopyFromDevice(d_mesh);

    printf("h_mesh.numberOfPoints : %d\n", h_mesh.numberOfPoints);

    vhm.Terminate();

}

void CUDAInstance::Test()
{
    PLYFormat ply;
    ply.Deserialize("D:\\Debug\\PLY\\input.ply");

    std::vector<float3> inputPoints;

    for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
    {
        auto x = ply.GetPoints()[i * 3 + 0];
        auto y = ply.GetPoints()[i * 3 + 1];
        auto z = ply.GetPoints()[i * 3 + 2];

        inputPoints.push_back(make_float3(x, y, z));
    }

    Octree octree;
    octree.Build(inputPoints);
}