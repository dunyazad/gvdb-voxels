#pragma warning(disable : 4819)

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>
#pragma comment(lib, "nvapi64.lib")

#include <cuda_common.cuh>

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <Serialization.hpp>

//HostPointCloud ProcessPointCloud(const HostPointCloud& h_input)
//{
//    CUDA_TS(VoxelHashMap);
//
//    DevicePointCloud d_input(h_input);
//
//    VoxelHashMap vhm;
//    vhm.Initialize(0.2f, d_input.numberOfPoints * 8, 32);
//
//#define OCCUPY_SDF
//#ifndef OCCUPY_SDF
//    vhm.Occupy(d_input);
//    //vhm.FilterByNormalGradientWithOffset(3, 0.5f, false);
//    //vhm.FilterBySDFGradient(10.0f, false);
//    vhm.FilterBySDFGradientWithOffset(10, 0.1f, false);
//    //vhm.FindOverlap(3, true);
//    //vhm.FilterByNormalGradient(0.5f, true);
//    HostPointCloud result = vhm.Serialize();
//    result.CompactValidPoints();
//#else
//    vhm.Occupy_SDF(d_input, 3);
//
//    //vhm.Dilation(3, 1);
//
//    auto hpcd = vhm.Serialize_SDF_Tidy();
//
//    PLYFormat plyVoxel;
//    for (size_t i = 0; i < hpcd.numberOfPoints; i++)
//    {
//        auto& p = hpcd.positions[i];
//        if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
//        auto& n = hpcd.normals[i];
//        auto& c = hpcd.colors[i];
//
//        plyVoxel.AddCube(p.x, p.y, p.z, n.x, n.y, n.z, c.x, c.y, c.z, 1.0f, 0.2f);
//    }
//    plyVoxel.Serialize("../../res/3D/VoxelHashMapVoxel.ply");
//
//    
//    vector<float3> vertices;
//    vector<float3> normals;
//    vector<float3> colors;
//    vector<uint3> triangles;
//    vhm.MarchingCubes(vertices, normals, colors, triangles);
//
//    PLYFormat ply;
//    for (size_t i = 0; i < vertices.size(); i++)
//    {
//        auto& p = vertices[i];
//        auto& n = normals[i];
//        auto& c = colors[i];
//
//        ply.AddPoint(p.x, p.y, p.z);
//        ply.AddNormal(n.x, n.y, n.z);
//        ply.AddColor(c.x, c.y, c.z);
//    }
//    for (size_t i = 0; i < triangles.size(); i++)
//    {
//        auto& t = triangles[i];
//
//        ply.AddTriangleIndex(t.x);
//        ply.AddTriangleIndex(t.z);
//        ply.AddTriangleIndex(t.y);
//    }
//    ply.Serialize("../../res/3D/MarchingCubes.ply");
//    //vhm.MarchingCubes("../../res/3D/MarchingCubes.ply");
//    //vhm.FilterBySDFGradientWithOffset(3, 10.0f, false);
//    //vhm.FindOverlap(3, true);
//    //vhm.SmoothSDF(3);
//    //vhm.FilterOppositeNormals();
//    //vhm.FilterByNormalGradient(0.1f, false);
//    HostPointCloud result = vhm.Serialize_SDF();
//    result.CompactValidPoints();
//#endif
//    
//    d_input.Terminate();
//
//    CUDA_TE(VoxelHashMap);
//
//    return result;
//}

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

void CUDAInstance::ProcessHalfEdgeMesh(const string& filename)
{
    h_mesh.DeserializePLY(filename);
}

HostPointCloud CUDAInstance::ProcessPointCloud(const HostPointCloud& h_input)
{
    CUDA_TS(ProcessPointCloud);

    d_input = h_input;

    vhm.Initialize(0.1f, d_input.numberOfPoints * 32, 32);

    vhm.Occupy(d_input, 3);

    HostPointCloud result = vhm.Serialize();
    result.CompactValidPoints();

    CUDA_TS(MarchingCubes);
    vhm.MarchingCubes(d_mesh);
    CUDA_TE(MarchingCubes);

    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    d_mesh.RemoveIsolatedVertices();

    printf("d_mesh.numberOfPoints : %d\n", d_mesh.numberOfPoints);

    h_mesh.CopyFromDevice(d_mesh);

    //h_mesh.SerializePLY("../../res/3D/host_mesh.ply");

    printf("h_mesh.numberOfPoints : %d\n", h_mesh.numberOfPoints);

    vhm.Terminate();

    for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
    {
        if (UINT32_MAX == h_mesh.vertexToHalfEdge[i])
        {
            printf("Vertex : %llu has no half-edge.\n", i);
        }
    }

    CUDA_TE(ProcessPointCloud);

    //ThrustHostHalfEdgeMesh th_mesh;
    //th_mesh.DeserializePLY("../../res/3D/HostHalfEdgeMesh_Normal.ply");

    //ThrustDeviceHalfEdgeMesh td_mesh;
    //td_mesh.CopyFromHost(th_mesh);

    //td_mesh.BuildHalfEdges();

    //th_mesh.CopyFromDevice(td_mesh);

    //th_mesh.SerializePLY("../../res/3D/Result.ply");

    return result;
}
