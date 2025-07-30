#pragma warning(disable : 4819)

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>
#pragma comment(lib, "nvapi64.lib")

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

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

    vhm.Initialize(0.1f, d_input.numberOfPoints * 8, 32);

    vhm.Occupy(d_input, 3);

    HostPointCloud result = vhm.Serialize();
    result.CompactValidPoints();

    //PLYFormat plyVoxel;
    //for (size_t i = 0; i < result.numberOfPoints; i++)
    //{
    //    auto& p = result.positions[i];
    //    if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
    //    auto& n = result.normals[i];
    //    auto& c = result.colors[i];

    //    plyVoxel.AddCube(p.x, p.y, p.z, n.x, n.y, n.z, c.x, c.y, c.z, 1.0f, 0.2f);
    //}
    //plyVoxel.Serialize("../../res/3D/VoxelHashMapVoxel.ply");

    CUDA_TS(MarchingCubes);
    vhm.MarchingCubes(d_mesh);
    CUDA_TE(MarchingCubes);
    //h_mesh.CopyFromDevice(d_mesh);

    h_mesh.DeserializePLY("../../res/3D/HostHalfEdgeMesh.ply");

    //h_mesh.SerializePLY("../../res/3D/HostHalfEdgeMesh.ply", false);

    //{
    //    auto& mesh = h_mesh;

    //    size_t numHalfEdges = mesh.numberOfFaces * 3;

    //    for (size_t i = 0; i < numHalfEdges; ++i)
    //    {
    //        const auto& he = mesh.halfEdges[i];

    //        if (he.oppositeIndex == UINT32_MAX)
    //            continue; // boundary edge, skip

    //        const auto& ohe = mesh.halfEdges[he.oppositeIndex];

    //        // 1. Opposite of opposite should point back
    //        if (ohe.oppositeIndex != i)
    //        {
    //            printf("[Invalid Opposite] at halfEdge %zu: ohe.oppositeIndex = %u\n", i, ohe.oppositeIndex);
    //        }

    //        // 2. Shared edge must be reversed (v0->v1 vs. v1->v0)
    //        unsigned int he_v0 = he.vertexIndex;
    //        unsigned int he_v1 = mesh.halfEdges[he.nextIndex].vertexIndex;

    //        unsigned int ohe_v0 = ohe.vertexIndex;
    //        unsigned int ohe_v1 = mesh.halfEdges[ohe.nextIndex].vertexIndex;

    //        if (!(he_v0 == ohe_v1 && he_v1 == ohe_v0))
    //        {
    //            printf("[Mismatch] Edge direction mismatch at %zu: %u¡æ%u vs %u¡æ%u\n",
    //                i, he_v0, he_v1, ohe_v0, ohe_v1);
    //        }

    //        // 3. Optional: check next cycle closure
    //        const auto& e1 = mesh.halfEdges[he.nextIndex];
    //        const auto& e2 = mesh.halfEdges[e1.nextIndex];
    //        if (e2.nextIndex != i)
    //        {
    //            printf("[Loop Error] Face loop broken at halfEdge %zu\n", i);
    //        }
    //    }
    //}

    //h_mesh.SerializePLY("../../res/3D/HostHalfEdgeMesh.ply", false);

 /*   PLYFormat plyMesh;
    for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
    {
        auto& p = h_mesh.positions[i];
        if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
        auto& n = h_mesh .normals[i];
        auto& c = h_mesh .colors[i];

        plyMesh.AddPoint(p.x, p.y, p.z);
        plyMesh.AddNormal(n.x, n.y, n.z);
        plyMesh.AddColor(c.x, c.y, c.z);
    }
    for (size_t i = 0; i < h_mesh.numberOfFaces; i++)
    {
        auto& index = h_mesh.faces[i];

        plyMesh.AddFace(index.x, index.y, index.z);
    }
    plyMesh.Serialize("../../res/3D/MarchingCubes.ply");*/

    //PLYFormat hePLY;
    //for (unsigned int i = 0; i < mesh.numberOfPoints; ++i)
    //{
    //    auto& p = mesh.positions[i];
    //    auto& n = mesh.normals[i];
    //    auto& c = mesh.colors[i];
    //    hePLY.AddPoint(p.x, p.y, p.z);
    //    hePLY.AddNormal(n.x, n.y, n.z);
    //    hePLY.AddColor(c.x, c.y, c.z);
    //}

    //for (size_t i = 0; i < mesh.numberOfFaces; ++i)
    //{
    //    auto& f = mesh.halfEdgeFaces[i];
    //    auto& he0 = mesh.halfEdges[f.halfEdgeIndex];
    //    auto& he1 = mesh.halfEdges[he0.nextIndex];
    //    auto& he2 = mesh.halfEdges[he1.nextIndex];

    //    hePLY.AddFace(he0.vertexIndex, he1.vertexIndex, he2.vertexIndex);
    //}
    //hePLY.Serialize("../../res/3D/HalfEdgeMesh.ply");


    CUDA_TE(ProcessPointCloud);

    return result;
}
