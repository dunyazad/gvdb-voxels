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


#include "cuda.h"
#pragma comment(lib, "cuda.lib")


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

void savePPM(const char* filename, const uchar* buffer, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; ++i) {
        // RGBA¿¡¼­ RGB¸¸ ¾¸
        fwrite(buffer + i * 4, 1, 3, fp);
    }
    fclose(fp);
    printf("Rendered image saved to %s\n", filename);
}





























using namespace nvdb;

// -------- BMP Save Utility --------
#pragma pack(push, 1)
struct BmpFileHeader {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
};
struct BmpInfoHeader {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};
#pragma pack(pop)

static void saveBMP(const char* filename, const unsigned char* buffer, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error opening %s\n", filename);
        return;
    }
    int bpp = 3;
    int pad = (4 - (width * bpp) % 4) % 4;
    int img = (width * bpp + pad) * height;

    BmpFileHeader fh = { 0x4D42, sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + img, 0, 0,
                         sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) };
    BmpInfoHeader ih = { sizeof(BmpInfoHeader), width, height, 1, 24, 0, img, 0, 0, 0, 0 };

    fwrite(&fh, sizeof(fh), 1, fp);
    fwrite(&ih, sizeof(ih), 1, fp);

    unsigned char zero[3] = { 0, 0, 0 };
    for (int y = height - 1; y >= 0; --y) {
        const unsigned char* row = buffer + y * width * 4;
        for (int x = 0; x < width; ++x) {
            unsigned char bgr[3] = { row[2], row[1], row[0] };
            fwrite(bgr, 1, 3, fp);
            row += 4;
        }
        fwrite(zero, 1, pad, fp);
    }

    fclose(fp);
    printf("Saved %s\n", filename);
}

// -------- Generate Sphere Surface Points --------
static std::vector<Vector3DF> makeSpherePoints(float radius, int n, const Vector3DF& c) {
    std::vector<Vector3DF> pts;
    pts.reserve(n);

    for (int i = 0; i < n; ++i) {
        float phi = acosf(1.0f - 2.0f * (float)i / (float)n);
        float theta = 3.1415926535f * (1.0f + sqrtf(5.0f)) * (float)i;
        float x = c.x + radius * cosf(theta) * sinf(phi);
        float y = c.y + radius * sinf(theta) * sinf(phi);
        float z = c.z + radius * cosf(phi);
        pts.emplace_back(x, y, z);
    }
    return pts;
}

// -------- Main Test --------
void CUDAInstance::Test(GLuint textureID)
{
	
}



void CUDAMain()
{
    PLYFormat ply;
    ply.Deserialize("D:\\Debug\\PLY\\inputA.ply");
    auto [minx, miny, minz] = ply.GetAABBMin();

    std::vector<glm::vec3> pointCloud;
    size_t pointCount = ply.GetPoints().size() / 3;
    pointCloud.reserve(pointCount);

    for (size_t i = 0; i < pointCount; ++i)
    {
        float x = ply.GetPoints()[i * 3 + 0];
        float y = ply.GetPoints()[i * 3 + 1];
        float z = ply.GetPoints()[i * 3 + 2];

        if (-10.0f < x && x < 10.0f &&
            -10.0f < y && y < 10.0f &&
            -10.0f < z && z < 10.0f)
        {
            pointCloud.emplace_back(x, y, z);
        }
    }

    HostPointCloud<glm::vec3> h_pcd;
	h_pcd.Initialize(static_cast<unsigned int>(pointCloud.size()));
    h_pcd.positions = new float3[h_pcd.numberOfPoints];
    memcpy(h_pcd.positions, pointCloud.data(), sizeof(float3) * h_pcd.numberOfPoints);

    DevicePointCloud<glm::vec3> d_pcd;
    d_pcd = h_pcd;
    d_pcd.UpdateBVH();

}
