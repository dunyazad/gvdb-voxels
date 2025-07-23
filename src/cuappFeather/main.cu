#pragma warning(disable : 4819)

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>

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

#pragma comment(lib, "nvapi64.lib")

#include <HashMap.hpp>
#include <BitVolume.hpp>
#include <DenseGrid.hpp>
#include <KDTree.hpp>
#include <VoxelHashMap.cuh>
#include <SCVoxelHashMap.cuh>
#include <HalfEdgeMesh.cuh>
#include <TSDF.hpp>

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

HostPointCloud ProcessPointCloud(const HostPointCloud& h_input)
{
    CUDA_TS(SCVoxelHashMap);

    DevicePointCloud d_input(h_input);

    SCVoxelHashMap vhm;
    vhm.Initialize(0.1f, d_input.numberOfPoints * 8, 32);

    vhm.Occupy(d_input, 3);
    
    HostPointCloud result = vhm.Serialize();
    //result.CompactValidPoints();

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
    HostHalfEdgeMesh mesh = vhm.MarchingCubes();
    CUDA_TE(MarchingCubes);

    PLYFormat plyMesh;
    for (size_t i = 0; i < mesh.numberOfPoints; i++)
    {
        auto& p = mesh.positions[i];
        if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
        auto& n = mesh.normals[i];
        auto& c = mesh.colors[i];

        plyMesh.AddPoint(p.x, p.y, p.z);
        plyMesh.AddNormal(n.x, n.y, n.z);
        plyMesh.AddColor(c.x, c.y, c.z);
    }
    for (size_t i = 0; i < mesh.numberOfFaces; i++)
    {
        auto& index = mesh.faces[i];

        plyMesh.AddFace(index.x, index.y, index.z);
    }
    plyMesh.Serialize("../../res/3D/MarchingCubes.ply");

    d_input.Terminate();

    CUDA_TE(SCVoxelHashMap);

    return result;
}

__host__ __device__
float3 rgb_to_hsv(uchar3 rgb) {
    float r = rgb.x / 255.0f;
    float g = rgb.y / 255.0f;
    float b = rgb.z / 255.0f;

    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float delta = cmax - cmin;

    float h = 0.0f;
    if (delta > 1e-6f) {
        if (cmax == r) {
            h = fmodf((g - b) / delta, 6.0f);
        }
        else if (cmax == g) {
            h = (b - r) / delta + 2.0f;
        }
        else {
            h = (r - g) / delta + 4.0f;
        }
        h *= 60.0f;
        if (h < 0.0f) h += 360.0f;
    }

    float s = (cmax == 0.0f) ? 0.0f : delta / cmax;
    float v = cmax;

    return make_float3(h, s, v); // H in degrees, S and V in [0,1]
}

__host__ __device__
uchar3 hsv_to_rgb(float3 hsv) {
    float h = hsv.x; // [0, 360)
    float s = hsv.y; // [0, 1]
    float v = hsv.z; // [0, 1]

    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r, g, b;
    if (h < 60.0f) {
        r = c; g = x; b = 0;
    }
    else if (h < 120.0f) {
        r = x; g = c; b = 0;
    }
    else if (h < 180.0f) {
        r = 0; g = c; b = x;
    }
    else if (h < 240.0f) {
        r = 0; g = x; b = c;
    }
    else if (h < 300.0f) {
        r = x; g = 0; b = c;
    }
    else {
        r = c; g = 0; b = x;
    }

    uchar3 rgb;
    rgb.x = static_cast<unsigned char>((r + m) * 255.0f);
    rgb.y = static_cast<unsigned char>((g + m) * 255.0f);
    rgb.z = static_cast<unsigned char>((b + m) * 255.0f);
    return rgb;
}

__host__ __device__ uint64_t IndexToKey(uint3 index)
{
    uint32_t ux = static_cast<uint32_t>(index.x + (1 << 20)) & 0x1FFFFF; // 21 bits
    uint32_t uy = static_cast<uint32_t>(index.y + (1 << 20)) & 0x1FFFFF; // 21 bits
    uint32_t uz = static_cast<uint32_t>(index.z + (1 << 20)) & 0x1FFFFF; // 21 bits

    // Pack into 64-bit key: |   21-bit Z   |   21-bit Y   |   21-bit X   |
    uint64_t key = (static_cast<uint64_t>(uz) << 42) |
        (static_cast<uint64_t>(uy) << 21) |
        (static_cast<uint64_t>(ux));
    return key;
}

__host__ __device__ uint3 KeyToIndex(uint64_t key)
{
    int32_t x = ((key >> 0) & 0x1FFFFF) - (1 << 20);
    int32_t y = ((key >> 21) & 0x1FFFFF) - (1 << 20);
    int32_t z = ((key >> 42) & 0x1FFFFF) - (1 << 20);
    return make_uint3(x, y, z);
}

__device__ uint64_t D_ToKey(float3 p, float resolution)
{
    // Quantize float3 to int32
    int32_t qx = __float2int_rd(p.x / resolution);
    int32_t qy = __float2int_rd(p.y / resolution);
    int32_t qz = __float2int_rd(p.z / resolution);

    // Map signed int32 to unsigned 21-bit value (2's complement to unsigned shift)
    // Shift by 20 to bring range [-2^20, 2^20) into [0, 2^21)
    uint32_t ux = static_cast<uint32_t>(qx + (1 << 20)) & 0x1FFFFF; // 21 bits
    uint32_t uy = static_cast<uint32_t>(qy + (1 << 20)) & 0x1FFFFF; // 21 bits
    uint32_t uz = static_cast<uint32_t>(qz + (1 << 20)) & 0x1FFFFF; // 21 bits

    // Pack into 64-bit key: |   21-bit Z   |   21-bit Y   |   21-bit X   |
    uint64_t key = (static_cast<uint64_t>(uz) << 42) |
        (static_cast<uint64_t>(uy) << 21) |
        (static_cast<uint64_t>(ux));
    return key;
}

__device__ float3 D_FromKey(uint64_t key, float resolution)
{
    int32_t x = ((key >> 0) & 0x1FFFFF) - (1 << 20);
    int32_t y = ((key >> 21) & 0x1FFFFF) - (1 << 20);
    int32_t z = ((key >> 42) & 0x1FFFFF) - (1 << 20);
    return make_float3(x * resolution, y * resolution, z * resolution);
}

__host__ uint64_t H_ToKey(const float3& p, float resolution)
{
    // Quantize
    int32_t qx = static_cast<int32_t>(std::floor(p.x / resolution));
    int32_t qy = static_cast<int32_t>(std::floor(p.y / resolution));
    int32_t qz = static_cast<int32_t>(std::floor(p.z / resolution));

    // Convert to unsigned with bias to handle negative coordinates
    uint32_t ux = static_cast<uint32_t>(qx + (1 << 20)) & 0x1FFFFF;
    uint32_t uy = static_cast<uint32_t>(qy + (1 << 20)) & 0x1FFFFF;
    uint32_t uz = static_cast<uint32_t>(qz + (1 << 20)) & 0x1FFFFF;

    // Pack into 64-bit key
    uint64_t key = (static_cast<uint64_t>(uz) << 42) |
        (static_cast<uint64_t>(uy) << 21) |
        (static_cast<uint64_t>(ux));
    return key;
}

__host__ float3 H_FromKey(uint64_t key, float resolution)
{
    int32_t x = ((key >> 0) & 0x1FFFFF) - (1 << 20);
    int32_t y = ((key >> 21) & 0x1FFFFF) - (1 << 20);
    int32_t z = ((key >> 42) & 0x1FFFFF) - (1 << 20);

    float3 result;
    result.x = static_cast<float>(x) * resolution;
    result.y = static_cast<float>(y) * resolution;
    result.z = static_cast<float>(z) * resolution;
    return result;
}


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <iostream>
#include <cmath>

struct SpherePointGenerator
{
    float voxelSize;
    float radius;
    int gridSize;

    SpherePointGenerator(float voxelSize_, float radius_)
        : voxelSize(voxelSize_), radius(radius_) {
        gridSize = static_cast<int>(radius / voxelSize) * 2 + 1;
    }

    __host__ __device__
        bool inSphere(int x, int y, int z) const {
        float fx = (x - gridSize / 2) * voxelSize;
        float fy = (y - gridSize / 2) * voxelSize;
        float fz = (z - gridSize / 2) * voxelSize;
        return (fx * fx + fy * fy + fz * fz) <= radius * radius;
    }

    __host__ __device__
        float3 operator()(int i) const {
        int z = i % gridSize;
        int y = (i / gridSize) % gridSize;
        int x = i / (gridSize * gridSize);

        if (inSphere(x, y, z)) {
            float fx = (x - gridSize / 2) * voxelSize;
            float fy = (y - gridSize / 2) * voxelSize;
            float fz = (z - gridSize / 2) * voxelSize;
            return make_float3(fx, fy, fz);
        }
        return make_float3(NAN, NAN, NAN); // Marker for invalid point
    }
};

struct IsValidPoint
{
    __host__ __device__
        bool operator()(const float3& p) const {
        return !isnan(p.x);
    }
};

















void cuMain(
	float voxelSize,
	std::vector<float3>& host_points,
	std::vector<float3>& host_normals,
	std::vector<uchar3>& host_colors,
	float3 center)
{
    std::vector<float3> h_colors;
    for (auto& c : host_colors)
    {
        h_colors.push_back(make_float3((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f));
    }

    thrust::device_vector<float3> d_points = host_points;
    thrust::device_vector<float3> d_normals = host_normals;
    thrust::device_vector<float3> d_colors = h_colors;

    //{
    //    CUDA_TS(TSDF);
    //    
    //    TSDF tsdf;
    //    tsdf.Initialize(make_float3(-17.0f, -63.0f, -31.0f), dim3(800, 760, 480));

    //    CUDA_TS(Occupy);
    //    tsdf.Occupy(
    //        thrust::raw_pointer_cast(d_points.data()),
    //        thrust::raw_pointer_cast(d_normals.data()),
    //        thrust::raw_pointer_cast(d_colors.data()),
    //        d_points.size());
    //    CUDA_SYNC();
    //    CUDA_TE(Occupy);

    //    tsdf.Serialize("../../res/3D/tsdf.ply");

    //    tsdf.Terminate();

    //    CUDA_TE(TSDF);
    //}
   
    //{
    //    CUDA_TS(DenseGrid);
    //    
    //    struct DenseGridVoxel
    //    {
    //        float3 normal;
    //        float3 color;
    //        unsigned int count;
    //    };

    //    DenseGrid<float3> dg;
    //    dg.Initialize(make_float3(-17.0f, -63.0f, -31.0f), dim3(800, 760, 480));

    //    CUDA_TS(Occupy);
    //    dg.Occupy(
    //        thrust::raw_pointer_cast(d_points.data()),
    //        thrust::raw_pointer_cast(d_normals.data()),
    //        thrust::raw_pointer_cast(d_colors.data()),
    //        d_points.size());
    //    CUDA_SYNC();
    //    CUDA_TE(Occupy);

    //    //dg.Serialize("../../res/3D/denseGrid.ply");

    //    dg.Terminate();

    //    CUDA_TE(DenseGrid);
    //}

    {
        //CUDA_TS(LUT);
        //{
        //    bool* d_lut;
        //    cudaMalloc(&d_lut, sizeof(bool) * (1 << 26));

        //    int threads = 256;
        //    int blocks = ((1 << 26) + threads - 1) / threads;

        //    Kernel_GenerateLUT << <blocks, threads >> > (d_lut);
        //    CUDA_SYNC();

        //    // ¿˙¿Â
        //    bool* h_lut = new bool[1 << 26];
        //    CUDA_COPY_D2H(h_lut, d_lut, sizeof(bool) * (1 << 26));

        //    std::ofstream fout("../../res/3D/simple_point_lut_cuda.bin", std::ios::binary);
        //    fout.write(reinterpret_cast<const char*>(h_lut), (1 << 26));
        //    fout.close();
        //}
        //CUDA_TE(LUT);
    }










    {
        //pointCloud.numberOfPoints = host_points.size();

        //cudaMalloc(&pointCloud.d_points, sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
        //cudaMalloc(&pointCloud.d_normals, sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
        //cudaMalloc(&pointCloud.d_colors, sizeof(Eigen::Vector3b) * pointCloud.numberOfPoints);

        //CUDA_COPY_H2D(pointCloud.d_points, host_points.data(), sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
        //CUDA_COPY_H2D(pointCloud.d_normals, host_normals.data(), sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
        //CUDA_COPY_H2D(pointCloud.d_colors, host_colors.data(), sizeof(Eigen::Vector3b) * pointCloud.numberOfPoints);



        //CUDA_TS(BitVolume);

        //BitVolume bitVolume;


        //CUDA_TS(BitVolume_Initialize);
        //bitVolume.Initialize(dim3(1000, 1000, 1000), 0.1f);
        //CUDA_SYNC();
        //CUDA_TE(BitVolume_Initialize);


        //CUDA_TS(BitVolume_Occupy);
        //bitVolume.OccupyFromEigenPoints(make_float3(-20, -70, -40), pointCloud.d_points, pointCloud.numberOfPoints);
        //CUDA_SYNC();
        //CUDA_TE(BitVolume_Occupy);



        //unsigned int* d_numberOfPoints = nullptr;
        //cudaMalloc(&d_numberOfPoints, sizeof(unsigned int));
        //float3* d_points = nullptr;
        //cudaMalloc(&d_points, sizeof(float3) * pointCloud.numberOfPoints);
        //bitVolume.SerializeToFloat3(d_points, d_numberOfPoints);
        //CUDA_SYNC();

        //CUDA_TS(BitVolume_MarchingCubes);
        //std::vector<float3> vertices;
        //std::vector<int3> faces;
        //bitVolume.MarchingCubes(vertices, faces);

        //PLYFormat mesh;
        //for (size_t i = 0; i < vertices.size(); i++)
        //{
        //    mesh.AddPoint(vertices[i].x, vertices[i].y, vertices[i].z);
        //}

        //for (size_t i = 0; i < faces.size(); i++)
        //{
        //    mesh.AddTriangleIndex(faces[i].x);
        //    mesh.AddTriangleIndex(faces[i].y);
        //    mesh.AddTriangleIndex(faces[i].z);
        //}

        //mesh.Serialize("../../res/3D/MarchingCubes.ply");

        //CUDA_TE(BitVolume_MarchingCubes);

        //unsigned int h_numberOfPoints = 0;
        //CUDA_COPY_D2H(&h_numberOfPoints, d_numberOfPoints, sizeof(unsigned int));

        //float3* h_points = new float3[pointCloud.numberOfPoints];
        //CUDA_COPY_D2H(h_points, d_points, sizeof(float3) * h_numberOfPoints);


        //PLYFormat ply;
        //for (size_t i = 0; i < h_numberOfPoints; i++)
        //{
        //    auto& p = h_points[i];

        //    ply.AddPoint(p.x, p.y, p.z);
        //}
        //ply.Serialize("../../res/3D/Temp.ply");

        //CUDA_TS(BitVolume_Terminate);
        //bitVolume.Terminate();
        //CUDA_TE(BitVolume_Terminate);

        //CUDA_TE(BitVolume);
    }










    {
        //float voxelSize = 0.1f;
        //float radius = 10.0f;

        //SpherePointGenerator generator(voxelSize, radius);
        //int gridSize = generator.gridSize;
        //int totalVoxels = gridSize * gridSize * gridSize;

        //thrust::device_vector<float3> d_points(totalVoxels);

        //// Generate all points (with NAN for out-of-sphere points)
        //thrust::transform(
        //    thrust::counting_iterator<int>(0),
        //    thrust::counting_iterator<int>(totalVoxels),
        //    d_points.begin(),
        //    generator
        //);

        //// Remove invalid (NAN) points
        //auto new_end = thrust::remove_if(
        //    d_points.begin(),
        //    d_points.end(),
        //    IsValidPoint{}
        //);
        //d_points.erase(new_end, d_points.end());
    }







    {
        //HashMap<uint64_t, HashMapVoxel> hashmap;
        //hashmap.Initialize(pointCloud.numberOfPoints * hashmap.info.maxProbe / 4);

        //LaunchKernel(Kernel_OccupyPointCloud, pointCloud.numberOfPoints, hashmap, pointCloud);

        //hashmap.Terminate();
    }
}

