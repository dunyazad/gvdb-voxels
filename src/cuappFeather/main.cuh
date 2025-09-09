#include <iostream>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

#include <cuda_common.cuh>

#include <SimplePointCloud.cuh>
#include <SimpleHashMap.hpp>
#include <BitVolume.cuh>
#include <DenseGrid.hpp>
#include <VoxelHashMap.cuh>
#include <SCVoxelHashMap.cuh>
#include <HalfEdgeMesh.cuh>
#include <VEFM.cuh>
#include <HalfEdgeMeshInterop.h>
#include <ThrustHalfEdgeMesh.cuh>
#include <Octree.cuh>
#include <LBVH.cuh>
#include <MarginLineFinder.cuh>
#include <HPOctree.cuh>

#include <Eigen/Core>
#include <Eigen/Dense>
namespace Eigen {
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}


class CUDAInstance
{
public:
    CUDAInstance();
    ~CUDAInstance();

    HostPointCloud<PointCloudProperty> h_input;
    DevicePointCloud<PointCloudProperty> d_input;
    SCVoxelHashMap<PointCloudProperty> vhm;
    HostHalfEdgeMesh<PointCloudProperty> h_mesh;
    DeviceHalfEdgeMesh<PointCloudProperty> d_mesh;

    HalfEdgeMeshInterop interop;

    HostPointCloud<PointCloudProperty> ProcessPointCloud(const HostPointCloud<PointCloudProperty>& input, float voxelSize = 0.2f, unsigned int occupyOffset = 3);
    void ProcessHalfEdgeMesh(const string& filename);
};
