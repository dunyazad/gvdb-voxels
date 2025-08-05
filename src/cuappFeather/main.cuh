#include <iostream>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

#include <cuda_common.cuh>

#include <PointCloud.cuh>
#include <HashMap.hpp>
#include <BitVolume.cuh>
#include <DenseGrid.hpp>
#include <VoxelHashMap.cuh>
#include <SCVoxelHashMap.cuh>
#include <HalfEdgeMesh.cuh>
#include <VEFM.cuh>
#include <HalfEdgeMeshInterop.h>
#include <ThrustHalfEdgeMesh.cuh>

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

    DevicePointCloud d_input;
    SCVoxelHashMap vhm;
    HostHalfEdgeMesh h_mesh;
    DeviceHalfEdgeMesh d_mesh;

    HalfEdgeMeshInterop interop;

    HostPointCloud ProcessPointCloud(const HostPointCloud& h_input);
    void ProcessHalfEdgeMesh(const string& filename);
};
