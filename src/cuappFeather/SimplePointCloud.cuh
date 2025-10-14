#pragma once

#include <cuda_common.cuh>

#include <Serialization.hpp>

#include <cuBQL/bvh.h>
#include <cuBQL/queries/pointData/findClosest.h>
#include <cuBQL/queries/pointData/knn.h>

template<typename T = void> struct HostPointCloud;
template<typename T = void> struct DevicePointCloud;

template<typename T>
struct HostPointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    T* properties = nullptr;

    HostPointCloud() {}

    HostPointCloud(const HostPointCloud<T>& other) { operator =(other); }
    HostPointCloud<T>& operator=(const HostPointCloud<T>& other)
    {
        Terminate();
        Initialize(other.numberOfPoints);

        CUDA_COPY_H2H(positions, other.positions, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_H2H(normals, other.normals, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_H2H(colors, other.colors, sizeof(float3) * other.numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            CUDA_COPY_H2H(properties, other.properties, sizeof(T) * other.numberOfPoints);
        }

        return *this;
    }

    HostPointCloud(const DevicePointCloud<T>& other) { operator =(other); }
    HostPointCloud<T>& operator=(const DevicePointCloud<T>& other)
    {
        Terminate();
        Initialize(other.numberOfPoints);

        CUDA_COPY_D2H(positions, other.positions, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_D2H(normals, other.normals, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_D2H(colors, other.colors, sizeof(float3) * other.numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            CUDA_COPY_D2H(properties, other.properties, sizeof(T) * other.numberOfPoints);
        }

        return *this;
    }

    void Initialize(unsigned int numberOfPoints)
    {
        if (numberOfPoints == 0) return;

        this->numberOfPoints = numberOfPoints;
        positions = new float3[numberOfPoints];
        normals = new float3[numberOfPoints];
        colors = new float3[numberOfPoints];

        if constexpr (!std::is_void_v<T>)
        {
            properties = new T[numberOfPoints];
        }
    }

    void Terminate()
    {
        if (numberOfPoints > 0)
        {
            delete[] positions;
            delete[] normals;
            delete[] colors;

            if constexpr (!std::is_void_v<T>)
            {
                delete[] properties;
                properties = nullptr;
            }

            positions = nullptr;
            normals = nullptr;
            colors = nullptr;
            numberOfPoints = 0;
        }
    }

    void DeserializePLY(const std::string& filename)
    {
        PLYFormat ply;
        ply.Deserialize(filename);

        auto n = ply.GetPoints().size() / 3;

        if (0 == n) return;

        Terminate();
        Initialize(n);

        numberOfPoints = n;
        for (size_t i = 0; i < n; i++)
        {
            positions[i].x = ply.GetPoints()[i * 3 + 0];
            positions[i].y = ply.GetPoints()[i * 3 + 1];
            positions[i].z = ply.GetPoints()[i * 3 + 2];
            
            normals[i].x = ply.GetNormals()[i * 3 + 0];
            normals[i].y = ply.GetNormals()[i * 3 + 1];
            normals[i].z = ply.GetNormals()[i * 3 + 2];

            colors[i].x = ply.GetColors()[i * 3 + 0];
            colors[i].y = ply.GetColors()[i * 3 + 1];
            colors[i].z = ply.GetColors()[i * 3 + 2];

            if constexpr (!std::is_void_v<T>)
            {
                properties[i].deepLearningClass = ply.GetDeepLearningClasses()[i];
                properties[i].label = ply.GetLabels()[i];
            }
        }

        CompactValidPoints();
    }

    void SerializePLY(const std::string& filename)
    {
        PLYFormat ply;

        ply.AddData((float*)positions, (float*)normals, (float*)colors, numberOfPoints, false);

        //for (size_t i = 0; i < numberOfPoints; i++)
        //{
        //    if constexpr (!std::is_void_v<T>)
        //    {
        //        ply.AddExtraAttrib(properties[i]);
        //    }
        //}

        ply.Serialize(filename);
    }

    void CompactValidPoints()
    {
        std::vector<float3> new_positions, new_normals, new_colors;
        new_positions.reserve(numberOfPoints);
        new_normals.reserve(numberOfPoints);
        new_colors.reserve(numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            std::vector<T> new_properties;
            new_properties.reserve(numberOfPoints);

            for (unsigned int i = 0; i < numberOfPoints; ++i)
            {
                const auto& p = positions[i];
                if (p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX)
                {
                    new_positions.push_back(p);
                    new_normals.push_back(normals[i]);
                    new_colors.push_back(colors[i]);
                    new_properties.push_back(properties[i]);
                }
            }

            Terminate();
            Initialize(static_cast<unsigned int>(new_positions.size()));
            std::copy(new_positions.begin(), new_positions.end(), positions);
            std::copy(new_normals.begin(), new_normals.end(), normals);
            std::copy(new_colors.begin(), new_colors.end(), colors);
            std::copy(new_properties.begin(), new_properties.end(), properties);
        }
        else
        {
            for (unsigned int i = 0; i < numberOfPoints; ++i)
            {
                const auto& p = positions[i];
                if (p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX)
                {
                    new_positions.push_back(p);
                    new_normals.push_back(normals[i]);
                    new_colors.push_back(colors[i]);
                }
            }

            Terminate();
            Initialize(static_cast<unsigned int>(new_positions.size()));
            std::copy(new_positions.begin(), new_positions.end(), positions);
            std::copy(new_normals.begin(), new_normals.end(), normals);
            std::copy(new_colors.begin(), new_colors.end(), colors);
        }
    }

    //HostPointCloud<T> Subset(const cuAABB& aabb)
    //{
    //    using PointType = std::remove_reference_t<decltype(positions[0])>;
    //    using NormalType = std::remove_reference_t<decltype(normals[0])>;
    //    using ColorType = std::remove_reference_t<decltype(colors[0])>;

    //    std::vector<PointType> temp_positions;
    //    std::vector<NormalType> temp_normals;
    //    std::vector<ColorType> temp_colors;

    //    std::vector<T> temp_properties;
    //    if constexpr (!std::is_void_v<T>) {
    //        temp_properties.reserve(numberOfPoints / 4);
    //    }

    //    temp_positions.reserve(numberOfPoints / 4);
    //    temp_normals.reserve(numberOfPoints / 4);
    //    temp_colors.reserve(numberOfPoints / 4);

    //    for (size_t i = 0; i < numberOfPoints; i++) {
    //        if (aabb.Contains(positions[i])) {
    //            temp_positions.push_back(positions[i]);
    //            temp_normals.push_back(normals[i]);
    //            temp_colors.push_back(colors[i]);

    //            if constexpr (!std::is_void_v<T>) {
    //                temp_properties.push_back(properties[i]);
    //            }
    //        }
    //    }

    //    HostPointCloud<T> result;
    //    result.Initialize(temp_positions.size());

    //    memcpy(result.positions, temp_positions.data(), temp_positions.size() * sizeof(PointType));
    //    memcpy(result.normals, temp_normals.data(), temp_normals.size() * sizeof(NormalType));
    //    memcpy(result.colors, temp_colors.data(), temp_colors.size() * sizeof(ColorType));

    //    if constexpr (!std::is_void_v<T>) {
    //        memcpy(result.properties, temp_properties.data(), temp_properties.size() * sizeof(T));
    //    }

    //    return result;
    //}

    void Subset(HostPointCloud<T>& result, const cuAABB& aabb)
    {
        using PointType = std::remove_reference_t<decltype(positions[0])>;
        using NormalType = std::remove_reference_t<decltype(normals[0])>;
        using ColorType = std::remove_reference_t<decltype(colors[0])>;

        std::vector<PointType> temp_positions;
        std::vector<NormalType> temp_normals;
        std::vector<ColorType> temp_colors;

        std::vector<T> temp_properties;
        if constexpr (!std::is_void_v<T>) {
            temp_properties.reserve(numberOfPoints / 4);
        }

        temp_positions.reserve(numberOfPoints / 4);
        temp_normals.reserve(numberOfPoints / 4);
        temp_colors.reserve(numberOfPoints / 4);

        for (size_t i = 0; i < numberOfPoints; i++) {
            if (aabb.Contains(positions[i])) {
                temp_positions.push_back(positions[i]);
                temp_normals.push_back(normals[i]);
                temp_colors.push_back(colors[i]);

                if constexpr (!std::is_void_v<T>) {
                    temp_properties.push_back(properties[i]);
                }
            }
        }

        result.Initialize(temp_positions.size());

        memcpy(result.positions, temp_positions.data(), temp_positions.size() * sizeof(PointType));
        memcpy(result.normals, temp_normals.data(), temp_normals.size() * sizeof(NormalType));
        memcpy(result.colors, temp_colors.data(), temp_colors.size() * sizeof(ColorType));

        if constexpr (!std::is_void_v<T>) {
            memcpy(result.properties, temp_properties.data(), temp_properties.size() * sizeof(T));
        }
    }
};

#ifdef __CUDACC__
template<typename T>
__global__ __forceinline__ void Kernel_DevicePointCloudCompactValidPoints(
    float3* in_positions, float3* in_normals, float3* in_colors,
    float3* out_positions, float3* out_normals, float3* out_colors,
    unsigned int* valid_counter, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 p = in_positions[idx];
    bool valid = (p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX);

    if (valid)
    {
        unsigned int writeIdx = atomicAdd(valid_counter, 1);
        out_positions[writeIdx] = p;
        out_normals[writeIdx] = in_normals[idx];
        out_colors[writeIdx] = in_colors[idx];
    }
}

template<typename T>
__global__ __forceinline__ void Kernel_Count(
    float3* positions, float3* normals, float3* colors,
    unsigned int* counter, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    atomicAdd(counter, 1);
}
#endif

void Call_DevicePointCloud_generateBoxes(
    cuBQL::box3f* boxForBuilder,
    const float3* positions,
    int numberOfPositions);

void Call_DevicePointCloud_gpuBuilder(
    cuBQL::bvh3f* bvh,
    cuBQL::box3f* boxes,
    unsigned int numberOfPoints);

template<typename T>
struct DevicePointCloud
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float3* colors = nullptr;
    unsigned int numberOfPoints = 0;

    T* properties = nullptr;

    cuBQL::bvh3f bvh;

    DevicePointCloud() {}

    DevicePointCloud(const DevicePointCloud<T>& other) { operator =(other); }
    DevicePointCloud& operator=(const DevicePointCloud<T>& other)
    {
        Terminate();
        Initialize(other.numberOfPoints);

        CUDA_COPY_D2D(positions, other.positions, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_D2D(normals, other.normals, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_D2D(colors, other.colors, sizeof(float3) * other.numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            CUDA_COPY_D2D(properties, other.properties, sizeof(T) * other.numberOfPoints);
        }
        CUDA_SYNC();

        return *this;
    }

    DevicePointCloud(const HostPointCloud<T>& other) { operator =(other); }

    DevicePointCloud<T>& operator=(const HostPointCloud<T>& other)
    {
        Terminate();
        Initialize(other.numberOfPoints);

        CUDA_COPY_H2D(positions, other.positions, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_H2D(normals, other.normals, sizeof(float3) * other.numberOfPoints);
        CUDA_COPY_H2D(colors, other.colors, sizeof(float3) * other.numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            CUDA_COPY_H2D(properties, other.properties, sizeof(T) * other.numberOfPoints);
        }
        CUDA_SYNC();

        return *this;
    }

    void Initialize(unsigned int numberOfPoints)
    {
        if (numberOfPoints == 0) return;

        this->numberOfPoints = numberOfPoints;
        CUDA_MALLOC(&positions, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&normals, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&colors, sizeof(float3) * numberOfPoints);

        if constexpr (!std::is_void_v<T>)
        {
            CUDA_MALLOC(&properties, sizeof(T) * numberOfPoints);
        }
    }

    void Terminate()
    {
        if (numberOfPoints > 0)
        {
            CUDA_SAFE_FREE(positions);
            CUDA_SAFE_FREE(normals);
            CUDA_SAFE_FREE(colors);

            if constexpr (!std::is_void_v<T>)
            {
                CUDA_SAFE_FREE(properties);
            }

            numberOfPoints = 0;
        }
    }

    void CompactValidPoints()
    {
#ifdef __CUDACC__
        if (numberOfPoints == 0) return;

        float3* new_positions;
        float3* new_normals;
        float3* new_colors;
        CUDA_MALLOC(&new_positions, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&new_normals, sizeof(float3) * numberOfPoints);
        CUDA_MALLOC(&new_colors, sizeof(float3) * numberOfPoints);

        unsigned int* d_valid_count = nullptr;
        CUDA_MALLOC(&d_valid_count, sizeof(unsigned int));
        CUDA_MEMSET(d_valid_count, 0, sizeof(unsigned int));

        LaunchKernel(Kernel_DevicePointCloudCompactValidPoints<T>, numberOfPoints,
            positions, normals, colors,
            new_positions, new_normals, new_colors,
            d_valid_count, numberOfPoints);

        unsigned int valid_count = 0;
        CUDA_COPY_D2H(&valid_count, d_valid_count, sizeof(unsigned int));

        CUDA_FREE(positions);
        CUDA_FREE(normals);
        CUDA_FREE(colors);

        positions = new_positions;
        normals = new_normals;
        colors = new_colors;
        numberOfPoints = valid_count;

        CUDA_FREE(d_valid_count);
        CUDA_SYNC();
#endif

        Count();
    }

    void Count()
    {
#ifdef __CUDACC__
        CUDA_TS(Count);

        unsigned int* d_counter = nullptr;
        CUDA_MALLOC(&d_counter, sizeof(unsigned int));
        CUDA_MEMSET(d_counter, 0, sizeof(unsigned int));

        LaunchKernel(Kernel_Count<T>, numberOfPoints,
            positions, normals, colors,
            d_counter, numberOfPoints);

        unsigned int counter = 0;
        CUDA_COPY_D2H(&counter, d_counter, sizeof(unsigned int));

        CUDA_FREE(d_counter);
        CUDA_SYNC();

        printf("Count : %d\n", counter);

        CUDA_TE(Count);
#endif
    }

    void UpdateBVH()
    {
        //cuBQL::cuda::free(bvh);

        CUDA_SAFE_FREE(boxes);
        CUDA_MALLOC(&boxes, numberOfPoints * sizeof(cuBQL::box3f));

        //bvh = cuBQL::bvh3f();

        CUDA_TS(InitializeBVH);

        Call_DevicePointCloud_generateBoxes(boxes, positions, numberOfPoints);

        Call_DevicePointCloud_gpuBuilder(&bvh, boxes, numberOfPoints);

        CUDA_SYNC();

        // free the boxes - we could actually re-use that memory below, but
        // let's just do this cleanly here.

        CUDA_TE(InitializeBVH);

        std::cout << "done building BVH over " << cuBQL::prettyNumber(numberOfPoints)
            << " points" << std::endl;
    }
};
