#pragma once

#include <cuda_common.cuh>

struct BitVolumeInfo
{
    dim3 dimensions = dim3(1000, 1000, 1000);
    unsigned int numberOfVoxels = 1000 * 1000 * 1000;

    dim3 allocated_dimensions = dim3(0, 0, 0);
    unsigned int numberOfAllocatedWords = 0;

    float voxelSize = 0.1f;
    float3 volumeMin;
    uint32_t* d_volume = nullptr;

    bool* d_simplePointLUT = nullptr;
};

__global__ void Kernel_OccupyFromPoints(BitVolumeInfo info, const float3* points, int numPoints);
__global__ void Kernel_OccupyFromEigenPoints(BitVolumeInfo info, const Eigen::Vector3f* points, int numPoints);
__global__ void Kernel_SerializeToFloat3(BitVolumeInfo info, float3* outPoints, unsigned int* outCount);
__global__ void Kernel_MarchingCubes(BitVolumeInfo info, float3* vertices, int3* faces, unsigned int* vCount, unsigned int* fCount);

struct BitVolume
{
    BitVolumeInfo info;

    void Initialize(dim3 dimensions = dim3(1000, 1000, 1000), float voxelSize = 0.1f)
    {
        info.voxelSize = voxelSize;
        info.dimensions = dimensions;
        info.numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

        unsigned int x = static_cast<unsigned int>(ceilf(dimensions.x / 32.0f));
        unsigned int y = dimensions.y;
        unsigned int z = dimensions.z;

        info.allocated_dimensions = dim3(x, y, z);
        info.numberOfAllocatedWords = x * y * z;

        cudaMalloc(&info.d_volume, sizeof(uint32_t) * info.numberOfAllocatedWords);
        cudaMemset(info.d_volume, 0, sizeof(uint32_t) * info.numberOfAllocatedWords);
    }

    void Terminate()
    {
        if (info.d_volume) cudaFree(info.d_volume);
        info.d_volume = nullptr;

        if (info.d_simplePointLUT) cudaFree(info.d_simplePointLUT);
        info.d_simplePointLUT = nullptr;
    }

    void Clear()
    {
        cudaMemset(info.d_volume, 0, sizeof(uint32_t) * info.numberOfAllocatedWords);
    }

    __device__ static bool isOccupied(const BitVolumeInfo& info, dim3 voxelIndex)
    {
        if (voxelIndex.x >= info.dimensions.x ||
            voxelIndex.y >= info.dimensions.y ||
            voxelIndex.z >= info.dimensions.z)
            return false;

        unsigned int wordX = voxelIndex.x / 32;
        unsigned int bitX = voxelIndex.x % 32;

        unsigned int wordIndex = voxelIndex.z * info.allocated_dimensions.y * info.allocated_dimensions.x +
            voxelIndex.y * info.allocated_dimensions.x +
            wordX;

        uint32_t word = info.d_volume[wordIndex];
        return (word >> bitX) & 1;
    }

    __device__ static void setOccupied(BitVolumeInfo& info, dim3 voxelIndex)
    {
        if (voxelIndex.x >= info.dimensions.x ||
            voxelIndex.y >= info.dimensions.y ||
            voxelIndex.z >= info.dimensions.z)
            return;

        unsigned int wordX = voxelIndex.x / 32;
        unsigned int bitX = voxelIndex.x % 32;

        unsigned int wordIndex = voxelIndex.z * info.allocated_dimensions.y * info.allocated_dimensions.x +
            voxelIndex.y * info.allocated_dimensions.x +
            wordX;

        atomicOr(&info.d_volume[wordIndex], 1u << bitX);
    }

    __device__ static void setNotOccupied(BitVolumeInfo& info, dim3 voxelIndex)
    {
        if (voxelIndex.x >= info.dimensions.x ||
            voxelIndex.y >= info.dimensions.y ||
            voxelIndex.z >= info.dimensions.z)
            return;

        unsigned int wordX = voxelIndex.x / 32;
        unsigned int bitX = voxelIndex.x % 32;

        unsigned int wordIndex = voxelIndex.z * info.allocated_dimensions.y * info.allocated_dimensions.x +
            voxelIndex.y * info.allocated_dimensions.x +
            wordX;

        uint32_t mask = ~(1u << bitX);

        atomicAnd(&info.d_volume[wordIndex], mask);
    }

    void OccupyFromPoints(float3 volumeMin, const float3* d_points, int numPoints)
    {
        info.volumeMin = volumeMin;

        const int threads = 256;
        const int blocks = (numPoints + threads - 1) / threads;

        Kernel_OccupyFromPoints << <blocks, threads >> > (info, d_points, numPoints);
        CUDA_SYNC();  // 필요시
    }

    void OccupyFromEigenPoints(float3 volumeMin, const Eigen::Vector3f* d_points, int numPoints)
    {
        info.volumeMin = volumeMin;

        const int threads = 256;
        const int blocks = (numPoints + threads - 1) / threads;
        Kernel_OccupyFromEigenPoints << <blocks, threads >> > (info, d_points, numPoints);
        CUDA_SYNC();
    }

    void SerializeToFloat3(float3* d_output, unsigned int* d_outputCount)
    {
        const int threads = 256;
        const int blocks = (info.numberOfAllocatedWords + threads - 1) / threads;

        cudaMemset(d_outputCount, 0, sizeof(unsigned int));

        Kernel_SerializeToFloat3 << <blocks, threads >> > (info, d_output, d_outputCount);
        CUDA_SYNC();  // optional
    }

    __device__ static float getVoxelCornerValue(const BitVolumeInfo& info, dim3 base, int dx, int dy, int dz)
    {
        dim3 pos(base.x + dx, base.y + dy, base.z + dz);
        return BitVolume::isOccupied(info, pos) ? 1.0f : 0.0f;
    }

    __device__ static float3 getEdgeVertexPosition(const BitVolumeInfo& info, dim3 base, int edge)
    {
        // Corner offsets in voxel space
        const int3 cornerOffsets[8] = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
        };

        const int2 edgeToCorners[12] = {
            {0,1}, {1,2}, {2,3}, {3,0}, // bottom face
            {4,5}, {5,6}, {6,7}, {7,4}, // top face
            {0,4}, {1,5}, {2,6}, {3,7}  // vertical edges
        };

        int2 cornerIdx = edgeToCorners[edge];
        int3 offsetA = cornerOffsets[cornerIdx.x];
        int3 offsetB = cornerOffsets[cornerIdx.y];

        float3 worldA = make_float3(
            info.volumeMin.x + (base.x + offsetA.x) * info.voxelSize,
            info.volumeMin.y + (base.y + offsetA.y) * info.voxelSize,
            info.volumeMin.z + (base.z + offsetA.z) * info.voxelSize
        );

        float3 worldB = make_float3(
            info.volumeMin.x + (base.x + offsetB.x) * info.voxelSize,
            info.volumeMin.y + (base.y + offsetB.y) * info.voxelSize,
            info.volumeMin.z + (base.z + offsetB.z) * info.voxelSize
        );

        // simple midpoint (binary volume)
        return make_float3(
            0.5f * (worldA.x + worldB.x),
            0.5f * (worldA.y + worldB.y),
            0.5f * (worldA.z + worldB.z)
        );
    }

    void MarchingCubes(std::vector<float3>& vertices, std::vector<int3>& faces)
    {
        constexpr int MAX_TRIANGLES = 10 * 1000 * 1000; // 예: 최대 1천만 개 삼각형
        constexpr int MAX_VERTICES = MAX_TRIANGLES * 3;

        float3* d_vertices;
        int3* d_faces;
        unsigned int* d_vCount;
        unsigned int* d_fCount;

        cudaMalloc(&d_vertices, sizeof(float3) * MAX_VERTICES);
        cudaMalloc(&d_faces, sizeof(int3) * MAX_TRIANGLES);
        cudaMalloc(&d_vCount, sizeof(unsigned int));
        cudaMalloc(&d_fCount, sizeof(unsigned int));
        cudaMemset(d_vCount, 0, sizeof(unsigned int));
        cudaMemset(d_fCount, 0, sizeof(unsigned int));

        dim3 threads(8, 8, 8);
        dim3 blocks(
            (info.dimensions.x + threads.x - 2) / (threads.x),
            (info.dimensions.y + threads.y - 2) / (threads.y),
            (info.dimensions.z + threads.z - 2) / (threads.z)
        );

        CUDA_TS(MarchingCubes);
        Kernel_MarchingCubes << <blocks, threads >> > (
            info, d_vertices, d_faces, d_vCount, d_fCount
            );
        CUDA_TE(MarchingCubes);

        unsigned int vertexCount = 0, faceCount = 0;
        CUDA_COPY_D2H(&vertexCount, d_vCount, sizeof(unsigned int));
        CUDA_COPY_D2H(&faceCount, d_fCount, sizeof(unsigned int));

        vertices.resize(vertexCount);
        faces.resize(faceCount);

        CUDA_COPY_D2H(vertices.data(), d_vertices, vertexCount * sizeof(float3));
        CUDA_COPY_D2H(faces.data(), d_faces, faceCount * sizeof(int3));

        cudaFree(d_vertices);
        cudaFree(d_faces);
        cudaFree(d_vCount);
        cudaFree(d_fCount);
    }
};

__global__ void Kernel_OccupyFromPoints(BitVolumeInfo info, const float3* points, int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float3 p = points[idx];

    float3 rel = make_float3(
        p.x - info.volumeMin.x,
        p.y - info.volumeMin.y,
        p.z - info.volumeMin.z);

    if (rel.x < 0 || rel.y < 0 || rel.z < 0) return;

    unsigned int x = static_cast<unsigned int>(rel.x / info.voxelSize);
    unsigned int y = static_cast<unsigned int>(rel.y / info.voxelSize);
    unsigned int z = static_cast<unsigned int>(rel.z / info.voxelSize);

    if (x >= info.dimensions.x || y >= info.dimensions.y || z >= info.dimensions.z)
        return;

    unsigned int wordX = x / 32;
    unsigned int bitX = x % 32;

    unsigned int wordIndex = z * info.allocated_dimensions.y * info.allocated_dimensions.x +
        y * info.allocated_dimensions.x +
        wordX;

    atomicOr(&info.d_volume[wordIndex], 1u << bitX);
}

__global__ void Kernel_OccupyFromEigenPoints(BitVolumeInfo info, const Eigen::Vector3f* points, int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Eigen::Vector3f p = points[idx];

    float3 rel = make_float3(
        p.x() - info.volumeMin.x,
        p.y() - info.volumeMin.y,
        p.z() - info.volumeMin.z);

    if (rel.x < 0 || rel.y < 0 || rel.z < 0) return;

    unsigned int x = static_cast<unsigned int>(rel.x / info.voxelSize);
    unsigned int y = static_cast<unsigned int>(rel.y / info.voxelSize);
    unsigned int z = static_cast<unsigned int>(rel.z / info.voxelSize);

    if (x >= info.dimensions.x || y >= info.dimensions.y || z >= info.dimensions.z)
        return;

    unsigned int wordX = x / 32;
    unsigned int bitX = x % 32;

    unsigned int wordIndex = z * info.allocated_dimensions.y * info.allocated_dimensions.x +
        y * info.allocated_dimensions.x +
        wordX;

    atomicOr(&info.d_volume[wordIndex], 1u << bitX);
}

__global__ void Kernel_SerializeToFloat3(BitVolumeInfo info, float3* outPoints, unsigned int* outCount)
{
    int wordIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (wordIdx >= info.numberOfAllocatedWords) return;

    uint32_t word = info.d_volume[wordIdx];
    if (word == 0) return;

    // Compute base voxel coordinates from wordIdx
    int wordsPerY = info.allocated_dimensions.x;
    int wordsPerZ = info.allocated_dimensions.y * wordsPerY;

    int z = wordIdx / wordsPerZ;
    int y = (wordIdx % wordsPerZ) / wordsPerY;
    int wordX = wordIdx % wordsPerY;

    for (int bit = 0; bit < 32; ++bit)
    {
        if ((word >> bit) & 1)
        {
            int x = wordX * 32 + bit;
            if (x >= info.dimensions.x) continue;  // padding bits check

            float3 pos;
            pos.x = info.volumeMin.x + x * info.voxelSize;
            pos.y = info.volumeMin.y + y * info.voxelSize;
            pos.z = info.volumeMin.z + z * info.voxelSize;

            int writeIdx = atomicAdd(outCount, 1);
            outPoints[writeIdx] = pos;
        }
    }
}

__global__ void Kernel_MarchingCubes(BitVolumeInfo info, float3* vertices, int3* faces, unsigned int* vCount, unsigned int* fCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= info.dimensions.x - 1 || y >= info.dimensions.y - 1 || z >= info.dimensions.z - 1) return;

    dim3 base(x, y, z);

    int cubeIndex = 0;
    float corner[8];

    for (int i = 0; i < 8; ++i) {
        int dx = i & 1;
        int dy = (i >> 1) & 1;
        int dz = (i >> 2) & 1;

        corner[i] = BitVolume::getVoxelCornerValue(info, base, dx, dy, dz);
        if (corner[i] > 0.5f) cubeIndex |= (1 << i);
    }

    if (cubeIndex == 0 || cubeIndex == 255) return;

    const int* tri = MC_TRI_TABLE[cubeIndex];  // Needs to be defined statically

    for (int i = 0; tri[i] != -1; i += 3) {
        float3 triVerts[3];

        for (int j = 0; j < 3; ++j) {
            int edge = tri[i + j];
            float3 p0 = BitVolume::getEdgeVertexPosition(info, base, edge);
            triVerts[j] = p0;
        }

        int baseIdx = atomicAdd(vCount, 3);
        for (int j = 0; j < 3; ++j)
            vertices[baseIdx + j] = triVerts[j];

        int fIdx = atomicAdd(fCount, 1);
        faces[fIdx] = make_int3(baseIdx, baseIdx + 1, baseIdx + 2);
    }
}
