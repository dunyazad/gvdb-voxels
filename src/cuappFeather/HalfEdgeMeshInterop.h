#pragma once

#include <cuda_common.cuh>

#include <HalfEdgeMesh.cuh>

class Renderable;

struct PointCloudProperty
{
    int deepLearningClass = INT32_MAX;
    int label = INT32_MAX;
};

class HalfEdgeMeshInterop
{
public:
    HalfEdgeMeshInterop();
    ~HalfEdgeMeshInterop();

    // Renderable에서 VBO 받아서 interop 세팅 (vertex/normal/color)
    void Initialize(Renderable* renderable);

    // DeviceHalfEdgeMesh에서 vertex/normal/color를 바로 VBO에 복사
    void UploadFromDevice(DeviceHalfEdgeMesh<PointCloudProperty>& deviceMesh);

    // 필요시 자원 해제
    void Terminate();

	inline bool IsInitialized() const { return initialized; }

private:
	bool initialized = false;
    Renderable* renderable = nullptr;

    cudaGraphicsResource* cudaVboPos = nullptr;
    cudaGraphicsResource* cudaVboNormal = nullptr;
    cudaGraphicsResource* cudaVboColor = nullptr;
    cudaGraphicsResource* cudaEbo = nullptr;

    unsigned int numVertices = 0;
    unsigned int numIndices = 0;
};
