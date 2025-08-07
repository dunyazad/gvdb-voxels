#include <HalfEdgeMeshInterop.h>

#include <libFeather.h>

HalfEdgeMeshInterop::HalfEdgeMeshInterop() {}
HalfEdgeMeshInterop::~HalfEdgeMeshInterop()
{
    Terminate();
}

void HalfEdgeMeshInterop::Initialize(Renderable* renderable)
{
    this->renderable = renderable;

    glBindVertexArray(renderable->GetVAO());

    // VBO/EBO 핸들
    GLuint vboPos = renderable->GetVertices().vbo;
    GLuint vboNormal = renderable->GetNormals().vbo;
    GLuint vboColor = renderable->GetColors3().vbo;
    GLuint ebo = renderable->GetIndices().vbo;
    numVertices = renderable->GetVertices().size();
    numIndices = renderable->GetIndices().size();

    // VBO 등록
    cudaGraphicsGLRegisterBuffer(&cudaVboPos, vboPos, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&cudaVboNormal, vboNormal, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&cudaVboColor, vboColor, cudaGraphicsRegisterFlagsNone);
    // EBO 등록
    cudaGraphicsGLRegisterBuffer(&cudaEbo, ebo, cudaGraphicsRegisterFlagsNone);
}

void HalfEdgeMeshInterop::Terminate()
{
    if (cudaVboPos) { cudaGraphicsUnregisterResource(cudaVboPos);    cudaVboPos = nullptr; }
    if (cudaVboNormal) { cudaGraphicsUnregisterResource(cudaVboNormal); cudaVboNormal = nullptr; }
    if (cudaVboColor) { cudaGraphicsUnregisterResource(cudaVboColor);  cudaVboColor = nullptr; }
    if (cudaEbo) { cudaGraphicsUnregisterResource(cudaEbo); cudaEbo = nullptr; }
}

void HalfEdgeMeshInterop::UploadFromDevice(DeviceHalfEdgeMesh& deviceMesh)
{
    if (deviceMesh.numberOfPoints > numIndices)
    {
        numVertices = deviceMesh.numberOfPoints;
        numIndices = deviceMesh.numberOfFaces * 3;

        renderable->GetIndices().resize(numIndices);
        renderable->GetVertices().resize(numVertices);
        renderable->GetNormals().resize(numVertices);
        renderable->GetColors3().resize(numVertices);

        renderable->GetIndices().Update();
        renderable->GetVertices().Update();
        renderable->GetNormals().Update();
        renderable->GetColors3().Update();

        Initialize(renderable);
    }

    // CUDA-OpenGL interop map
    cudaGraphicsMapResources(1, &cudaVboPos, 0);
    cudaGraphicsMapResources(1, &cudaVboNormal, 0);
    cudaGraphicsMapResources(1, &cudaVboColor, 0);
    cudaGraphicsMapResources(1, &cudaEbo, 0);

    float3* d_vboPos = nullptr;
    float3* d_vboNormal = nullptr;
    float3* d_vboColor = nullptr;
    uint32_t* d_ebo = nullptr;
    size_t  sizePos, sizeNormal, sizeColor, sizeEbo;

    cudaGraphicsResourceGetMappedPointer((void**)&d_vboPos, &sizePos, cudaVboPos);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboNormal, &sizeNormal, cudaVboNormal);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboColor, &sizeColor, cudaVboColor);
    cudaGraphicsResourceGetMappedPointer((void**)&d_ebo, &sizeEbo, cudaEbo);

    // DeviceHalfEdgeMesh의 device pointer에서 VBO로 직접 복사
    CUDA_COPY_D2D(d_vboPos, deviceMesh.positions, sizeof(float3) * numVertices);
    CUDA_COPY_D2D(d_vboNormal, deviceMesh.normals, sizeof(float3) * numVertices);
    CUDA_COPY_D2D(d_vboColor, deviceMesh.colors, sizeof(float3) * numVertices);
    CUDA_COPY_D2D(d_ebo, deviceMesh.faces, sizeof(uint32_t) * numIndices);

    cudaGraphicsUnmapResources(1, &cudaVboPos, 0);
    cudaGraphicsUnmapResources(1, &cudaVboNormal, 0);
    cudaGraphicsUnmapResources(1, &cudaVboColor, 0);
    cudaGraphicsUnmapResources(1, &cudaEbo, 0);
}
