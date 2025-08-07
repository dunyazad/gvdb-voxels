#pragma once

#include <Mesh.cuh>

HostMesh::HostMesh() {}

HostMesh::HostMesh(const DeviceMesh& other)
{
	operator =(other);
}

HostMesh& HostMesh::operator=(const DeviceMesh& other)
{
	Terminate();
	Intialize(other.numberOfPoints, other.numberOfFaces);

	CUDA_COPY_D2H(positions, other.positions, sizeof(float3) * other.numberOfPoints);
	CUDA_COPY_D2H(normals, other.normals, sizeof(float3) * other.numberOfPoints);
	CUDA_COPY_D2H(colors, other.colors, sizeof(float3) * other.numberOfPoints);

	CUDA_COPY_D2H(faces, other.faces, sizeof(uint3) * other.numberOfFaces);

	CUDA_SYNC();

	return *this;
}

void HostMesh::Intialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
	if (numberOfPoints == 0) return;

	this->numberOfPoints = numberOfPoints;
	positions = new float3[numberOfPoints];
	normals = new float3[numberOfPoints];
	colors = new float3[numberOfPoints];

	this->numberOfFaces = numberOfFaces;
	faces = new uint3[numberOfFaces];
}

void HostMesh::Terminate()
{
	if (numberOfPoints > 0)
	{
		delete[] positions;
		delete[] normals;
		delete[] colors;

		positions = nullptr;
		normals = nullptr;
		colors = nullptr;
		numberOfPoints = 0;

		delete[] faces;

		faces = nullptr;
		numberOfFaces = 0;
	}
}

void HostMesh::CompactValidPoints()
{
	//std::vector<float3> new_positions, new_normals, new_colors;
	//new_positions.reserve(numberOfPoints);
	//new_normals.reserve(numberOfPoints);
	//new_colors.reserve(numberOfPoints);

	//for (unsigned int i = 0; i < numberOfPoints; ++i)
	//{
	//	const auto& p = positions[i];
	//	if (p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX)
	//	{
	//		new_positions.push_back(p);
	//		new_normals.push_back(normals[i]);
	//		new_colors.push_back(colors[i]);
	//	}
	//}

	//Terminate();
	//Intialize(static_cast<unsigned int>(new_positions.size()));
	//std::copy(new_positions.begin(), new_positions.end(), positions);
	//std::copy(new_normals.begin(), new_normals.end(), normals);
	//std::copy(new_colors.begin(), new_colors.end(), colors);
}

DeviceMesh::DeviceMesh() {}

DeviceMesh::DeviceMesh(const HostMesh& other)
{
	operator =(other);
}

DeviceMesh& DeviceMesh::operator=(const HostMesh& other)
{
	Terminate();
	Intialize(other.numberOfPoints, other.numberOfFaces);

	CUDA_COPY_H2D(positions, other.positions, sizeof(float3) * other.numberOfPoints);
	CUDA_COPY_H2D(normals, other.normals, sizeof(float3) * other.numberOfPoints);
	CUDA_COPY_H2D(colors, other.colors, sizeof(float3) * other.numberOfPoints);

	CUDA_COPY_H2D(faces, other.faces, sizeof(uint3) * other.numberOfFaces);

	CUDA_SYNC();

	return *this;
}

void DeviceMesh::Intialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
	if (numberOfPoints == 0) return;

	this->numberOfPoints = numberOfPoints;
	CUDA_MALLOC(&positions, sizeof(float3) * numberOfPoints);
	CUDA_MALLOC(&normals, sizeof(float3) * numberOfPoints);
	CUDA_MALLOC(&colors, sizeof(float3) * numberOfPoints);

	this->numberOfFaces = numberOfFaces;
	CUDA_MALLOC(&faces, sizeof(uint3) * numberOfFaces);
}

void DeviceMesh::Terminate()
{
	if (numberOfPoints > 0)
	{
		CUDA_FREE(positions);
		CUDA_FREE(normals);
		CUDA_FREE(colors);

		positions = nullptr;
		normals = nullptr;
		colors = nullptr;
		numberOfPoints = 0;

		CUDA_FREE(faces);

		faces = nullptr;
		numberOfFaces = 0;

		//CUDA_SYNC();
	}
}

__global__ void Kernel_DeviceMeshCompactValidPoints(
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

void DeviceMesh::CompactValidPoints()
{
	//if (numberOfPoints == 0) return;

	//float3* new_positions;
	//float3* new_normals;
	//float3* new_colors;
	//CUDA_MALLOC(&new_positions, sizeof(float3) * numberOfPoints);
	//CUDA_MALLOC(&new_normals, sizeof(float3) * numberOfPoints);
	//CUDA_MALLOC(&new_colors, sizeof(float3) * numberOfPoints);

	//unsigned int* d_valid_count = nullptr;
	//CUDA_MALLOC(&d_valid_count, sizeof(unsigned int));
	//CUDA_MEMSET(d_valid_count, 0, sizeof(unsigned int));

	//LaunchKernel(Kernel_DeviceMeshCompactValidPoints, numberOfPoints,
	//	positions, normals, colors,
	//	new_positions, new_normals, new_colors,
	//	d_valid_count, numberOfPoints);

	//unsigned int valid_count = 0;
	//CUDA_COPY_D2H(&valid_count, d_valid_count, sizeof(unsigned int));

	//CUDA_FREE(positions);
	//CUDA_FREE(normals);
	//CUDA_FREE(colors);

	//positions = new_positions;
	//normals = new_normals;
	//colors = new_colors;
	//numberOfPoints = valid_count;

	//CUDA_FREE(d_valid_count);
	//CUDA_SYNC();
}