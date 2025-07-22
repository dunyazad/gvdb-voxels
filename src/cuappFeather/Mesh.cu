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

	cudaMemcpy(positions, other.positions, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(normals, other.normals, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, other.colors, sizeof(float3) * other.numberOfPoints, cudaMemcpyDeviceToHost);

	cudaMemcpy(faces, other.faces, sizeof(uint3) * other.numberOfFaces, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

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

	cudaMemcpy(positions, other.positions, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(normals, other.normals, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(colors, other.colors, sizeof(float3) * other.numberOfPoints, cudaMemcpyHostToDevice);

	cudaMemcpy(faces, other.faces, sizeof(uint3) * other.numberOfFaces, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	return *this;
}

void DeviceMesh::Intialize(unsigned int numberOfPoints, unsigned int numberOfFaces)
{
	if (numberOfPoints == 0) return;

	this->numberOfPoints = numberOfPoints;
	cudaMalloc(&positions, sizeof(float3) * numberOfPoints);
	cudaMalloc(&normals, sizeof(float3) * numberOfPoints);
	cudaMalloc(&colors, sizeof(float3) * numberOfPoints);

	this->numberOfFaces = numberOfFaces;
	cudaMalloc(&faces, sizeof(uint3) * numberOfFaces);
}

void DeviceMesh::Terminate()
{
	if (numberOfPoints > 0)
	{
		cudaFree(positions);
		cudaFree(normals);
		cudaFree(colors);

		positions = nullptr;
		normals = nullptr;
		colors = nullptr;
		numberOfPoints = 0;

		cudaFree(faces);

		faces = nullptr;
		numberOfFaces = 0;

		//cudaDeviceSynchronize();
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
	//cudaMalloc(&new_positions, sizeof(float3) * numberOfPoints);
	//cudaMalloc(&new_normals, sizeof(float3) * numberOfPoints);
	//cudaMalloc(&new_colors, sizeof(float3) * numberOfPoints);

	//unsigned int* d_valid_count = nullptr;
	//cudaMalloc(&d_valid_count, sizeof(unsigned int));
	//cudaMemset(d_valid_count, 0, sizeof(unsigned int));

	//LaunchKernel(Kernel_DeviceMeshCompactValidPoints, numberOfPoints,
	//	positions, normals, colors,
	//	new_positions, new_normals, new_colors,
	//	d_valid_count, numberOfPoints);

	//unsigned int valid_count = 0;
	//cudaMemcpy(&valid_count, d_valid_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//cudaFree(positions);
	//cudaFree(normals);
	//cudaFree(colors);

	//positions = new_positions;
	//normals = new_normals;
	//colors = new_colors;
	//numberOfPoints = valid_count;

	//cudaFree(d_valid_count);
	//cudaDeviceSynchronize();
}