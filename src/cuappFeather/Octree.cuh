#pragma once

#include <cuda_common.cuh>

#include <Serialization.hpp>

__host__ __device__ uint64_t expandBits(uint32_t v);
__host__ __device__ uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z);

struct OctreeNode
{
	uint32_t data_or_child_idx;
	uint8_t child_mask;
};

struct Octree
{
	void Build(std::vector<float3> h_points, unsigned int MAX_OCTREE_DEPTH = 12);

	thrust::device_vector<uint64_t> d_node_codes;
	thrust::device_vector<OctreeNode> d_nodes;
};