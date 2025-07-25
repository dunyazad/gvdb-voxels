#pragma once
#include <cuda_common.cuh>

namespace VFM
{
	using Index = unsigned int;
	using Count = unsigned int;
	constexpr Index InvalidIndex = 0xFFFFFFFFu;
	constexpr Count InvalidCount = 0xFFFFFFFFu;

	struct V
	{
		Count nF = 0;
		Index* iFs = nullptr; // 1-ring face index array (device ptr)
	};

	struct F
	{
		Index iV0 = InvalidIndex, iV1 = InvalidIndex, iV2 = InvalidIndex;
		Index iF0 = InvalidIndex, iF1 = InvalidIndex, iF2 = InvalidIndex;
	};

	struct M
	{
		float3* d_positions = nullptr;
		uint3* d_faces = nullptr;

		Count nV = 0;
		V* Vs = nullptr;

		Count nF = 0;
		F* Fs = nullptr;

		Count nVFRelations = 0;     // total size of flat VFRelations
		Index* VFRelations = nullptr; // flat 1-ring face table (device ptr)

		void Initialize(Count nV, Count nF);
		void Initialize(Count nV, float3* d_positions, Count nF, uint3* d_faces);
		void Terminate();

		void BuildVFRelation();
	};

	__global__ void Kernel_CountVFRelation(const uint3* d_faces, int nF, Count* d_vFaceCount);
	__global__ void Kernel_BuildVFRelation(const uint3* d_faces, int nF, const Count* d_vfOffsets, Count* d_atomicOffsets, Index* d_VFRelations);
	__global__ void Kernel_SetVFPointer(V* Vs, const Count* d_vfOffsets, Index* d_VFRelations, const Count* d_vFaceCount, int nV);
}
