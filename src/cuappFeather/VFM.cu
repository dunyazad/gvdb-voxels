#include <VFM.cuh>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace VFM
{

	void M::Initialize(Count nV, Count nF)
	{
		this->nV = nV;
		this->nF = nF;
		CUDA_MALLOC(&d_positions, nV * sizeof(float3));
		CUDA_MALLOC(&Vs, nV * sizeof(V));
		CUDA_MALLOC(&d_faces, nF * sizeof(uint3));
		CUDA_MALLOC(&Fs, nF * sizeof(F));
		if (VFRelations) CUDA_FREE(VFRelations);
		VFRelations = nullptr;
		nVFRelations = 0;
	}

	void M::Initialize(Count nV, float3* d_positions_, Count nF, uint3* d_faces_)
	{
		this->nV = nV;
		this->nF = nF;
		CUDA_MALLOC(&d_positions, nV * sizeof(float3));
		CUDA_COPY_H2D(d_positions, d_positions_, nV * sizeof(float3));
		CUDA_MALLOC(&Vs, nV * sizeof(V));
		CUDA_MALLOC(&d_faces, nF * sizeof(uint3));
		CUDA_COPY_H2D(d_faces, d_faces_, nF * sizeof(uint3));
		CUDA_MALLOC(&Fs, nF * sizeof(F));
		if (VFRelations) CUDA_FREE(VFRelations);
		VFRelations = nullptr;
		nVFRelations = 0;
	}

	void M::Terminate()
	{
		if (d_positions) CUDA_FREE(d_positions);
		if (Vs) CUDA_FREE(Vs);
		if (d_faces) CUDA_FREE(d_faces);
		if (Fs) CUDA_FREE(Fs);
		if (VFRelations) CUDA_FREE(VFRelations);
		d_positions = nullptr;
		Vs = nullptr;
		d_faces = nullptr;
		Fs = nullptr;
		VFRelations = nullptr;
		nVFRelations = 0;
	}

	// ---------------- CUDA 커널 -----------------

	__global__ void Kernel_CountVFRelation(const uint3* d_faces, int nF, Count* d_vFaceCount)
	{
		int fidx = blockIdx.x * blockDim.x + threadIdx.x;
		if (fidx >= nF) return;
		uint3 f = d_faces[fidx];
		atomicAdd(&d_vFaceCount[f.x], 1);
		atomicAdd(&d_vFaceCount[f.y], 1);
		atomicAdd(&d_vFaceCount[f.z], 1);
	}

	__global__ void Kernel_BuildVFRelation(const uint3* d_faces, int nF, const Count* d_vfOffsets, Count* d_atomicOffsets, Index* d_VFRelations)
	{
		int fidx = blockIdx.x * blockDim.x + threadIdx.x;
		if (fidx >= nF) return;
		uint3 f = d_faces[fidx];
		for (int k = 0; k < 3; ++k)
		{
			Index v = (k == 0) ? f.x : (k == 1) ? f.y : f.z;
			Count pos = atomicAdd(&d_atomicOffsets[v], 1);
			d_VFRelations[d_vfOffsets[v] + pos] = fidx;
		}
	}

	__global__ void Kernel_SetVFPointer(V* Vs, const Count* d_vfOffsets, Index* d_VFRelations, const Count* d_vFaceCount, int nV)
	{
		int v = blockIdx.x * blockDim.x + threadIdx.x;
		if (v >= nV) return;
		Vs[v].nF = d_vFaceCount[v];
		Vs[v].iFs = (d_vFaceCount[v] ? d_VFRelations + d_vfOffsets[v] : nullptr);
	}

	// ---------------- host-side build function (풀 CUDA) -----------------

	void M::BuildVFRelation()
	{
		// 1. device-side 카운트 버퍼 할당/초기화
		thrust::device_vector<Count> d_vFaceCount(nV, 0);

		// 2. 각 vertex별 포함 face 수 세기 (atomic)
		dim3 block(128);
		dim3 grid((nF + block.x - 1) / block.x);
		LaunchKernel(Kernel_CountVFRelation, nF,
			d_faces, nF, thrust::raw_pointer_cast(d_vFaceCount.data()));
		CUDA_SYNC();

		// 3. prefix sum (vertex별 offset)
		thrust::device_vector<Count> d_vfOffsets(nV, 0);
		thrust::exclusive_scan(d_vFaceCount.begin(), d_vFaceCount.end(), d_vfOffsets.begin());

		// 4. 전체 개수 nVFRelations 계산 (마지막 오프셋 + 마지막 count)
		Count lastOffset, lastCount;
		CUDA_COPY_D2H(&lastOffset, thrust::raw_pointer_cast(d_vfOffsets.data()) + (nV - 1), sizeof(Count));
		CUDA_COPY_D2H(&lastCount, thrust::raw_pointer_cast(d_vFaceCount.data()) + (nV - 1), sizeof(Count));
		nVFRelations = lastOffset + lastCount;

		// 5. VFRelations, atomic offset 할당
		if (VFRelations) CUDA_FREE(VFRelations);
		CUDA_MALLOC(&VFRelations, nVFRelations * sizeof(Index));
		Count* d_atomicOffsets = nullptr;
		CUDA_MALLOC(&d_atomicOffsets, nV * sizeof(Count));
		CUDA_MEMSET(d_atomicOffsets, 0, nV * sizeof(Count));

		// 6. flat VFRelations 생성
		LaunchKernel(Kernel_BuildVFRelation, nF,
			d_faces, nF, thrust::raw_pointer_cast(d_vfOffsets.data()), d_atomicOffsets, VFRelations);
		CUDA_SYNC();

		// 7. Vs[v].nF, Vs[v].iFs 포인터 세팅 (커널)
		LaunchKernel(Kernel_SetVFPointer, nV,
			Vs, thrust::raw_pointer_cast(d_vfOffsets.data()), VFRelations,
			thrust::raw_pointer_cast(d_vFaceCount.data()), nV);
		CUDA_SYNC();

		// 8. 메모리 해제
		CUDA_FREE(d_atomicOffsets);
	}
}
