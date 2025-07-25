#include <VEFM.cuh>

namespace VEFM
{
	void M::Initialize(Count nV, Count nF)
	{
		this->nV = nV;
		this->nE = nF * 3;
		this->nF = nF;
		CUDA_MALLOC(&d_positions, nV * sizeof(float3));
		CUDA_MALLOC(&Vs, nV * sizeof(V));
		CUDA_MALLOC(&Es, nE * sizeof(E));
		CUDA_MALLOC(&d_faces, nF * sizeof(uint3));
		CUDA_MALLOC(&Fs, nF * sizeof(F));
	}

	void M::Initialize(Count nV, float3* d_positions_, Count nF, uint3* d_faces_)
	{
		this->nV = nV;
		this->nE = nF * 3;
		this->nF = nF;
		CUDA_MALLOC(&d_positions, nV * sizeof(float3));
		CUDA_COPY_D2D(d_positions, d_positions_, nV * sizeof(float3));
		CUDA_MALLOC(&Vs, nV * sizeof(V));
		CUDA_MALLOC(&Es, nE * sizeof(E));
		CUDA_MALLOC(&d_faces, nF * sizeof(uint3));
		CUDA_COPY_D2D(d_faces, d_faces_, nF * sizeof(uint3));
		CUDA_MALLOC(&Fs, nF * sizeof(F));
	}

	void M::Terminate()
	{
		if (d_positions) CUDA_FREE(d_positions);
		if (Vs) CUDA_FREE(Vs);
		if (Es) CUDA_FREE(Es);
		if (d_faces) CUDA_FREE(d_faces);
		if (Fs) CUDA_FREE(Fs);
		d_positions = nullptr;
		Vs = nullptr;
		Es = nullptr;
		d_faces = nullptr;
		Fs = nullptr;

		nV = 0;
		nE = 0;
		nF = 0;
	}
}
