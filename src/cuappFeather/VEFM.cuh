#pragma once
#include <cuda_common.cuh>

#include <HashMap.hpp>

namespace VEFM
{
	using Index = unsigned int;
	using Count = unsigned int;
	constexpr Index InvalidIndex = 0xFFFFFFFFu;
	constexpr Count InvalidCount = 0xFFFFFFFFu;

	struct V
	{
		Count nE = 0; //1이면 있을 수 없는 상태
		Index* iEs = nullptr;
	};

	struct E
	{
		Index iV0;
		Index iV1;

		Count nF = 0; //1이면 Border Edge
		Index* iFs = nullptr;
	};

	struct F
	{
		Index iV0 = InvalidIndex;
		Index iV1 = InvalidIndex;
		Index iV2 = InvalidIndex;

		Index iE0 = InvalidIndex;
		Index iE1 = InvalidIndex;
		Index iE2 = InvalidIndex;
	};

	struct M
	{
		float3* d_positions = nullptr;
		uint3* d_faces = nullptr;

		Count nV = 0;
		V* Vs = nullptr;

		Count nE = 0;
		E* Es = nullptr;

		Count nF = 0;
		F* Fs = nullptr;

		void Initialize(Count nV, Count nF);
		void Initialize(Count nV, float3* d_positions, Count nF, uint3* d_faces);
		void Terminate();
	};
}
