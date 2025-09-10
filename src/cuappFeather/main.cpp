#pragma warning(disable : 4819)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <libFeather.h>

#include "main.cuh"

#include <iostream>
using namespace std;

using VD = VisualDebugging;

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

typedef enum {
	DL_TOOTH,
	DL_GINGIVA1,
	DL_GINGIVA2,
	DL_TONGUE,
	DL_CHEEK,
	DL_LIP,
	DL_ETC,
	DL_DENTIFORM_TOOTH,
	DL_DENTIFORM_GINGIVA1,
	DL_DENTIFORM_GINGIVA2,
	DL_PLASTER,
	DL_FINGER,
	DL_METAL,
	DL_PALATAL,
	DL_ABUTMENT,
	DL_SCANBODY,
	DL_GINGIVA3,
	DL_OBTURA,
	DL_3DPRTMODEL,
	DL_RETRACTOR,
	DL_CLASS_LAST
} DL_Class_Names;

bool IsTooth(int deepLearningClass)
{
	switch (deepLearningClass)
	{
case DL_TOOTH:
	return true;
case DL_GINGIVA1:
	return false;
case DL_GINGIVA2:
	return false;
case DL_TONGUE:
	return false;
case DL_CHEEK:
	return false;
case DL_LIP:
	return false;
case DL_ETC:
	return false;
case DL_DENTIFORM_TOOTH:
	return true;
case DL_DENTIFORM_GINGIVA1:
	return false;
case DL_DENTIFORM_GINGIVA2:
	return false;
case DL_PLASTER:
	return false;
case DL_FINGER:
	return false;
case DL_METAL:
	return true;
case DL_PALATAL:
	return false;
case DL_ABUTMENT:
	return true;
case DL_SCANBODY:
	return false;
case DL_GINGIVA3:
	return false;
case DL_OBTURA:
	return false;
case DL_3DPRTMODEL:
	return false;
case DL_RETRACTOR:
	return false;
case DL_CLASS_LAST:
	return false;

	default:
		return false;
		break;
	}
}

float3 calculateVertexNormal(unsigned int v_idx, const HostHalfEdgeMesh<PointCloudProperty>& h_mesh)
{
	float3 normal_sum = make_float3(0.0f, 0.0f, 0.0f);

	unsigned int start_he_idx = h_mesh.vertexToHalfEdge[v_idx];
	if (start_he_idx == UINT32_MAX) return normal_sum; // 고립된 정점

	unsigned int current_he_idx = start_he_idx;
	do
	{
		const auto& he = h_mesh.halfEdges[current_he_idx];
		const auto& next_he = h_mesh.halfEdges[he.nextIndex];
		const auto& prev_he = h_mesh.halfEdges[next_he.nextIndex];

		// 면 법선 계산 (가중치 없이 단순 합산)
		const auto& p0 = h_mesh.positions[he.vertexIndex];
		const auto& p1 = h_mesh.positions[next_he.vertexIndex];
		const auto& p2 = h_mesh.positions[prev_he.vertexIndex];
		normal_sum += cross(p1 - p0, p2 - p0);

		// 다음 면으로 이동
		if (he.oppositeIndex == UINT32_MAX) break; // 경계에 도달하면 한쪽 방향 순회 중단

		current_he_idx = h_mesh.halfEdges[he.oppositeIndex].nextIndex;

	} while (current_he_idx != start_he_idx);

	return normalize(normal_sum);
}

float3 CatmullRom(const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t) {
	float t2 = t * t;
	float t3 = t2 * t;

	float3 a, b, c, d;

	// 계수 계산
	a.x = -0.5f * p0.x + 1.5f * p1.x - 1.5f * p2.x + 0.5f * p3.x;
	a.y = -0.5f * p0.y + 1.5f * p1.y - 1.5f * p2.y + 0.5f * p3.y;
	a.z = -0.5f * p0.z + 1.5f * p1.z - 1.5f * p2.z + 0.5f * p3.z;

	b.x = p0.x - 2.5f * p1.x + 2.0f * p2.x - 0.5f * p3.x;
	b.y = p0.y - 2.5f * p1.y + 2.0f * p2.y - 0.5f * p3.y;
	b.z = p0.z - 2.5f * p1.z + 2.0f * p2.z - 0.5f * p3.z;

	c.x = -0.5f * p0.x + 0.5f * p2.x;
	c.y = -0.5f * p0.y + 0.5f * p2.y;
	c.z = -0.5f * p0.z + 0.5f * p2.z;

	d.x = p1.x;
	d.y = p1.y;
	d.z = p1.z;

	// 보간된 점 계산
	float3 result;
	result.x = a.x * t3 + b.x * t2 + c.x * t + d.x;
	result.y = a.y * t3 + b.y * t2 + c.y * t + d.y;
	result.z = a.z * t3 + b.z * t2 + c.z * t + d.z;

	return result;
}

std::vector<float3> GenerateCatmullRomSpline(const std::vector<float3>& controlPoints, int pointsPerSegment) {
	std::vector<float3> splinePoints;
	if (controlPoints.size() < 4) {
		return splinePoints; // 최소 4개의 제어점이 필요
	}

	for (size_t i = 0; i < controlPoints.size() - 3; ++i) {
		const float3& p0 = controlPoints[i];
		const float3& p1 = controlPoints[i + 1];
		const float3& p2 = controlPoints[i + 2];
		const float3& p3 = controlPoints[i + 3];

		for (int j = 0; j < pointsPerSegment; ++j) {
			float t = (float)j / (float)pointsPerSegment;
			splinePoints.push_back(CatmullRom(p0, p1, p2, p3, t));
		}
	}

	// 마지막 구간의 끝점을 명시적으로 추가하여 P(n-2)에서 P(n-1)까지 완전히 그리도록 함
	splinePoints.push_back(controlPoints[controlPoints.size() - 2]);


	return splinePoints;
}

void smoothOpenPolyline_MovingAverage(std::vector<float3>& points, int iterations) {
	if (points.size() < 3) return; // 점이 3개 미만이면 적용 불가

	for (int iter = 0; iter < iterations; ++iter) {
		std::vector<float3> smoothedPoints = points; // 복사본 생성

		// 시작점과 끝점을 제외한 내부 점들만 처리 (1부터 n-2까지)
		for (size_t i = 1; i < points.size() - 1; ++i) {
			const float3& prev = points[i - 1];
			const float3& curr = points[i];
			const float3& next = points[i + 1];

			smoothedPoints[i].x = (prev.x + curr.x + next.x) / 3.0f;
			smoothedPoints[i].y = (prev.y + curr.y + next.y) / 3.0f;
			smoothedPoints[i].z = (prev.z + curr.z + next.z) / 3.0f;
		}
		points = smoothedPoints; // 결과로 교체
	}
}

void smoothOpenPolyline_MovingAverage_Hard(std::vector<float3>& points, int iterations) {
	if (points.size() < 3) return; // 점이 3개 미만이면 적용 불가

	for (int iter = 0; iter < iterations; ++iter) {
		for (size_t i = 2; i < points.size() - 2; ++i) {
			const float3& pprev = points[i - 2];
			const float3& prev = points[i - 1];
			const float3& curr = points[i];
			const float3& next = points[i + 1];
			const float3& nnext = points[i + 2];

			points[i].x = (pprev.x + prev.x + next.x + nnext.x) / 4.0f;
			points[i].y = (pprev.y + prev.y + next.y + nnext.y) / 4.0f;
			points[i].z = (pprev.z + prev.z + next.z + nnext.z) / 4.0f;
		}
	}
}


void smoothOpenPolyline_ParameterizedMovingAverage(std::vector<float3>& points, int iterations, int windowSize) {
	// windowSize는 홀수여야 하고, 3 이상이어야 의미가 있습니다.
	if (windowSize < 3 || windowSize % 2 == 0) {
		// 오류를 출력하거나, 아무 작업도 하지 않고 반환할 수 있습니다.
		// 예를 들어, windowSize가 4이면 대칭이 아니므로 중심점을 정의하기 어렵습니다.
		return;
	}

	// 윈도우 크기가 전체 포인트 수보다 크거나 같으면 스무딩이 불가능합니다.
	if (points.size() <= windowSize) {
		return;
	}

	// 중심점으로부터 양쪽으로 몇 개의 점을 볼 것인지를 결정 (예: windowSize=5 -> radius=2)
	int radius = windowSize / 2;

	for (int iter = 0; iter < iterations; ++iter) {
		std::vector<float3> smoothedPoints = points; // 복사본 생성

		// 양 끝의 'radius' 개수만큼의 점들을 제외한 내부 점들만 처리
		// 예: radius=2 이면, 2번 인덱스부터 points.size() - 3 인덱스까지 순회
		for (size_t i = radius; i < points.size() - radius; ++i) {

			// 윈도우 내의 모든 점들의 합을 계산
			float3 sum = { 0.0f, 0.0f, 0.0f };
			for (int j = -radius; j <= radius; ++j) {
				const float3& p = points[i + j];
				sum.x += p.x;
				sum.y += p.y;
				sum.z += p.z;
			}

			// 합계를 윈도우 크기로 나누어 평균을 계산
			smoothedPoints[i].x = sum.x / windowSize;
			smoothedPoints[i].y = sum.y / windowSize;
			smoothedPoints[i].z = sum.z / windowSize;
		}
		points = smoothedPoints; // 결과로 교체
	}
}

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001

MarginLineFinder marginLineFinder;
vector<float3> marginLinePoints;
vector<int> deepLearningClasses;
std::vector<std::vector<float3>> borderRingPoints;

CUDAInstance cuInstance;

bool ForceGPUPerformance()
{
	NvAPI_Status status;

	status = NvAPI_Initialize();
	if (status != NVAPI_OK)
	{
		return false;
	}

	NvDRSSessionHandle hSession = 0;
	status = NvAPI_DRS_CreateSession(&hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	// (2) load all the system settings into the session
	status = NvAPI_DRS_LoadSettings(hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	NvDRSProfileHandle hProfile = 0;
	status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
	if (status != NVAPI_OK)
	{
		return false;
	}

	NVDRS_SETTING drsGet = { 0, };
	drsGet.version = NVDRS_SETTING_VER;
	status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
	if (status != NVAPI_OK)
	{
		return false;
	}
	auto m_gpu_performance = drsGet.u32CurrentValue;

	NVDRS_SETTING drsSetting = { 0, };
	drsSetting.version = NVDRS_SETTING_VER;
	drsSetting.settingId = PREFERRED_PSTATE_ID;
	drsSetting.settingType = NVDRS_DWORD_TYPE;
	drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;

	status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
	if (status != NVAPI_OK)
	{
		return false;
	}

	status = NvAPI_DRS_SaveSettings(hSession);
	if (status != NVAPI_OK)
	{
		return false;
	}

	// (6) We clean up. This is analogous to doing a free()
	NvAPI_DRS_DestroySession(hSession);
	hSession = 0;

	return true;
}

int main(int argc, char** argv)
{
	ForceGPUPerformance();

	cout << "AppFeather" << endl;

	Feather.Initialize(1920, 1080);
	Feather.SetClearColor(Color::slategray());

	auto w = Feather.GetFeatherWindow();

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&](Entity entity, const KeyEvent& event) {
			//printf("KeyEvent : keyCode=%d, scanCode=%d, action=%d, mods=%d\n", event.keyCode, event.scanCode, event.action, event.mods);

			if(GLFW_KEY_GRAVE_ACCENT == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_0");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							auto current = renderable->GetActiveShaderIndex();
							auto next = (current + 1) % renderable->GetShaders().size();
							renderable->SetActiveShaderIndex(next);
							printf("Shader index : %d\n", next);
						}
					}
				}
			}
			else if (GLFW_KEY_1 == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_0");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							renderable->ToggleVisible();
						}
					}
				}
			}
			else if (GLFW_KEY_2 == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("Compound_Class_1");
					if (entity != InvalidEntity)
					{
						auto renderable = Feather.GetComponent<Renderable>(entity);
						if (renderable)
						{
							renderable->ToggleVisible();
						}
					}
				}
			}
			else if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			else if (GLFW_KEY_BACKSPACE == event.keyCode)
			{
				VD::ClearAll();
			}
			else if (GLFW_KEY_F1 == event.keyCode)
			{
				
			}
			else if (GLFW_KEY_TAB == event.keyCode)
			{
				
			}
			else if (GLFW_KEY_SPACE == event.keyCode)
			{
				if (event.action == 1)
				{
				}
			}
			else if (GLFW_KEY_ENTER == event.keyCode)
			{
				if (event.action == 1)
				{
					vector<float3> result;
					marginLineFinder.FindMarginLinePoints(result);

					for (size_t i = 0; i < result.size(); i++)
					{
						auto& p = result[i];

						VD::AddWiredBox("MarginLine", glm::vec3(XYZ(p)), glm::vec3(0.1f, 0.1f, 0.1f), Color::red());
					}
				}
			}
			else if (GLFW_KEY_INSERT == event.keyCode)
			{
				if (event.action == 1)
				{
				}
			}
			else if (GLFW_KEY_HOME == event.keyCode)
			{
				if (event.action == 1)
				{
				}
			}
			else if (GLFW_KEY_END == event.keyCode)
			{
			}
			else if (GLFW_KEY_UP == event.keyCode)
			{
				if (event.action == 1)
				{
					auto index = VD::ShowNextSelection();
					printf("index : %d\n", index);
				}
			}
			else if (GLFW_KEY_DOWN == event.keyCode)
			{
				if (event.action == 1)
				{
					auto index = VD::ShowPreviousSelection();
					printf("index : %d\n", index);
				}
			}
			else if (GLFW_KEY_BACKSLASH == event.keyCode)
			{
				if (event.action == 1)
				{

				}
			}
			else if (GLFW_KEY_PAGE_UP == event.keyCode)
			{
				if (event.action == 1)
				{
				}
			}
			else if (GLFW_KEY_PAGE_DOWN == event.keyCode)
			{
			}
			});
	}
#pragma endregion

#pragma region Camera
	{
		Entity cam = Feather.CreateEntity("Camera");
		auto pcam = Feather.CreateComponent<PerspectiveCamera>(cam);
		auto pcamMan = Feather.CreateComponent<CameraManipulatorTrackball>(cam);
		pcamMan->SetCamera(pcam);

		Feather.CreateEventCallback<FrameBufferResizeEvent>(cam, [pcam](Entity entity, const FrameBufferResizeEvent& event) {
			auto window = Feather.GetFeatherWindow();
			auto aspectRatio = (f32)window->GetWidth() / (f32)window->GetHeight();
			pcam->SetAspectRatio(aspectRatio);
			});

		Feather.CreateEventCallback<KeyEvent>(cam, [](Entity entity, const KeyEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnKey(event);
			});

		Feather.CreateEventCallback<MousePositionEvent>(cam, [](Entity entity, const MousePositionEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMousePosition(event);
			});

		Feather.CreateEventCallback<MouseButtonEvent>(cam, [&](Entity entity, const MouseButtonEvent& event) {
			auto manipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);
			manipulator->OnMouseButton(event);
			auto camera = manipulator->GetCamera();

			auto renderable = Feather.GetComponent<Renderable>(entity);
			if (event.button == 0 && event.action == 0)
			{
			}
			});

		Feather.CreateEventCallback<MouseWheelEvent>(cam, [](Entity entity, const MouseWheelEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMouseWheel(event);
			});
	}
#pragma endregion

#pragma region Panels
	{
		auto gui = Feather.CreateEntity("Status Panel");

		auto statusPanel = Feather.CreateComponent<StatusPanel>(gui);
		Feather.CreateEventCallback<MousePositionEvent>(gui, [](Entity entity, const MousePositionEvent& event) {
			auto component = Feather.GetComponent<StatusPanel>(entity);
			component->mouseX = event.xpos;
			component->mouseY = event.ypos;
			});
	}

	{
		auto entity = Feather.CreateEntity("Control Panel");
		auto controlPanel = Feather.CreateComponent<ControlPanel>(entity, "Control Panel");

		auto GetNearestPoints = [&]() {
			cuInstance.d_mesh.UpdateBVH();

			//vector<float3> inputPositions(cuInstance.h_mesh.numberOfPoints);
			vector<float3> inputPositions(cuInstance.h_input.numberOfPoints);
			for (size_t i = 0; i < cuInstance.h_input.numberOfPoints; i++)
			{
				inputPositions[i] = cuInstance.h_input.positions[i];
			}

			TS(NN);
			vector<float3> resultPositions;
			vector<int> resultTriangleIndices;
			cuInstance.d_mesh.GetNearestPoints(inputPositions, resultPositions, resultTriangleIndices);
			TE(NN);

			for (size_t i = 0; i < resultPositions.size(); i++)
			{
				if (IsTooth(deepLearningClasses[i]))
				{
					auto& q = inputPositions[i];
					auto& p = resultPositions[i];
					auto& iii = cuInstance.h_mesh.faces[resultTriangleIndices[i]];
					auto& v0 = cuInstance.h_mesh.positions[iii.x];
					auto& v1 = cuInstance.h_mesh.positions[iii.y];
					auto& v2 = cuInstance.h_mesh.positions[iii.z];

					//VD::AddLine("NN GetNearestPoints", { XYZ(q) }, { XYZ(p) }, Color::yellow());
					//VD::AddWiredBox("NN GetNearestPoints - input", { XYZ(q) }, glm::vec3(0.01f, 0.01f, 0.01f), Color::red());
					//VD::AddWiredBox("NN GetNearestPoints - result", { XYZ(p) }, glm::vec3(0.01f, 0.01f, 0.01f), Color::blue());

					VD::AddLine("Triangles", { XYZ(v0) }, { XYZ(v1) }, Color::green());
					VD::AddLine("Triangles", { XYZ(v1) }, { XYZ(v2) }, Color::green());
					VD::AddLine("Triangles", { XYZ(v2) }, { XYZ(v0) }, Color::green());
				}
			}

			auto meshEntity = Feather.GetEntityByName("MarchingCubesMesh");
			auto meshRenderable = Feather.GetComponent<Renderable>(meshEntity);
			cuInstance.interop.Initialize(meshRenderable);
			cuInstance.interop.UploadFromDevice(cuInstance.d_mesh);
			};

		auto ShowVertexDLClasses = [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;
			for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
			{
				auto& p = h_mesh.positions[i];
				auto& cl = h_mesh.properties[i].deepLearningClass;

				if (0 != cl && 1 != cl)
					VD::AddText("DeepLearningClass", std::to_string(cl), { XYZ(p) });
			}
			};

		auto BorderLines = [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;
			for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
			{
				auto& he = h_mesh.halfEdges[i];
				if (UINT32_MAX == he.oppositeIndex) continue;

				auto& oe = h_mesh.halfEdges[he.oppositeIndex];

				auto& dlc = h_mesh.properties[he.vertexIndex].deepLearningClass;
				auto& odlc = h_mesh.properties[oe.vertexIndex].deepLearningClass;

				if (dlc == 0 && odlc == 1)
				{
					auto& v = h_mesh.positions[he.vertexIndex];
					auto& ov = h_mesh.positions[oe.vertexIndex];
					VD::AddLine("Border Line", { XYZ(v) }, { XYZ(ov) }, Color::red(), Color::red());
				}
			}
			};

		auto DLClasses = [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;

			map<int, string> names;
			for (size_t i = 0; i < h_mesh.numberOfFaces; i++)
			{
				auto& f = h_mesh.faces[i];
				auto& v0 = h_mesh.positions[f.x];
				auto& v1 = h_mesh.positions[f.y];
				auto& v2 = h_mesh.positions[f.z];
				auto& property0 = h_mesh.properties[f.x];
				auto& property1 = h_mesh.properties[f.y];
				auto& property2 = h_mesh.properties[f.z];

				auto dc0 = property0.deepLearningClass;
				auto dc1 = property1.deepLearningClass;
				auto dc2 = property2.deepLearningClass;

				glm::vec4 c0 = dc0 != 1 ? Color::red() : Color::blue();
				glm::vec4 c1 = dc1 != 1 ? Color::red() : Color::blue();
				glm::vec4 c2 = dc2 != 1 ? Color::red() : Color::blue();

				//if((0 == dc0 || 1 == dc0) || (0 == dc1 || 1 == dc1) || (0 == dc2 || 1 == dc2))
				//if (IsTooth(dc0) || IsTooth(dc2) || IsTooth(dc2))
				string name = "DLClassses" + to_string(dc0);
				names[dc0] = name;
				VD::AddTriangle(name, { XYZ(v0) }, { XYZ(v1) }, { XYZ(v2) }, c0, c1, c2);
			}

			for (auto& kvp : names)
			{
				VD::AddToSelectionList(kvp.second);
			}
			};

		auto MarginLinesNonManifold = [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;
			if (h_mesh.numberOfFaces == 0) return;

			// 1. 데이터 구조 : 한 시작 정점에 여러 경계 에지가 연결될 수 있도록 vector 사용
			std::map<unsigned int, std::vector<unsigned int>> borderEdgeStartMap;
			std::set<unsigned int> borderHalfEdgeIndices;

			for (unsigned int i = 0; i < h_mesh.numberOfFaces * 3; ++i)
			{
				const auto& he = h_mesh.halfEdges[i];
				if (he.oppositeIndex == UINT32_MAX)
				{
					borderHalfEdgeIndices.insert(i);
					borderEdgeStartMap[he.vertexIndex].push_back(i);
				}
			}
			printf("Found %zu border half-edges.\n", borderHalfEdgeIndices.size());

			std::vector<std::vector<unsigned int>> borderRings;

			// 2. 모든 경계 에지를 처리할 때까지
			while (!borderHalfEdgeIndices.empty())
			{
				std::vector<unsigned int> currentRing;
				std::vector<float3> currentRingPoints;
				unsigned int startHeIdx = *borderHalfEdgeIndices.begin();
				unsigned int currentHeIdx = startHeIdx;
				unsigned int prevHeIdx = UINT32_MAX; // 이전 에지를 추적

				do
				{
					currentRing.push_back(currentHeIdx);
					borderHalfEdgeIndices.erase(currentHeIdx);

					const auto& currentHe = h_mesh.halfEdges[currentHeIdx];
					const auto& nextHeInFace = h_mesh.halfEdges[currentHe.nextIndex];
					unsigned int endVertex = nextHeInFace.vertexIndex;

					currentRingPoints.push_back(h_mesh.positions[currentHe.vertexIndex]);

					// 다음 경계 에지 후보들을 찾음
					auto it = borderEdgeStartMap.find(endVertex);
					if (it == borderEdgeStartMap.end() || it->second.empty())
					{
						// 경로의 끝 (닫히지 않은 루프)
						break;
					}

					auto& candidates = it->second;
					unsigned int nextHeIdx = UINT32_MAX;

					if (candidates.size() == 1)
					{
						nextHeIdx = candidates[0];
					}
					else // 3. Non-manifold junction 처리
					{
						// 여러 후보 중 가장 "자연스러운" 다음 에지를 선택
						float3 v_normal = calculateVertexNormal(endVertex, h_mesh);
						float3 pos_prev = h_mesh.positions[currentHe.vertexIndex];
						float3 pos_curr = h_mesh.positions[endVertex];
						float3 vec_in = normalize(pos_curr - pos_prev);

						float best_angle = -FLT_MAX; // 가장 큰 양수 각도(가장 왼쪽으로 꺾이는 경로)를 찾음

						for (unsigned int candidate_idx : candidates)
						{
							// 자기 자신으로 돌아가는 경로는 제외
							if (candidate_idx == currentHeIdx) continue;

							const auto& candidate_he = h_mesh.halfEdges[candidate_idx];
							const auto& candidate_next_he = h_mesh.halfEdges[candidate_he.nextIndex];
							float3 pos_next = h_mesh.positions[candidate_next_he.vertexIndex];
							float3 vec_out = normalize(pos_next - pos_curr);

							// 법선(normal)을 축으로 두 벡터 사이의 부호 있는 각도 계산
							float dot_val = dot(vec_in, vec_out);
							float3 cross_val = cross(vec_in, vec_out);
							float angle = atan2(dot(cross_val, v_normal), dot_val);

							if (angle > best_angle)
							{
								best_angle = angle;
								nextHeIdx = candidate_idx;
							}
						}
					}

					prevHeIdx = currentHeIdx;
					currentHeIdx = nextHeIdx;

					if (currentHeIdx == UINT32_MAX || borderHalfEdgeIndices.find(currentHeIdx) == borderHalfEdgeIndices.end())
					{
						// 유효한 다음 경로가 없거나 이미 처리된 경로이면 중단
						break;
					}

				} while (currentHeIdx != startHeIdx);
				currentRing.push_back(startHeIdx);
				currentRingPoints.push_back(h_mesh.positions[h_mesh.halfEdges[startHeIdx].vertexIndex]);

				borderRings.push_back(currentRing);
				borderRingPoints.push_back(currentRingPoints);
			}

			// 4. Visualize the found border loops.
			printf("Assembled into %zu border ring(s).\n", borderRings.size());

			std::vector<glm::vec4> colors = {
				Color::red(),
				Color::green(),
				Color::blue(),
				Color::yellow(),
				Color::magenta(),
				Color::cyan()
			};

			for (size_t i = 0; i < borderRings.size(); ++i)
			{
				const auto& ring = borderRings[i];
				const auto& color = colors[i % colors.size()];
				std::string lineName = "BorderLine_" + std::to_string(i);

				printf("Ring %zu has %zu edges.\n", i, ring.size());

				if ((ring.size() > 300) && (ring.back() == ring.front()))
				{
					for (unsigned int heIdx : ring)
					{
						const auto& he = h_mesh.halfEdges[heIdx];
						const auto& nextHe = h_mesh.halfEdges[he.nextIndex];

						const auto& v0 = h_mesh.positions[he.vertexIndex];
						const auto& v1 = h_mesh.positions[nextHe.vertexIndex];

						VD::AddLine(lineName, { XYZ(v0) }, { XYZ(v1) }, color, color);
					}
				}
			}
			};

		auto CatmulRomSpline = [&]() {
			std::vector<glm::vec4> colors = {
				Color::red(),
				Color::green(),
				Color::blue(),
				Color::yellow(),
				Color::magenta(),
				Color::cyan()
			};

			for (size_t i = 0; i < borderRingPoints.size(); i++)
			{
				auto& points = borderRingPoints[i];

				if ((points.size() > 300) && (points.back() == points.front()))
				{
					auto result = GenerateCatmullRomSpline(points, 10);

					const auto& color = colors[i % colors.size()];

					for (size_t j = 0; j < result.size() - 1; j++)
					{
						auto& p0 = result[j];
						auto& p1 = result[(j + 1) % result.size()];

						VD::AddLine("MarginLine", { XYZ(p0) }, { XYZ(p1) }, color, color);
					}
				}
			}
			};

		auto UpdateBVH = [&]() {
			cuInstance.d_mesh.UpdateBVH();
			};

		auto Smoothing = [&]() {
			TS(Smoothing);

			//for (size_t iteration = 0; iteration < 10; iteration++)
			{
				for (size_t i = 0; i < borderRingPoints.size(); i++)
				{
					auto& points = borderRingPoints[i];

					if ((points.size() > 300) && (points.back() == points.front()))
					{
						smoothOpenPolyline_MovingAverage_Hard(points, 100);
						//smoothOpenPolyline_ParameterizedMovingAverage(points, 7, 100);

						vector<float3> resultPositions;
						vector<int> resultTriangleIndices;
						cuInstance.d_mesh.GetNearestPoints(points, resultPositions, resultTriangleIndices);

						for (size_t i = 0; i < resultPositions.size(); i++)
						{
							auto& q = points[i];
							auto& p = resultPositions[i];
							q = p;
							
							//VD::AddLine("NN GetNearestPoints", { XYZ(q) }, { XYZ(p) }, Color::yellow());
							//VD::AddWiredBox("NN GetNearestPoints - input", { XYZ(q) }, glm::vec3(0.01f, 0.01f, 0.01f), Color::red());
							//VD::AddWiredBox("NN GetNearestPoints - result", { XYZ(p) }, glm::vec3(0.01f, 0.01f, 0.01f), Color::blue());

							//VD::AddLine("Triangles", { XYZ(v0) }, { XYZ(v1) }, Color::green());
							//VD::AddLine("Triangles", { XYZ(v1) }, { XYZ(v2) }, Color::green());
							//VD::AddLine("Triangles", { XYZ(v2) }, { XYZ(v0) }, Color::green());
						}
					}
				}
			}
			TE(Smoothing);
			};

		auto ShowMarginLines = [&]() {
			std::vector<glm::vec4> colors = {
				Color::red(),
				Color::green(),
				Color::blue(),
				Color::yellow(),
				Color::magenta(),
				Color::cyan()
			};

			for (size_t i = 0; i < borderRingPoints.size(); i++)
			{
				auto& points = borderRingPoints[i];
				const auto& color = colors[i % colors.size()];

				//if ((points.size() > 300) && (points.back() == points.front()))
				if (points.size() > 300)
				{
					for (size_t j = 0; j < points.size() - 1; j++)
					{
						auto& p0 = points[j];
						auto& p1 = points[(j + 1) % points.size()];

						VD::AddLine("MarginLine", { XYZ(p0) }, { XYZ(p1) }, color, color);
					}
				}
			}
			};

		controlPanel->AddButton("GetNearestPoints", 0, 0, GetNearestPoints);
		controlPanel->AddButton("Show Vertex DLClasses", 0, 0, ShowVertexDLClasses);
		controlPanel->AddButton("Border Lines", 0, 0, BorderLines);
		controlPanel->AddButton("DLClasses", 0, 0, DLClasses);
		controlPanel->AddButton("CatmulRom-Spline", 0, 0, CatmulRomSpline);
		controlPanel->AddButton("Margin Lines (Non-manifold)", 0, 0, MarginLinesNonManifold);
		controlPanel->AddButton("UpdateBVH", 0, 0, UpdateBVH);
		controlPanel->AddButton("Smoothing", 0, 0, Smoothing);
		controlPanel->AddButton("Show MarginLines", 0, 0, ShowMarginLines);
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		string modelPathRoot = "D:\\Debug\\PLY\\";
		string compound = modelPathRoot + "Compound.ply";

		HostPointCloud<PointCloudProperty> h_pointCloud;

		PLYFormat ply;
		ply.Deserialize(compound);

		auto [cx, cy, cz] = ply.GetAABBCenter();
		auto [mx, my, mz] = ply.GetAABBMin();
		auto [Mx, My, Mz] = ply.GetAABBMax();
		f32 lx = Mx - mx;
		f32 ly = My - my;
		f32 lz = Mz - mz;

		auto aabbMin = make_float3(mx, my, mz);
		auto aabbMax = make_float3(Mx, My, Mz);

		h_pointCloud.Initialize(ply.GetPoints().size() / 3);

		//auto entity = Feather.CreateEntity("Compound");
		//auto renderable = Feather.CreateComponent<Renderable>(entity);
		//renderable->Initialize(Renderable::Points);
		//renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		deepLearningClasses.clear();

		for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
		{
			auto deepLearningClass = ply.GetDeepLearningClasses()[i];
			deepLearningClasses.push_back(deepLearningClass);

			//if ((false == IsTooth(deepLearningClass)) && 1 != deepLearningClass)
			if (false == IsTooth(deepLearningClass))
			{
				continue;
			}

			auto x = ply.GetPoints()[3 * i + 0];
			auto y = ply.GetPoints()[3 * i + 1];
			auto z = ply.GetPoints()[3 * i + 2];

			auto nx = ply.GetNormals()[3 * i + 0];
			auto ny = ply.GetNormals()[3 * i + 1];
			auto nz = ply.GetNormals()[3 * i + 2];

			auto r = ply.GetColors()[3 * i + 0];
			auto g = ply.GetColors()[3 * i + 1];
			auto b = ply.GetColors()[3 * i + 2];

			glm::vec3 position = { x, y, z };
			glm::vec3 normal = { nx, ny, nz };
			glm::vec4 color = { r, g, b, 1.0f };

			//renderable->AddVertex(position);
			//renderable->AddNormal(normal);
			//renderable->AddColor(color);

			h_pointCloud.positions[i] = make_float3(x, y, z);
			h_pointCloud.normals[i] = make_float3(nx, ny, nz);
			h_pointCloud.colors[i] = make_float3(r, g, b);
			h_pointCloud.properties[i] = {deepLearningClass};
		}

		cuInstance.ProcessPointCloud(h_pointCloud, 0.05f, 3);
		{
			auto entity = Feather.CreateEntity("PointCloud");

			auto renderable = Feather.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
			renderable->AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
			renderable->SetActiveShaderIndex(1);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere({ 0.0f, 0.0f, 0.0f }, 0.05f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable->AddIndices(indices);
			renderable->AddVertices(vertices);
			renderable->AddNormals(normals);
			renderable->AddColors(colors);
			renderable->AddUVs(uvs);

			for (size_t i = 0; i < h_pointCloud.numberOfPoints; i++)
			{
				auto& p = h_pointCloud.positions[i];
				if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;
				auto& n = h_pointCloud.normals[i];
				auto& c = h_pointCloud.colors[i];

				glm::vec3 position(XYZ(p));
				glm::vec3 normal(XYZ(n));
				glm::vec4 color(XYZ(c), 1.0f);

				renderable->AddInstanceColor(color);
				renderable->AddInstanceNormal(normal);

				glm::mat4 tm = glm::identity<glm::mat4>();
				glm::mat4 rot = glm::mat4(1.0f);
				if (glm::length(normal) > 0.0001f)
				{
					glm::vec3 axis = glm::normalize(glm::cross(glm::vec3(0, 0, 1), normal));
					float angle = acos(glm::dot(glm::normalize(normal), glm::vec3(0, 0, 1)));
					if (glm::length(axis) > 0.0001f)
						rot = glm::rotate(glm::mat4(1.0f), angle, axis);
				}
				tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.125f));
				renderable->AddInstanceTransform(tm);
				renderable->IncreaseNumberOfInstances();
			}
		}
		{
			auto meshEntity = Feather.CreateEntity("MarchingCubesMesh");
			//ApplyHalfEdgeMeshToEntity(meshEntity, cudaInstance.h_mesh);
			auto meshRenderable = Feather.CreateComponent<Renderable>(meshEntity);
			meshRenderable->Initialize(Renderable::GeometryMode::Triangles);
			meshRenderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
			meshRenderable->AddShader(Feather.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
			meshRenderable->AddShader(Feather.CreateShader("Flat", File("../../res/Shaders/Flat.vs"), File("../../res/Shaders/Flat.gs"), File("../../res/Shaders/Flat.fs")));
			meshRenderable->SetActiveShaderIndex(0);

			cuInstance.interop.Initialize(meshRenderable);
			cuInstance.interop.UploadFromDevice(cuInstance.d_mesh);

			Feather.CreateEventCallback<KeyEvent>(meshEntity, [cx, cy, cz, lx, ly, lz](Entity entity, const KeyEvent& event) {
				auto renderable = Feather.GetComponent<Renderable>(entity);
				if (nullptr == renderable) return;

				if (0 == event.action)
				{
					if (GLFW_KEY_GRAVE_ACCENT == event.keyCode)
					{
						renderable->NextDrawingMode();
					}
					else if (GLFW_KEY_1 == event.keyCode)
					{
						renderable->SetActiveShaderIndex(0);
					}
					else if (GLFW_KEY_2 == event.keyCode)
					{
						renderable->SetActiveShaderIndex(1);
					}
				}
				});
		}

		//cuInstance.d_mesh.LaplacianSmoothing(10);

		{
			float3 new_aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			float3 new_aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			vector<float3> inputPoints(cuInstance.h_mesh.numberOfFaces);
			
			for (size_t i = 0; i < cuInstance.h_mesh.numberOfFaces; i++)
			{
				auto& f = cuInstance.h_mesh.faces[i];
				auto& v0 = cuInstance.h_mesh.positions[f.x];
				auto& v1 = cuInstance.h_mesh.positions[f.y];
				auto& v2 = cuInstance.h_mesh.positions[f.z];
				inputPoints[i] = (v0 + v1 + v2) / 3.0f;

				new_aabbMin.x = fminf(new_aabbMin.x, inputPoints[i].x);
				new_aabbMin.y = fminf(new_aabbMin.y, inputPoints[i].y);
				new_aabbMin.z = fminf(new_aabbMin.z, inputPoints[i].z);

				new_aabbMax.x = fmaxf(new_aabbMax.x, inputPoints[i].x);
				new_aabbMax.y = fmaxf(new_aabbMax.y, inputPoints[i].y);
				new_aabbMax.z = fmaxf(new_aabbMax.z, inputPoints[i].z);
			}

			TS(Octree);
			HPOctree octree;
			octree.Initialize(inputPoints.data(), inputPoints.size(), { new_aabbMin, new_aabbMax }, 10);
			TE(Octree);

			{
				auto domain_aabb = octree.domainAABB;
				auto domain_length = octree.domain_length;

				vector<HPOctreeNode> h_nodes(octree.numberOfNodes);
				CUDA_COPY_D2H(h_nodes.data(), octree.d_nodes, sizeof(HPOctreeNode)* octree.numberOfNodes);

				function<void(unsigned int, unsigned int)> DrawNode;
				DrawNode = [&](unsigned int nodeIndex, unsigned int depth) {
					const HPOctreeNode& node = h_nodes[nodeIndex];
					const uint64_t& code = node.key.ToCode();
					auto aabb = node.key.GetAABB(domain_aabb, domain_length);
					string name = "HPOctreeKey_" + to_string(node.key.d);
					VD::AddWiredBox(name, { glm::vec3(XYZ(aabb.min)), glm::vec3(XYZ(aabb.max)) }, Color::blue());

					if (node.childCount > 0)
					{
						for (unsigned int i = 0; i < node.childCount; ++i)
						{
							DrawNode(node.firstChildIndex + i, depth + 1);
						}
					}
					};

				DrawNode(0, 0);

				for (size_t i = 0; i <= octree.maxDepth; i++)
				{
					string name = "HPOctreeKey_" + to_string(i);
					VD::AddToSelectionList(name);
				}
			}

			{
				vector<float3> queryPoints(cuInstance.h_mesh.numberOfPoints);
				memcpy(queryPoints.data(), cuInstance.h_mesh.positions, sizeof(float3) * cuInstance.h_mesh.numberOfPoints);

				vector<float3> h_positions(octree.numberOfPositions);
				CUDA_COPY_D2H(h_positions.data(), octree.d_positions, sizeof(float3)* octree.numberOfPositions);

				TS(Query);
				auto results = octree.Search(queryPoints);
				TE(Query);

				for (size_t i = 0; i < results.size(); i++)
				{
					auto& q = queryPoints[i];
					auto& r = h_positions[results[i].index];
					VD::AddLine("NN", { XYZ(q) }, { XYZ(r) }, Color::red());
				}
			}

			{
				/*
				auto domain_length = octree.domain_length;
				auto domain_aabb = octree.domainAABB;

				auto result = octree.Dump();
				unsigned int maxDepth = 0;

				map<uint64_t, unsigned int> countMap;
				for (auto& key : result)
				{
					auto& p = key.ToPosition(domain_aabb, domain_length);
					auto aabb = key.GetAABB(domain_aabb, domain_length);

					if (key.d > maxDepth) maxDepth = key.d;

					countMap[key.ToCode()]++;

					string name = "HPOctreeKey_" + to_string(key.d);
					VD::AddWiredBox(name, { glm::vec3(XYZ(aabb.min)), glm::vec3(XYZ(aabb.max)) }, Color::blue());
				}

				for (auto& kvp : countMap)
				{
					if (kvp.second > 1)
					{
						HPOctreeKey key;
						key.FromCode(kvp.first);
						printf("%d, %d, %d, %d : %d\n", key.d, key.x, key.y, key.z, kvp.second);
					}
				}

				for (size_t i = 0; i <= maxDepth; i++)
				{
					string name = "HPOctreeKey_" + to_string(i);
					VD::AddToSelectionList(name);
				}
				*/
			}

			//vector<float3> h_positions(cuInstance.d_mesh.numberOfPoints);
			//CUDA_COPY_D2H(h_positions.data(), cuInstance.d_mesh.positions, sizeof(float) * cuInstance.d_mesh.numberOfPoints);

			//vector<float3> input(cuInstance.h_mesh.numberOfPoints);
			//memcpy(input.data(), cuInstance.h_mesh.positions, sizeof(float3) * cuInstance.h_mesh.numberOfPoints);
			//auto results = octree.SearchOnDevice(input);

			//for (size_t i = 0; i < cuInstance.h_mesh.numberOfPoints; i++)
			//{
			//	if (results[i].index < cuInstance.d_mesh.numberOfPoints)
			//	{
			//		auto& p0 = input[i];
			//		auto& p1 = h_positions[results[i].index];
			//		VD::AddLine("NN", glm::vec3(XYZ(p0)), glm::vec3(XYZ(p1)), Color::red());
			//	}
			//}
		}

		h_pointCloud.Terminate();
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
