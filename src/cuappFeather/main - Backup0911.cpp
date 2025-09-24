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
					auto entity = Feather.GetEntityByName("PointCloud");
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
					auto entity = Feather.GetEntityByName("PointCloud");
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
					VD::ToggleVisibility("Normals");
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

		auto ToggleNormal = [&]() {
			VD::ToggleVisibility("Normals");
			};

		controlPanel->AddButton("Toggle Normals", 0, 0, ToggleNormal);

	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		std::vector<float> pointVariances;
		float maxVariance = 0.0f;
		float minVariance = FLT_MAX;

		auto heatmapColor = [](float value, float minVal, float maxVal) -> glm::vec4 {
			if (maxVal <= minVal) return { 0.5f, 0.5f, 0.5f, 1.0f }; // 분산이 없는 경우 회색

			float t = (value - minVal) / (maxVal - minVal); // 0.0 ~ 1.0 사이로 정규화

			float r = 0.0f, g = 0.0f, b = 0.0f;
			if (t < 0.5f) { // Blue -> Green
				b = 1.0f - (t * 2.0f);
				g = t * 2.0f;
			}
			else { // Green -> Red
				g = 1.0f - ((t - 0.5f) * 2.0f);
				r = (t - 0.5f) * 2.0f;
			}
			return { r, g, b, 1.0f };
			};



		string modelPathRoot = "D:\\Debug\\PLY\\";
		//string compound = modelPathRoot + "Compound.ply";
		string patch = modelPathRoot + "Source0.ply";
		string dlMap = modelPathRoot + "DeepLearningInference.ply";

		unsigned short dlClass[400 * 480];
		{
			PLYFormat ply;
			ply.Deserialize(dlMap);
			for (size_t y = 0; y < 480; y++)
			{
				for (size_t x = 0; x < 400; x++)
				{
					auto dl = ply.GetLabels()[y * 400 + (400 - x - 1)];
					dlClass[y * 400 + x] = dl;

					//VD::AddBox("DL", { (float)x, (float)y, 0.0f }, { 0.5f, 0.5f, 0.5f }, dl == 7 ? Color::white() : Color::red());
				}
			}
		}

		std::vector<glm::vec4> colors = {
				Color::red(),
				Color::green(),
				Color::blue(),
				Color::yellow(),
				Color::magenta(),
				Color::cyan()
		};

		HostPointCloud<PointCloudProperty> h_pointCloud;

		PLYFormat ply;
		ply.Deserialize(patch);

		auto [cx, cy, cz] = ply.GetAABBCenter();
		auto [mx, my, mz] = ply.GetAABBMin();
		auto [Mx, My, Mz] = ply.GetAABBMax();
		f32 lx = Mx - mx;
		f32 ly = My - my;
		f32 lz = Mz - mz;

		auto aabbMin = make_float3(mx, my, mz);
		auto aabbMax = make_float3(Mx, My, Mz);

		h_pointCloud.Initialize(ply.GetPoints().size() / 3);
		pointVariances.resize(h_pointCloud.numberOfPoints, 0.0f);

		for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
		{
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

			auto tx = x + (25.6f / 2.0f) * 0.5f;
			auto ty = y + (48.0f / 2.0f) * 0.5f;

			auto rx = tx / (25.6f / 2.0f);
			auto ry = ty / (48.0f / 2.0f);
			auto indexX = (unsigned int)floorf(rx * 400.0f);
			auto indexY = (unsigned int)floorf(ry * 480.0f);
			auto dl = dlClass[indexY * 400 + indexX];

			auto dlcolor = colors[dl - 6];

			h_pointCloud.positions[i] = make_float3(x, y, z);
			h_pointCloud.normals[i] = make_float3(nx, ny, nz);
			h_pointCloud.colors[i] = make_float3(r, g, b);

			float normalScale = 0.25f;
			//VD::AddLine("Normals", { x,y,z }, { x + nx * normalScale, y + ny * normalScale, z + nz * normalScale }, Color::white());
			VD::AddLine("Normals", { x,y,z }, { x + nx * normalScale, y + ny * normalScale, z + nz * normalScale }, dlcolor);
		}

		struct GridElement
		{
			unsigned int pointIndices[8];
			unsigned int numberOfIndices = 0;
		};
		GridElement* grid = new GridElement[256 * 480];

		printf("Populating grid...\n");
		for (size_t i = 0; i < h_pointCloud.numberOfPoints; i++)
		{
			auto& p = h_pointCloud.positions[i];
			if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;

			auto tp = p + make_float3(12.8f, 24.0f, 0.0f);
			unsigned int xIndex = (unsigned int)floorf(tp.x * 10.0f);
			unsigned int yIndex = (unsigned int)floorf(tp.y * 10.0f);

			if (xIndex < 256 && yIndex < 480) {
				unsigned int flattenIndex = yIndex * 256 + xIndex;
				if (grid[flattenIndex].numberOfIndices < 8) {
					grid[flattenIndex].pointIndices[grid[flattenIndex].numberOfIndices++] = i;
				}
			}
		}
		printf("...Grid populated.\n");

		/*
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
				tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
				renderable->AddInstanceTransform(tm);
				renderable->IncreaseNumberOfInstances();

				{
					auto tp = p + make_float3(12.8f, 24.0f, 0.0f);
					unsigned int xIndex = (unsigned int)floorf(tp.x * 10.0f);
					unsigned int yIndex = (unsigned int)floorf(tp.y * 10.0f);
					unsigned int flattenIndex = yIndex * 256 + xIndex;
					grid[flattenIndex].pointIndices[grid[flattenIndex].numberOfIndices++] = i;
				}
			}
		}
		*/

		printf("\nCalculating variance and covariance for neighboring normals...\n");
		for (unsigned int i = 0; i < h_pointCloud.numberOfPoints; ++i)
		{
			const auto& p = h_pointCloud.positions[i];
			if (p.x == FLT_MAX) continue;

			auto tp = p + make_float3(12.8f, 24.0f, 0.0f);
			int xIndex = static_cast<int>(floorf(tp.x * 10.0f));
			int yIndex = static_cast<int>(floorf(tp.y * 10.0f));

			std::vector<float3> neighborNormals;
			int offset = 3;
			for (int dy = -offset; dy <= offset; ++dy) {
				for (int dx = -offset; dx <= offset; ++dx) {
					int currentX = xIndex + dx;
					int currentY = yIndex + dy;
					if (currentX >= 0 && currentX < 256 && currentY >= 0 && currentY < 480) {
						unsigned int flattenIndex = currentY * 256 + currentX;
						GridElement& cell = grid[flattenIndex];
						for (unsigned int k = 0; k < cell.numberOfIndices; ++k) {
							neighborNormals.push_back(h_pointCloud.normals[cell.pointIndices[k]]);
						}
					}
				}
			}

			if (neighborNormals.size() < 2) continue;

			float3 meanNormal = { 0.f, 0.f, 0.f };
			for (const auto& n : neighborNormals) {
				meanNormal.x += n.x; meanNormal.y += n.y; meanNormal.z += n.z;
			}
			float numNeighbors = static_cast<float>(neighborNormals.size());
			meanNormal.x /= numNeighbors; meanNormal.y /= numNeighbors; meanNormal.z /= numNeighbors;

			float3 variance = { 0.f, 0.f, 0.f };
			for (const auto& n : neighborNormals) {
				float3 dev = { n.x - meanNormal.x, n.y - meanNormal.y, n.z - meanNormal.z };
				variance.x += dev.x * dev.x;
				variance.y += dev.y * dev.y;
				variance.z += dev.z * dev.z;
			}
			float n_minus_1 = numNeighbors - 1.0f;
			variance.x /= n_minus_1;
			variance.y /= n_minus_1;
			variance.z /= n_minus_1;

			float varianceMagnitude = sqrtf(variance.x * variance.x + variance.y * variance.y + variance.z * variance.z);
			pointVariances[i] = varianceMagnitude;

			if (varianceMagnitude > maxVariance) maxVariance = varianceMagnitude;
			if (varianceMagnitude < minVariance) minVariance = varianceMagnitude;
		}
		printf("...Done. minVariance: %f, maxVariance: %f\n\n", minVariance, maxVariance);

		{
			auto entity = Feather.CreateEntity("PointCloud");
			auto renderable = Feather.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
			renderable->AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
			renderable->SetActiveShaderIndex(1);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere({ 0.0f, 0.0f, 0.0f }, 0.05f, 6, 6);
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

				glm::vec3 position(XYZ(p));
				glm::vec3 normal(XYZ(n));

				float variance = pointVariances[i];
				glm::vec4 color = heatmapColor(variance, minVariance, maxVariance);

				renderable->AddInstanceColor(color);
				renderable->AddInstanceNormal(normal);

				glm::mat4 tm = glm::identity<glm::mat4>();
				glm::mat4 rot = glm::mat4(1.0f);
				if (glm::length(normal) > 0.0001f) {
					glm::vec3 axis = glm::normalize(glm::cross(glm::vec3(0, 0, 1), normal));
					float angle = acos(glm::dot(glm::normalize(normal), glm::vec3(0, 0, 1)));
					if (glm::length(axis) > 0.0001f)
						rot = glm::rotate(glm::mat4(1.0f), angle, axis);
				}
				tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
				renderable->AddInstanceTransform(tm);
				renderable->IncreaseNumberOfInstances();
			}
		}

		h_pointCloud.Terminate();

		delete[] grid;
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
