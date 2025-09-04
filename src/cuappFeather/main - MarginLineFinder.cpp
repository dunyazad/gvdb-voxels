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

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001

MarginLineFinder marginLineFinder;
vector<float3> marginLinePoints;


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

		auto InsertPoints = [&]() {
			TS(InsertPoints);
			{
				auto entity = Feather.GetEntityByName("Compound_Class_0");
				if (entity != InvalidEntity)
				{
					auto renderable = Feather.GetComponent<Renderable>(entity);
					if (renderable)
					{
						vector<float3> points;
						for (size_t i = 0; i < renderable->GetVertices().size(); i++)
						{
							auto& p = renderable->GetVertices()[i];
							points.push_back(make_float3(XYZ(p)));
						}
						marginLineFinder.InsertPoints(points, 1);
					}
				}
			}
			{
				auto entity = Feather.GetEntityByName("Compound_Class_1");
				if (entity != InvalidEntity)
				{
					auto renderable = Feather.GetComponent<Renderable>(entity);
					if (renderable)
					{
						vector<float3> points;
						for (size_t i = 0; i < renderable->GetVertices().size(); i++)
						{
							auto& p = renderable->GetVertices()[i];
							points.push_back(make_float3(XYZ(p)));
						}
						marginLineFinder.InsertPoints(points, 2);
					}
				}
			}
			TE(InsertPoints);
			};

		auto Dump = [&]() {
			vector<float3> resultPositions;
			vector<uint64_t> resultTags;
			marginLineFinder.Dump(resultPositions, resultTags);

			auto colors = Color::GetContrastingColors(15);

			for (size_t i = 0; i < resultPositions.size(); i++)
			{
				auto& p = resultPositions[i];
				auto& tag = resultTags[i];

				auto color = colors[tag % colors.size()];

				VD::AddBox("MarginLine Dump", glm::vec3(XYZ(p)), glm::vec3(0.1f, 0.1f, 0.1f), color);
			}
			};

		auto FindMarginLines = [&]() {
			marginLineFinder.FindMarginLinePoints(marginLinePoints);

			for (size_t i = 0; i < marginLinePoints.size(); i++)
			{
				auto& p = marginLinePoints[i];

				VD::AddBox("MarginLine", glm::vec3(XYZ(p)), glm::vec3(0.1f, 0.1f, 0.1f), Color::blue());
			}

			VD::AddToSelectionList("MarginLine");
			};

		auto ClearAndReInitialize = [&]() {
			marginLineFinder.Clear();
			marginLineFinder.Initialize(0.1f, 1000000 * 64, 64);
			};

		auto InsertMarginLinePoints = [&]() {
			marginLineFinder.InsertPoints(marginLinePoints, 3);
			};

		auto MarginLinesNoiseRemove = [&]() {
			vector<float3> result;
			marginLineFinder.MarginLineNoiseRemoval(result);

			for (size_t i = 0; i < result.size(); i++)
			{
				auto& p = result[i];

				VD::AddBox("MarginLineNoiseRemoved", glm::vec3(XYZ(p)), glm::vec3(0.1f, 0.1f, 0.1f), Color::red());
			}

			VD::AddToSelectionList("MarginLineNoiseRemoved");
			};

		auto Clustering = [&]() {
			std::vector<float3> clustered_points;
			std::vector<uint64_t> cluster_ids;
			std::vector<uint64_t> cluster_counts;
			marginLineFinder.Clustering(clustered_points, cluster_ids, cluster_counts);

			// 2. 결과를 그룹화 (std::unordered_map 사용으로 성능 향상)
			std::unordered_map<uint64_t, std::vector<pair<float3, uint64_t>>> clusters;
			for (size_t i = 0; i < clustered_points.size(); ++i)
			{
				//printf("count : %d\n", cluster_counts[i]);

				clusters[cluster_ids[i]].push_back(make_pair(clustered_points[i], cluster_counts[i]));
			}

			if (clusters.empty()) return;

			// 3. 색상 생성
			auto colors = Color::GetContrastingColors(clusters.size());

			// 4. 시각화 (올바른 순회 방식 및 일관된 색상 할당)
			for (auto const& [cluster_id, points] : clusters)
			{
				// 클러스터 ID의 해시 값을 기반으로 일관된 색상 할당
				// 이렇게 하면 실행할 때마다 같은 ID의 클러스터는 같은 색상을 가짐
				size_t hash_val = std::hash<uint64_t>{}(cluster_id);
				auto color = colors[hash_val % colors.size()];

				for (auto const& pair : points)
				{
					if (pair.second < 100)
					{
						//VD::AddWiredBox("ClusteredPoints_Filtered", glm::vec3(XYZ(pair.first)), glm::vec3(1.0f, 1.0f, 1.0f), Color::black());
					}
					else
					{
						VD::AddBox("ClusteredPoints", glm::vec3(XYZ(pair.first)), glm::vec3(0.1f, 0.1f, 0.1f), color);
					}
				}
			}
			};

		controlPanel->AddButton("Insert Points", 0, 0, InsertPoints);
		controlPanel->AddButton("Dump", 0, 0, Dump);
		controlPanel->AddButton("Find Margin Lines", 0, 0, FindMarginLines);
		controlPanel->AddButton("Clear & Re-Initialize", 0, 0, ClearAndReInitialize);
		controlPanel->AddButton("Insert MarginLine Points", 0, 0, InsertMarginLinePoints);
		controlPanel->AddButton("Margin Lines Noise Remove", 0, 0, MarginLinesNoiseRemove);
		controlPanel->AddButton("Clustering", 0, 0, Clustering);

		controlPanel->AddButton("Do", 300, 100, [&](){
			InsertPoints();
			FindMarginLines();
			ClearAndReInitialize();
			InsertMarginLinePoints();
			MarginLinesNoiseRemove();
			VD::ClearAll();
			Clustering();
			});
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		marginLineFinder.Initialize(0.1f, 1000000 * 64, 64);

		string modelPathRoot = "D:\\Debug\\PLY\\";
		string class_0 = modelPathRoot + "Compound_Class_0.ply";
		{
			PLYFormat ply;
			ply.Deserialize(class_0);

			auto entity = Feather.CreateEntity("Compound_Class_0");
			auto renderable = Feather.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::Points);
			renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

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

				renderable->AddVertex(position);
				renderable->AddNormal(normal);
				renderable->AddColor(color);
			}
		}
		
		string class_1 = modelPathRoot + "Compound_Class_1.ply";
		{
			PLYFormat ply;
			ply.Deserialize(class_1);

			auto entity = Feather.CreateEntity("Compound_Class_1");
			auto renderable = Feather.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::Points);
			renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

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

				renderable->AddVertex(position);
				renderable->AddNormal(normal);
				renderable->AddColor(color);
			}
		}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
