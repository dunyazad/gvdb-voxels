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
vector<int> deepLearningClasses;

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

		controlPanel->AddButton("GetNearestPoints", 0, 0, [&]() {
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
				if (0 == deepLearningClasses[i])// || 1 == deepLearningClasses[i])
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
			});
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		string modelPathRoot = "D:\\Debug\\PLY\\";
		string compound = modelPathRoot + "Compound.ply";

		HostPointCloud h_pointCloud;

		PLYFormat ply;
		ply.Deserialize(compound);

		auto [cx, cy, cz] = ply.GetAABBCenter();
		auto [mx, my, mz] = ply.GetAABBMin();
		auto [Mx, My, Mz] = ply.GetAABBMax();
		f32 lx = Mx - mx;
		f32 ly = My - my;
		f32 lz = Mz - mz;

		h_pointCloud.Intialize(ply.GetPoints().size() / 3);

		//auto entity = Feather.CreateEntity("Compound");
		//auto renderable = Feather.CreateComponent<Renderable>(entity);
		//renderable->Initialize(Renderable::Points);
		//renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		deepLearningClasses.clear();

		for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
		{
			auto deepLearningClass = ply.GetDeepLearningClasses()[i];
			deepLearningClasses.push_back(deepLearningClass);

			//if (1 < deepLearningClass)
			//{
			//	continue;
			//}

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
		}

		cuInstance.ProcessPointCloud(h_pointCloud, 0.2f);

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

		h_pointCloud.Terminate();
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
