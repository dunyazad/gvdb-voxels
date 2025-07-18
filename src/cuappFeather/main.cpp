#pragma warning(disable : 4819)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#include <iostream>
#include <libFeather.h>

#include "main.cuh"

#include <iostream>
using namespace std;


#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

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

#pragma region Print GPU Performance Mode
//{
//	NvAPI_Status status;

//	status = NvAPI_Initialize();
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_Initialize() status != NVAPI_OK\n");
//		return;
//	}

//	NvDRSSessionHandle hSession = 0;
//	status = NvAPI_DRS_CreateSession(&hSession);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_CreateSession() status != NVAPI_OK\n");
//		return;
//	}

//	// (2) load all the system settings into the session
//	status = NvAPI_DRS_LoadSettings(hSession);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_LoadSettings() status != NVAPI_OK\n");
//		return;
//	}

//	NvDRSProfileHandle hProfile = 0;
//	status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_GetBaseProfile() status != NVAPI_OK\n");
//		return;
//	}

//	NVDRS_SETTING drsGet = { 0, };
//	drsGet.version = NVDRS_SETTING_VER;
//	status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_GetSetting() status != NVAPI_OK\n");
//		return;
//	}

//	auto gpu_performance = drsGet.u32CurrentValue;

//	printf("gpu_performance : %d\n", gpu_performance);

//	// (6) We clean up. This is analogous to doing a free()
//	NvAPI_DRS_DestroySession(hSession);
//	hSession = 0;
//}
#pragma endregion

//const string resource_file_name = "0_Initial_Noise";
//const string resource_file_name = "Compound_Full";
const string resource_file_name = "Bridge";
//const string resource_file_name = "Reintegrate";
const string resource_file_name_ply = "../../res/3D/" + resource_file_name + ".ply";
const string resource_file_name_alp = "../../res/3D/" + resource_file_name + ".alp";

const f32 voxelSize = 0.1f;

void ApplyPointCloudToEntity(Entity entity, const HostPointCloud& h_pointCloud);
void ApplyPointCloudToEntity(Entity entity, const DevicePointCloud& d_pointCloud);

void ApplyPointCloudToEntity(Entity entity, const DevicePointCloud& d_pointCloud)
{
	HostPointCloud h_pointCloud(d_pointCloud);
	ApplyPointCloudToEntity(entity, h_pointCloud);
}

void ApplyPointCloudToEntity(Entity entity, const HostPointCloud& h_pointCloud)
{
	auto renderable = Feather.GetComponent<Renderable>(entity);
	if (nullptr == renderable)
	{
		renderable = Feather.CreateComponent<Renderable>(entity);

		renderable->Initialize(Renderable::GeometryMode::Triangles);
		renderable->AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
		renderable->AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
		renderable->SetActiveShaderIndex(1);
	}
	else
	{
		renderable->Clear();
	}

	auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.05f, 6, 6);
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

		renderable->AddInstanceColor(MiniMath::V4(c.x, c.y, c.z, 1.0f));
		renderable->AddInstanceNormal({ n.x, n.y, n.z });

		MiniMath::M4 model = MiniMath::M4::identity();
		model.m[0][0] = 2.0f;
		model.m[1][1] = 2.0f;
		model.m[2][2] = 2.0f;
		model = MiniMath::translate(model, { p.x, p.y, p.z });
		renderable->AddInstanceTransform(model);
	}
	renderable->EnableInstancing(h_pointCloud.numberOfPoints);
}

#pragma once
#include <algorithm>
#include <cmath>

int main(int argc, char** argv)
{
	ForceGPUPerformance();

	cout << "AppFeather" << endl;

	Feather.Initialize(1920, 1080);

	auto w = Feather.GetFeatherWindow();

	thread* cudaThread = nullptr;

	ALPFormat<PointPNC> alp;

	bool tick = false;

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&tick](Entity entity, const KeyEvent& event) {
			if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			//else if (GLFW_KEY_TAB == event.keyCode)
			//{
			//	if (0 == event.action)
			//	{
			//		auto o = Feather.GetEntityByName("Input Point Cloud _ O");
			//		Feather.GetComponent<Renderable>(o)->SetVisible(tick);
			//		auto p = Feather.GetEntityByName("Input Point Cloud");
			//		Feather.GetComponent<Renderable>(p)->SetVisible(!tick);

			//		tick = !tick;
			//	}
			//}
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

		Feather.CreateEventCallback<MouseButtonEvent>(cam, [](Entity entity, const MouseButtonEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMouseButton(event);
			});

		Feather.CreateEventCallback<MouseWheelEvent>(cam, [](Entity entity, const MouseWheelEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMouseWheel(event);
			});
	}
#pragma endregion

#pragma region Status Panel
	{
		auto gui = Feather.CreateEntity("Status Panel");
		auto statusPanel = Feather.CreateComponent<StatusPanel>(gui);

		Feather.CreateEventCallback<MousePositionEvent>(gui, [](Entity entity, const MousePositionEvent& event) {
			auto component = Feather.GetComponent<StatusPanel>(entity);
			component->mouseX = event.xpos;
			component->mouseY = event.ypos;
			});
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {

#define PLY_Model_File
#ifdef PLY_Model_File
#pragma region Load PLY and Convert to ALP format
		{
			auto t = Time::Now();

			if (false == alp.Deserialize(resource_file_name_alp))
			{
				bool foundZero = false;

				PLYFormat ply;
				ply.Deserialize(resource_file_name_ply);
				//ply.SwapAxisYZ();

				vector<PointPNC> points;
				for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
				{
					auto px = ply.GetPoints()[i * 3];
					auto py = ply.GetPoints()[i * 3 + 1];
					auto pz = ply.GetPoints()[i * 3 + 2];

					if (0 == px && 0 == py && 0 == pz)
					{
						if (false == foundZero)
						{
							foundZero = true;
						}
						else
						{
							continue;
						}
					}

					auto nx = ply.GetNormals()[i * 3];
					auto ny = ply.GetNormals()[i * 3 + 1];
					auto nz = ply.GetNormals()[i * 3 + 2];

					if (false == ply.GetColors().empty())
					{
						if (ply.UseAlpha())
						{
							auto cx = ply.GetColors()[i * 4];
							auto cy = ply.GetColors()[i * 4 + 1];
							auto cz = ply.GetColors()[i * 4 + 2];
							auto ca = ply.GetColors()[i * 4 + 3];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
						else
						{
							auto cx = ply.GetColors()[i * 3];
							auto cy = ply.GetColors()[i * 3 + 1];
							auto cz = ply.GetColors()[i * 3 + 2];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
					}
					else
					{
						points.push_back({ {px, py, pz}, {nx, ny, nz}, {1.0f, 1.0f, 1.0f} });
					}
				}
				alog("PLY %llu points loaded\n", points.size());

				alp.AddPoints(points);
				alp.Serialize(resource_file_name_alp);
			}

			t = Time::End(t, "Loading Compound");

			auto entity = Feather.CreateEntity("Input Point Cloud _ O");

			HostPointCloud h_pointCloud;
			h_pointCloud.Intialize(alp.GetPoints().size());

			for (size_t i = 0; i < alp.GetPoints().size(); i++)
			{
				auto& p = alp.GetPoints()[i];

				h_pointCloud.positions[i] = make_float3(p.position.x, p.position.y, p.position.z);
				h_pointCloud.normals[i] = make_float3(p.normal.x, p.normal.y, p.normal.z);
				h_pointCloud.colors[i] = make_float3(p.color.x, p.color.y, p.color.z);
			}
			ApplyPointCloudToEntity(entity, h_pointCloud);

			alog("ALP %llu points loaded\n", alp.GetPoints().size());

			auto [x, y, z] = alp.GetAABBCenter();
			f32 cx = x;
			f32 cy = y;
			f32 cz = z;

			auto [mx, my, mz] = alp.GetAABBMin();
			auto [Mx, My, Mz] = alp.GetAABBMax();
			f32 lx = Mx - mx;
			f32 ly = My - my;
			f32 lz = Mz - mz;

			printf("min : %f, %f, %f\n", mx, my, mz);
			printf("max : %f, %f, %f\n", Mx, My, Mz);
			printf("dimensions : %f, %f, %f\n", lx, ly, lz);

			Entity cam = Feather.GetEntityByName("Camera");
			auto pcam = Feather.GetComponent<PerspectiveCamera>(cam);
			auto cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(cam);
			cameraManipulator->SetRadius(lx + ly + lz);
			auto camera = cameraManipulator->GetCamera();
			camera->SetEye({ cx,cy,cz + cameraManipulator->GetRadius() });
			camera->SetTarget({ cx,cy,cz });

			cameraManipulator->MakeDefault();


			Feather.CreateEventCallback<KeyEvent>(entity, [cx, cy, cz, lx, ly, lz](Entity entity, const KeyEvent& event) {
				auto renderable = Feather.GetComponent<Renderable>(entity);
				if (GLFW_KEY_M == event.keyCode)
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
				else if (GLFW_KEY_F1 == event.keyCode)
				{
					//for (size_t i = 0; i < host_colors.size(); i++)
					//{
					//	auto& color = host_colors[i];
					//	renderable->SetInstanceColor(i, MiniMath::V4(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f, 1.0f));
					//}
				}
				else if (GLFW_KEY_F2 == event.keyCode)
				{
					//for (size_t i = 0; i < host_colors.size(); i++)
					//{
					//	auto& color = host_colors_hsv[i];
					//	renderable->SetInstanceColor(i, MiniMath::V4(color.x, color.y, 0.25f, 1.0f));
					//}
				}
				else if (GLFW_KEY_F3 == event.keyCode)
				{
				}
				//else if (GLFW_KEY_R == event.keyCode)
				//{
				//	auto entities = Feather.GetRegistry().view<CameraManipulatorTrackball>();
				//	for (auto& entity : entities)
				//	{
				//		auto cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);

				//		cameraManipulator.SetRadius((lx + ly + lz) * 2.0f);

				//		auto camera = cameraManipulator.GetCamera();
				//		camera->SetEye({ cx,cy,cz + cameraManipulator.GetRadius() });
				//		camera->SetTarget({ cx,cy,cz });
				//	}
				//}
				});

			t = Time::End(t, "Upload to GPU");

			auto hashToFloat = [](uint32_t seed) -> float {
				seed ^= seed >> 13;
				seed *= 0x5bd1e995;
				seed ^= seed >> 15;
				return (seed & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
			};

			auto result = ProcessPointCloud(h_pointCloud);
			ApplyPointCloudToEntity(entity, result);

			PLYFormat ply;
			for (size_t i = 0; i < result.numberOfPoints; i++)
			{
				auto& p = result.positions[i];
				if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;

				auto& n = result.normals[i];
				auto& c = result.colors[i];

				ply.AddPoint(p.x, p.y, p.z);
				ply.AddNormal(n.x, n.y, n.z);
				ply.AddColor(c.x, c.y, c.z);
			}
			ply.Serialize("../../res/3D/VoxelHashMap.ply");

			result.Terminate();
			h_pointCloud.Terminate();

			//for (size_t i = 0; i < host_colors.size(); i++)
			//{
			//	auto& color = host_colors[i];
			//	renderable->SetInstanceColor(i, MiniMath::V4(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f, 1.0f));
			//}

			{ // AABB
				auto m = alp.GetAABBMin();
				float x = get<0>(m);
				float y = get<1>(m);
				float z = get<2>(m);
				auto M = alp.GetAABBMax();
				float X = get<0>(M);
				float Y = get<1>(M);
				float Z = get<2>(M);

				auto entity = Feather.CreateEntity("AABB");
				auto renderable = Feather.CreateComponent<Renderable>(entity);
				renderable->Initialize(Renderable::GeometryMode::Lines);

				renderable->AddShader(Feather.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

				renderable->AddVertex({ x, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable->AddVertex({ x, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable->AddVertex({ x, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ X, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable->AddVertex({ x, Y, Z }); renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
			}
		}
#pragma endregion
#endif
	});

	Feather.AddOnUpdateCallback([](f32 timeDelta) {
	
		});

	Feather.Run();

	Feather.Terminate();
	
	return 0;
}
