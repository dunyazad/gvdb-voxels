#pragma warning(disable : 4819)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#include <iostream>
#include <libFeather.h>

#include "main.cuh"

#include <iostream>
using namespace std;

using VD = VisualDebugging;

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

CUDAInstance cudaInstance;

glm::vec3 initialPosition;
unsigned int halfEdgeIndex = 2000;
unsigned int vertexIndex = 0;
vector<unsigned int> oneRing;
vector<float3> oneRingPositions;

//#define SAVE_VOXEL_HASHMAP_POINT_CLOUD

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

const string resource_file_name = "0_Initial";
//const string resource_file_name = "0_Initial_Noise";
//const string resource_file_name = "Compound_Full";
//const string resource_file_name = "Bridge";
//const string resource_file_name = "Reintegrate";
//const string resource_file_name = "KOL";
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
		tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
		renderable->AddInstanceTransform(tm);
		renderable->IncreaseNumberOfInstances();
	}
}

void ApplyHalfEdgeMeshToEntity(Entity entity, const HostHalfEdgeMesh& h_mesh);
void ApplyHalfEdgeMeshToEntity(Entity entity, const DeviceHalfEdgeMesh& d_mesh);

void ApplyHalfEdgeMeshToEntity(Entity entity, const HostHalfEdgeMesh& h_mesh)
{
	auto renderable = Feather.GetComponent<Renderable>(entity);
	if (nullptr == renderable)
	{
		renderable = Feather.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::GeometryMode::Triangles);
		renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
		renderable->AddShader(Feather.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
		renderable->SetActiveShaderIndex(0);
	}
	else
	{
		renderable->Clear();
	}

	printf("numPoints: %d, numFaces: %d\n", cudaInstance.h_mesh.numberOfPoints, cudaInstance.h_mesh.numberOfFaces);
	for (unsigned int i = 0; i < cudaInstance.h_mesh.numberOfFaces; ++i)
	{
		auto tri = cudaInstance.h_mesh.faces[i];
		if (tri.x >= cudaInstance.h_mesh.numberOfPoints ||
			tri.y >= cudaInstance.h_mesh.numberOfPoints ||
			tri.z >= cudaInstance.h_mesh.numberOfPoints)
		{
			printf("Invalid face[%u]: %u %u %u\n", i, tri.x, tri.y, tri.z);
		}
	}

	if (h_mesh.numberOfPoints == 0 || h_mesh.numberOfFaces == 0)
	{
		printf("Mesh is empty\n");
		return;
	}

	for (unsigned int i = 0; i < h_mesh.numberOfPoints; ++i)
	{
		auto position = glm::vec3(XYZ(h_mesh.positions[i]));
		renderable->AddVertex(position);
		if (h_mesh.normals) renderable->AddNormal({ h_mesh.normals[i].x, h_mesh.normals[i].y, h_mesh.normals[i].z });
		if (h_mesh.colors) renderable->AddColor({ h_mesh.colors[i].x, h_mesh.colors[i].y, h_mesh.colors[i].z, 1.0f });
	}

	for (unsigned int i = 0; i < h_mesh.numberOfFaces; ++i)
	{
		const auto& tri = h_mesh.faces[i];
		if (tri.x >= h_mesh.numberOfPoints || tri.y >= h_mesh.numberOfPoints || tri.z >= h_mesh.numberOfPoints)
		{
			printf("Face %d has out-of-range index: %d %d %d (max: %d)\n", i, tri.x, tri.y, tri.z, h_mesh.numberOfPoints - 1);
			continue;
		}
		renderable->AddIndex(tri.x);
		renderable->AddIndex(tri.y);
		renderable->AddIndex(tri.z);
	}
}
void ApplyHalfEdgeMeshToEntity(Entity entity, const DeviceHalfEdgeMesh& d_mesh)
{
	HostHalfEdgeMesh h_mesh(d_mesh);
	ApplyHalfEdgeMeshToEntity(entity, h_mesh);

	h_mesh.Terminate();
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

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&](Entity entity, const KeyEvent& event) {
			if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			else if (GLFW_KEY_BACKSPACE == event.keyCode)
			{
				VD::ClearAll();
			}
			else if (GLFW_KEY_F1 == event.keyCode)
			{
				for (unsigned int i = 0; i < cudaInstance.h_mesh.numberOfPoints; ++i)
				{
					auto position = glm::vec3(XYZ(cudaInstance.h_mesh.positions[i]));

					stringstream ss;
					ss << i;
					VD::AddText("OneRingVertices", ss.str(), position, Color::white());
				}
			}
			else if (GLFW_KEY_ENTER == event.keyCode)
			{
				printf("Finding Border\n");

				auto& mesh = cudaInstance.h_mesh;

				for (size_t i = 0; i < mesh.numberOfFaces * 3; i++)
				{
					auto& he = mesh.halfEdges[i];
					if (UINT32_MAX == he.oppositeIndex)
					{
						auto ne = mesh.halfEdges[he.nextIndex];

						auto& v0 = mesh.positions[he.vertexIndex];
						auto& v1 = mesh.positions[ne.vertexIndex];

						VD::AddLine("BorderLines", { XYZ(v0) }, { XYZ(v1) }, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });
					}
				}
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
				auto ray = manipulator->GetCamera()->ScreenPointToRay(event.xpos, event.ypos, w->GetWidth(), w->GetHeight());
				//VD::Clear("PickingRay");
				//VD::AddBox("PickingBox", ray.origin, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });

				int hitIndex = -1;
				float outHit = FLT_MAX;
				if (cudaInstance.h_mesh.PickFace(
					make_float3(ray.origin.x, ray.origin.y, ray.origin.z),
					make_float3(ray.direction.x, ray.direction.y, ray.direction.z),
					hitIndex, outHit))
				{
					auto& f = cudaInstance.h_mesh.faces[hitIndex];
					auto& p0 = cudaInstance.h_mesh.positions[f.x];
					auto& p1 = cudaInstance.h_mesh.positions[f.y];
					auto& p2 = cudaInstance.h_mesh.positions[f.z];

					auto v0 = glm::vec3(XYZ(p0));
					auto v1 = glm::vec3(XYZ(p1));
					auto v2 = glm::vec3(XYZ(p2));

					auto normal = glm::trianglenormal(v0, v1, v2);

					//VD::AddTriangle("Picked", v0, v1, v2, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });
					VD::AddSphere("PickedS", v0, normal, 0.025f, { 1.0f, 0.0f, 0.0f, 1.0f });
					VD::AddSphere("PickedS", v1, normal, 0.025f, { 0.0f, 1.0f, 0.0f, 1.0f });
					VD::AddSphere("PickedS", v2, normal, 0.025f, { 0.0f, 0.0f, 1.0f, 1.0f });
					auto hitPosition = ray.origin + glm::normalize(ray.direction) * outHit;
					VD::AddSphere("PickedPosition", hitPosition, normal, 0.025f, { 0.0f, 0.0f, 1.0f, 1.0f });

					VD::AddLine("PickingRay", ray.origin, hitPosition, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });

					if (GLFW_MOD_CONTROL == event.mods)
					{
						camera->SetTarget(hitPosition);
						camera->SetEye(hitPosition + (-ray.direction) * manipulator->GetRadius());
					}

					auto d0 = glm::distance2(hitPosition, v0);
					auto d1 = glm::distance2(hitPosition, v1);
					auto d2 = glm::distance2(hitPosition, v2);
					unsigned int vi = -1;
					if (d0 < d1 && d0 < d2)
					{
						auto vn = glm::vec3(XYZ(cudaInstance.h_mesh.normals[f.x]));
						VD::AddWiredBox("Nearest", v0, vn, { 0.05f, 0.05, 0.05 }, { 1.0f, 1.0f, 1.0f, 1.0f });
						vi = f.x;
					}
					else if (d1 < d0 && d1 < d2)
					{
						auto vn = glm::vec3(XYZ(cudaInstance.h_mesh.normals[f.y]));
						VD::AddWiredBox("Nearest", v1, vn, { 0.05f, 0.05, 0.05 }, { 1.0f, 1.0f, 1.0f, 1.0f });
						vi = f.y;
					}
					else if (d2 < d0 && d2 < d1)
					{
						auto vn = glm::vec3(XYZ(cudaInstance.h_mesh.normals[f.z]));
						VD::AddWiredBox("Nearest", v2, vn, { 0.05f, 0.05, 0.05 }, { 1.0f, 1.0f, 1.0f, 1.0f });
						vi = f.z;
					}

					printf("[h_mesh] hitIndex : %d, outHit : %f\n", hitIndex, outHit);

					printf("OneRing Center : %d\n", vi);
					auto vis = cudaInstance.h_mesh.GetOneRingVertices(vi);
					for (auto& vi : vis)
					{
						printf("%d, ", vi);
						glm::vec3 position = glm::vec3(XYZ(cudaInstance.h_mesh.positions[vi]));
						VD::AddWiredBox("OneRing", position, normal, { 0.025f, 0.025, 0.025 }, Color::green());

						stringstream ss;
						ss << vi;
						VD::AddText("OneRingVertices", ss.str(), position, Color::white());
					}
					printf("\n");



					cudaInstance.d_mesh.PickFace(
						make_float3(ray.origin.x, ray.origin.y, ray.origin.z),
						make_float3(ray.direction.x, ray.direction.y, ray.direction.z),
						hitIndex, outHit);

					printf("[d_mesh] hitIndex : %d, outHit : %f\n", hitIndex, outHit);

				}
			}
			});

		Feather.CreateEventCallback<MouseWheelEvent>(cam, [](Entity entity, const MouseWheelEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity)->OnMouseWheel(event);
			});
	}
#pragma endregion

#pragma region Status Panel
	{
		auto gui = Feather.CreateEntity("Panel");

		auto panel = Feather.CreateComponent<Panel>(gui, "Mouse Position");
		Feather.CreateEventCallback<MousePositionEvent>(gui, [](Entity entity, const MousePositionEvent& event) {
			auto component = Feather.GetComponent<Panel>(entity);
			component->mouseX = event.xpos;
			component->mouseY = event.ypos;
			});
	}
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

				alp.FromPLY(ply);
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
			//ApplyPointCloudToEntity(entity, h_pointCloud);

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
				if (nullptr == renderable) return;

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
					//	renderable->SetInstanceColor(i, glm::vec4(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f, 1.0f));
					//}
				}
				else if (GLFW_KEY_F2 == event.keyCode)
				{
					//for (size_t i = 0; i < host_colors.size(); i++)
					//{
					//	auto& color = host_colors_hsv[i];
					//	renderable->SetInstanceColor(i, glm::vec4(color.x, color.y, 0.25f, 1.0f));
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

			auto result = cudaInstance.ProcessPointCloud(h_pointCloud);
			//ApplyPointCloudToEntity(entity, result);

			auto meshEntity = Feather.CreateEntity("MarchingCubesMesh");
			ApplyHalfEdgeMeshToEntity(meshEntity, cudaInstance.h_mesh);

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

#ifdef SAVE_VOXEL_HASHMAP_POINT_CLOUD
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
#endif // SAVE_VOXEL_HASHMAP_POINT_CLOUD

			//result.Terminate();
			h_pointCloud.Terminate();
		}
#pragma endregion
#endif

		{ // AABB
			auto m = alp.GetAABBMin();
			float x = get<0>(m);
			float y = get<1>(m);
			float z = get<2>(m);
			auto M = alp.GetAABBMax();
			float X = get<0>(M);
			float Y = get<1>(M);
			float Z = get<2>(M);

			auto width = w->GetWidth();
			auto height = w->GetHeight();

			Entity cam = Feather.GetEntityByName("Camera");
			auto pcam = Feather.GetComponent<PerspectiveCamera>(cam);

			auto projection = pcam->GetProjectionMatrix();
			auto view = pcam->GetViewMatrix();

			VD::AddWiredBox("AABB", { x, y, z }, { X, Y, Z }, Color::blue());
		}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
