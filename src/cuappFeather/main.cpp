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

		controlPanel->AddButton("DLClasses", 0, 0, [&]() {
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
			});

		controlPanel->AddButton("Border Lines (Non-manifold)", 0, 0, [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;
			if (h_mesh.numberOfFaces == 0) return;

			// 1. 데이터 구조 변경: 한 시작 정점에 여러 경계 에지가 연결될 수 있도록 vector 사용
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

			// 2. 모든 경계 에지를 처리할 때까지 루프 추적
			while (!borderHalfEdgeIndices.empty())
			{
				std::vector<unsigned int> currentRing;
				unsigned int startHeIdx = *borderHalfEdgeIndices.begin();
				unsigned int currentHeIdx = startHeIdx;
				unsigned int prevHeIdx = UINT32_MAX; // 이전 에지를 추적하기 위한 변수

				do
				{
					currentRing.push_back(currentHeIdx);
					borderHalfEdgeIndices.erase(currentHeIdx);

					const auto& currentHe = h_mesh.halfEdges[currentHeIdx];
					const auto& nextHeInFace = h_mesh.halfEdges[currentHe.nextIndex];
					unsigned int endVertex = nextHeInFace.vertexIndex;

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

				borderRings.push_back(currentRing);
			}

			// 4. (Optional) Visualize the found border loops.
			printf("Assembled into %zu border ring(s).\n", borderRings.size());

			// Define some distinct colors for visualizing different rings.
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

						// Get the start and end vertex positions for the edge.
						const auto& v0 = h_mesh.positions[he.vertexIndex];
						const auto& v1 = h_mesh.positions[nextHe.vertexIndex];

						// Use your visualization utility to add the line segment.
						VD::AddLine(lineName, { XYZ(v0) }, { XYZ(v1) }, color, color);
					}
				}
			}
			});

		controlPanel->AddButton("Show Vertex DLClasses", 0, 0, [&]() {
			HostHalfEdgeMesh<PointCloudProperty> h_mesh = cuInstance.d_mesh;
			for (size_t i = 0; i < h_mesh.numberOfPoints; i++)
			{
				auto& p = h_mesh.positions[i];
				auto& cl = h_mesh.properties[i].deepLearningClass;

				if(0 != cl && 1 != cl)
				VD::AddText("DeepLearningClass", std::to_string(cl), { XYZ(p) });
			}
			});

		controlPanel->AddButton("Margin Lines", 0, 0, [&]() {
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
					VD::AddLine("MarginLine", { XYZ(v) }, { XYZ(ov) }, Color::red(), Color::red());
				}
			}
			});


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

		cuInstance.ProcessPointCloud(h_pointCloud, 0.1f, 3);

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
