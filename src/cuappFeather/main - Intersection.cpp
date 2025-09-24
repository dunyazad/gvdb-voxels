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

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001

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

			if (GLFW_KEY_GRAVE_ACCENT == event.keyCode)
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

		auto findIntersections = [&]() {
			// 1. 교차를 계산할 메쉬 엔티티와 Renderable 컴포넌트를 가져옵니다.
			auto meshEntity = Feather.GetEntityByName("Mesh");
			if (meshEntity == InvalidEntity)
			{
				cout << "Error: 'Mesh' entity not found." << endl;
				return;
			}
			auto renderable = Feather.GetComponent<Renderable>(meshEntity);
			if (!renderable)
			{
				cout << "Error: Renderable component not found on 'Mesh' entity." << endl;
				return;
			}

			// Renderable로부터 버텍스와 인덱스 데이터를 가져옵니다. (메서드 이름은 실제 클래스에 맞게 확인 필요)
			const auto& vertices = renderable->GetVertices();
			const auto& indices = renderable->GetIndices();

			if (vertices.empty() || indices.empty())
			{
				cout << "Warning: Mesh data is empty." << endl;
				return;
			}

			// 2. 교차를 테스트할 수학적 평면을 정의합니다. (별도 struct 없이)
			// 예: Y = 0.5 평면
			glm::vec3 planeNormal = { 0.0f, 1.0f, 0.0f }; // 평면의 법선 벡터
			glm::vec3 pointOnPlane = { 0.0f, 0.0f, 0.0f }; // 평면 위의 한 점
			// 평면 방정식: dot(normal, P) - dot(normal, pointOnPlane) = 0
			// d = -dot(normal, pointOnPlane)
			float planeD = -glm::dot(planeNormal, pointOnPlane);

			cout << "Calculating intersections with plane (Normal: " << planeNormal.y << ", Point: " << pointOnPlane.y << ")" << endl;

			// 3. 모든 교차 선분들을 저장할 벡터
			std::vector<glm::vec3> intersectionLinePoints;

			// 4. 메쉬의 모든 삼각형을 순회하며 교차점을 찾습니다.
			for (size_t i = 0; i < indices.size(); i += 3)
			{
				// 현재 삼각형의 세 정점
				const glm::vec3& v0 = vertices[indices[i]];
				const glm::vec3& v1 = vertices[indices[i + 1]];
				const glm::vec3& v2 = vertices[indices[i + 2]];

				// 각 정점과 평면 사이의 부호 있는 거리 계산
				float d0 = glm::dot(planeNormal, v0) + planeD;
				float d1 = glm::dot(planeNormal, v1) + planeD;
				float d2 = glm::dot(planeNormal, v2) + planeD;

				// 임시 변수에 저장하여 순회
				glm::vec3 triVerts[3] = { v0, v1, v2 };
				float distances[3] = { d0, d1, d2 };
				std::vector<glm::vec3> currentTriangleIntersectionPoints;

				// 삼각형의 각 변(edge)에 대해 교차 검사
				for (int j = 0; j < 3; ++j)
				{
					int k = (j + 1) % 3; // 다음 정점 인덱스

					// 두 정점의 거리 부호가 다르면, 해당 변은 평면을 관통함
					if (distances[j] * distances[k] < 0)
					{
						// 선형 보간으로 정확한 교차점 계산
						float t = distances[j] / (distances[j] - distances[k]);
						glm::vec3 intersectionPoint = triVerts[j] + t * (triVerts[k] - triVerts[j]);
						currentTriangleIntersectionPoints.push_back(intersectionPoint);
					}
				}

				// 한 삼각형에서 교차점이 2개 발생했다면, 이는 교차 선분을 의미함
				if (currentTriangleIntersectionPoints.size() == 2)
				{
					intersectionLinePoints.push_back(currentTriangleIntersectionPoints[0]);
					intersectionLinePoints.push_back(currentTriangleIntersectionPoints[1]);
				}
			}

			// 5. 찾은 교차선들을 시각적으로 디버깅 (VisualDebugging 사용)
			VD::ClearAll(); // 이전 디버그 드로잉 초기화
			cout << "Found " << intersectionLinePoints.size() / 2 << " intersection line segments." << endl;
			for (size_t i = 0; i < intersectionLinePoints.size(); i += 2)
			{
				// VD에 선분을 추가하는 기능이 있다고 가정 (VD::AddLine)
				VD::AddLine("Intersection", intersectionLinePoints[i], intersectionLinePoints[i + 1], Color::cyan());
			}
			};

		controlPanel->AddButton("Find Intersections", 0, 0, findIntersections);
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		string modelPathRoot = "D:\\Debug\\PLY\\";
		string meshName = "Mesh.ply";
		PLYFormat ply;
		ply.Deserialize(modelPathRoot + meshName);

		{
			auto entity = Feather.CreateEntity("Mesh");
			auto renderable = Feather.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::Triangles);
			renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
			renderable->AddShader(Feather.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
			renderable->AddShader(Feather.CreateShader("Flat", File("../../res/Shaders/Flat.vs"), File("../../res/Shaders/Flat.gs"), File("../../res/Shaders/Flat.fs")));
			renderable->SetActiveShaderIndex(0);


			auto n = ply.GetPoints().size() / 3;
			renderable->AddVertices((glm::vec3*)ply.GetPoints().data(), n);
			renderable->AddNormals((glm::vec3*)ply.GetNormals().data(), n);
			renderable->AddColors((glm::vec3*)ply.GetColors().data(), n);
			renderable->AddIndices((ui32*)ply.GetTriangleIndices().data(), ply.GetTriangleIndices().size());

			Feather.CreateEventCallback<KeyEvent>(entity, [&](Entity entity, const KeyEvent& event) {
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
		//{
		//	auto entity = Feather.CreateEntity("Plane");
		//	auto renderable = Feather.CreateComponent<Renderable>(entity);
		//	renderable->Initialize(Renderable::Triangles);
		//	//renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
		//	//renderable->AddShader(Feather.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
		//	renderable->AddShader(Feather.CreateShader("Flat", File("../../res/Shaders/Flat.vs"), File("../../res/Shaders/Flat.gs"), File("../../res/Shaders/Flat.fs")));
		//	//renderable->SetActiveShaderIndex(0);


		//	auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildPlane(20.0f, 20.0f, { 0.0f, 0.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, Color::red());
		//	//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
		//	renderable->AddIndices(indices);
		//	renderable->AddVertices(vertices);
		//	renderable->AddNormals(normals);
		//	renderable->AddColors(colors);
		//	renderable->AddUVs(uvs);
		//}

		auto FindIntersections = [&](
			const float* positions,
			unsigned int numberOfPositions,
			const unsigned int* triangleIndices,
			unsigned int numberOfTriangleIndices,
			glm::vec3 planePoint,
			glm::vec3 planeNormal) -> vector<glm::vec3> {
				vector<glm::vec3> intersectionLinePoints;

				float planeD = -glm::dot(planeNormal, planePoint);

				unsigned int numberOfTriangles = numberOfTriangleIndices / 3;
				for (unsigned int i = 0; i < numberOfTriangles; i++)
				{
					unsigned int baseIndex = i * 3;

					auto i0 = triangleIndices[baseIndex + 0];
					auto i1 = triangleIndices[baseIndex + 1];
					auto i2 = triangleIndices[baseIndex + 2];

					const glm::vec3 v0 = { positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2] };
					const glm::vec3 v1 = { positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2] };
					const glm::vec3 v2 = { positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2] };

					float d0 = glm::dot(planeNormal, v0) + planeD;
					float d1 = glm::dot(planeNormal, v1) + planeD;
					float d2 = glm::dot(planeNormal, v2) + planeD;

					glm::vec3 triVerts[3] = { v0, v1, v2 };
					float distances[3] = { d0, d1, d2 };
					std::vector<glm::vec3> currentTriangleIntersectionPoints;

					for (int j = 0; j < 3; ++j)
					{
						int k = (j + 1) % 3;

						if (distances[j] * distances[k] < 0)
						{
							float t = distances[j] / (distances[j] - distances[k]);
							glm::vec3 intersectionPoint = triVerts[j] + t * (triVerts[k] - triVerts[j]);
							currentTriangleIntersectionPoints.push_back(intersectionPoint);
						}
					}

					if (currentTriangleIntersectionPoints.size() == 2)
					{
						intersectionLinePoints.push_back(currentTriangleIntersectionPoints[0]);
						intersectionLinePoints.push_back(currentTriangleIntersectionPoints[1]);
					}
				}

				return intersectionLinePoints;
			};

		auto intersectionLinePoints = FindIntersections(
			ply.GetPoints().data(),
			ply.GetPoints().size() / 3,
			ply.GetTriangleIndices().data(),
			ply.GetTriangleIndices().size(),
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f));

		cout << "Found " << intersectionLinePoints.size() / 2 << " intersection line segments." << endl;
		for (size_t i = 0; i < intersectionLinePoints.size(); i += 2)
		{
			VD::AddLine("Intersection", intersectionLinePoints[i], intersectionLinePoints[i + 1], Color::cyan());
		}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}