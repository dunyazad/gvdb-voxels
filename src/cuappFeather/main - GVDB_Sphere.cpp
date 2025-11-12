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

#include "main.cuh"
#include <libFeather.h>

#include "HPOctree.cuh"
#include "HybridOctree.cuh"

//#include <Eigen/Sparse>
//using SparseMatrixf = Eigen::SparseMatrix<float>;
//using VecXf = Eigen::VectorXf;

using VD = VisualDebugging;

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001


#pragma region GVDB Debugging
static size_t fmt2_sz = 0;
static char* fmt2 = NULL;
static FILE* fd = NULL;
static bool bLogReady = false;
static bool bPrintLogging = true;
static int  printLevel = -1; // <0 mean no level prefix

void sample_print(int argc, char const* argv)
{
}

void nvprintSetLevel(int l)
{
	printLevel = l;
}
int nvprintGetLevel()
{
	return printLevel;
}
void nvprintSetLogging(bool b)
{
	bPrintLogging = b;
}
void nvprintf2(va_list& vlist, const char* fmt, int level)
{
	if (bPrintLogging == false)
		return;
	if (fmt2_sz == 0) {
		fmt2_sz = 1024;
		fmt2 = (char*)malloc(fmt2_sz);
	}
	while ((vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
	{
		fmt2_sz *= 2;
		if (fmt2) free(fmt2);
		fmt2 = (char*)malloc(fmt2_sz);
	}
#ifdef WIN32
	OutputDebugStringA(fmt2);
#ifdef _DEBUG

	if (bLogReady == false)
	{
		fd = fopen("Log.txt", "w");
		bLogReady = true;
	}
	if (fd)
	{
		//fprintf(fd, prefix);
		fprintf(fd, fmt2);
	}
#endif
#endif
	sample_print(level, fmt2);
	//::printf(prefix);
	::printf(fmt2);
}
void nvprintf(const char* fmt, ...)
{
	//    int r = 0;
	va_list  vlist;
	va_start(vlist, fmt);
	nvprintf2(vlist, fmt, printLevel);
}
void nvprintfLevel(int level, const char* fmt, ...)
{
	va_list  vlist;
	va_start(vlist, fmt);
	nvprintf2(vlist, fmt, level);
}
void nverror()
{
	nvprintf("Error. Application will exit.");
	exit(-1);
}
void checkGL(const char* msg)
{
	GLenum errCode;
	errCode = glGetError();
	if (errCode != GL_NO_ERROR) {
		nvprintf("%s, GL ERROR: 0x%x\n", msg, errCode);
	}
}
#pragma endregion

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

CUDAInstance cuInstance;

AABB ApplyPointCloudToEntity(Entity entity, const HostPointCloud<PointCloudProperty>& h_pointCloud)
{
	Feather.CreateEventCallback<KeyEvent>(entity, [](Entity entity, const KeyEvent& event) {
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

	auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere({ 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6);
	//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
	renderable->AddIndices(indices);
	renderable->AddVertices(vertices);
	renderable->AddNormals(normals);
	renderable->AddColors(colors);
	renderable->AddUVs(uvs);

	AABB aabb{ {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX} };

	auto cs = Color::GetContrastingColors(32);
	std::map<unsigned int, unsigned int> colorMap;

	for (size_t i = 0; i < h_pointCloud.numberOfPoints; i++)
	{
		auto& p = h_pointCloud.positions[i];
		if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;

		auto& n = h_pointCloud.normals[i];
		auto& c = h_pointCloud.colors[i];
		auto l = h_pointCloud.properties[i].label;

		//if (0 == l) continue;

		if (colorMap.end() == colorMap.find(l))
		{
			colorMap[l] = colorMap.size() % 32;
		}

		auto mc = cs[colorMap[l]];

		glm::vec3 position(XYZ(p));
		glm::vec3 normal(XYZ(n));
		glm::vec4 color(XYZ(c), 1.0f);

		aabb.min.x = std::min(aabb.min.x, position.x);
		aabb.min.y = std::min(aabb.min.y, position.y);
		aabb.min.z = std::min(aabb.min.z, position.z);

		aabb.max.x = std::max(aabb.max.x, position.x);
		aabb.max.y = std::max(aabb.max.y, position.y);
		aabb.max.z = std::max(aabb.max.z, position.z);

		renderable->AddInstanceColor(color);
		//renderable->AddInstanceColor(mc);
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

		tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
		
		renderable->AddInstanceTransform(tm);
		renderable->IncreaseNumberOfInstances();
	}

	return aabb;
}

AABB GetDomainAABB(const AABB& aabb)
{
	auto delta = aabb.max - aabb.min;
	float hl = std::max({ delta.x, delta.y, delta.z });
	auto center = (aabb.min + aabb.max) * 0.5f;
	float half_length = hl * 0.5f;
	return {
		{center.x - half_length, center.y - half_length, center.z - half_length},
		{center.x + half_length, center.y + half_length, center.z + half_length}
	};
}

std::tuple<AABB, std::vector<AABB>> SplitAABB(const AABB& input_aabb, unsigned int cellsPerAxis)
{
	AABB totalAABB{ {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX} };

	std::vector<AABB> subAABBs;
	if (cellsPerAxis == 0) return { totalAABB, subAABBs };

	auto aabb = GetDomainAABB(input_aabb);
	auto domainSize = aabb.max - aabb.min;
	auto cellSize = domainSize / static_cast<float>(cellsPerAxis);

	domainSize += cellSize;
	cellSize = domainSize / static_cast<float>(cellsPerAxis);

	for (unsigned int z = 0; z < cellsPerAxis; ++z)
	{
		for (unsigned int y = 0; y < cellsPerAxis; ++y)
		{
			for (unsigned int x = 0; x < cellsPerAxis; ++x)
			{
				glm::vec3 cellMin =
				{
					aabb.min.x + x * cellSize.x - cellSize.x * 0.5f,
					aabb.min.y + y * cellSize.y - cellSize.y * 0.5f,
					aabb.min.z + z * cellSize.z - cellSize.z * 0.5f
				};
				glm::vec3 cellMax = cellMin + cellSize;
				AABB subAABB = { cellMin, cellMax };

				if (input_aabb.Intersects(subAABB))
				{
					totalAABB.min.x = std::min(totalAABB.min.x, cellMin.x);
					totalAABB.min.y = std::min(totalAABB.min.y, cellMin.y);
					totalAABB.min.z = std::min(totalAABB.min.z, cellMin.z);

					totalAABB.max.x = std::max(totalAABB.max.x, cellMax.x);
					totalAABB.max.y = std::max(totalAABB.max.y, cellMax.y);
					totalAABB.max.z = std::max(totalAABB.max.z, cellMax.z);

					subAABBs.push_back({ cellMin, cellMax });
				}
			}
		}
	}

	return { totalAABB, subAABBs };
}

void SortByAABBs(HostPointCloud<PointCloudProperty>& pointCloud, const AABB& totalAABB, const std::vector<AABB>& subAABBs)
{
	if (pointCloud.numberOfPoints == 0 || subAABBs.empty())
	{
		return;
	}

	std::vector<std::vector<unsigned int>> bins(subAABBs.size());
	for (unsigned int i = 0; i < pointCloud.numberOfPoints; ++i)
	{
		const auto& p = pointCloud.positions[i];
		for (size_t j = 0; j < subAABBs.size(); ++j)
		{
			if (subAABBs[j].Contains({ XYZ(p) }))
			{
				bins[j].push_back(i);
				break;
			}
		}
	}

	HostPointCloud<PointCloudProperty> sortedPointCloud;
	sortedPointCloud.Initialize(pointCloud.numberOfPoints);

	unsigned int writeIndex = 0;

	auto colors = Color::GetContrastingColors(bins.size());
	int colorIndex = 0;
	for (const auto& bin : bins)
	{
		auto color = colors[colorIndex++];

		for (const auto originalIndex : bin)
		{
			sortedPointCloud.positions[writeIndex] = pointCloud.positions[originalIndex];
			sortedPointCloud.normals[writeIndex] = pointCloud.normals[originalIndex];
			sortedPointCloud.colors[writeIndex] = pointCloud.colors[originalIndex];

			if constexpr (!std::is_void_v<PointCloudProperty>)
			{
				sortedPointCloud.properties[writeIndex] = pointCloud.properties[originalIndex];
			}

			//VD::AddWiredBox("sorted", { XYZ(pointCloud.positions[originalIndex]) }, { 0.05f, 0.05f, 0.05f }, color);

			writeIndex++;
		}
	}

	pointCloud = std::move(sortedPointCloud);
}

int main(int argc, char** argv)
{
	ForceGPUPerformance();
	CUDA_FREE(0);

	std::cout << "AppFeather" << std::endl;

	Feather.Initialize(1920, 1080);
	Feather.SetClearColor(Color::slategray());
	Feather.SetConsoleWindowIndex(3);
	Feather.SetMainWindowIndex(2);

	auto w = Feather.GetFeatherWindow();
	
	VolumeGVDB gvdb;
	Vector3DF	m_pretrans, m_scale, m_angs, m_trans;

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&](Entity entity, const KeyEvent& event) {
			//printf("KeyEvent : keyCode=%d, scanCode=%d, action=%d, mods=%d\n", event.keyCode, event.scanCode, event.action, event.mods);

			if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			else if (GLFW_KEY_SPACE == event.keyCode)
			{
				if (event.action == 1)
				{
					Feather.GetImmediateModeRenderSystem()->ToggleEnable();
				}
			}
			else if (GLFW_KEY_BACKSPACE == event.keyCode)
			{
				VD::ClearAll();
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
			/*
			else if(GLFW_KEY_GRAVE_ACCENT == event.keyCode)
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
			*/
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

		auto Temp = [&]() {
			};
	
		//controlPanel->AddButton("Temp", 0, 0, Temp);
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {
		{
			{
				auto entity = Feather.CreateEntity("gvdb plane");
				auto renderable = Feather.CreateComponent<Renderable>(entity);
				renderable->Initialize(Renderable::GeometryMode::Triangles);
				renderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Texturing.vs"), File("../../res/Shaders/Texturing.fs")));

				auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildPlane(192, 108, 2, 2, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f });
				renderable->AddIndices(indices);
				renderable->AddVertices(vertices);
				renderable->AddNormals(normals);
				renderable->AddColors(colors);
				renderable->AddUVs(uvs);

				auto texture = Feather.CreateComponent<Texture>(entity);
				texture->AllocTextureData(1920, 1080);
				texture->GetTextureID();

				texture->Bind();
			}
			




			int w = 1920;
			int h = 1080;
			m_pretrans.Set(-125, -160, -125);
			m_scale.Set(1, 1, 1);
			m_angs.Set(0, 0, 0);
			m_trans.Set(0, 0, 0);

			gvdb.SetVerbose(true);

			// Initialize GVDB	
			gvdb.SetVerbose(true);
			gvdb.SetCudaDevice(GVDB_DEV_FIRST);
			gvdb.Initialize();
			gvdb.AddPath("../../External/gvdb-voxels/source/shared_assets/");

			// Load VBX
			char scnpath[1024];
			if (!gvdb.getScene()->FindFile("explosion.vbx", scnpath)) {
				printf("Cannot find vbx file.\n");
			}
			printf("Loading VBX. %s\n", scnpath);
			gvdb.SetChannelDefault(16, 16, 16);
			//gvdb.LoadVBX(scnpath);



			float radius = 100.0f;

			gvdb.Clear();
			gvdb.Configure(3, 3, 3, 3, 5);
			gvdb.AddChannel(0, T_FLOAT, 1, 0);

			const int numSamples = 100000;
			const Vector3DF center(0, 0, 0);

			// 대략 반경 100 근처만 활성화
			for (int z = -2; z <= 2; ++z)
				for (int y = -2; y <= 2; ++y)
					for (int x = -2; x <= 2; ++x)
						gvdb.ActivateSpace(Vector3DI(x, y, z));

			gvdb.FinishTopology();
			gvdb.UpdateAtlas();

			// 포인트 클라우드 생성
			std::vector<Vector3DF> points(numSamples);
			std::vector<Vector3DF> colors(numSamples, Vector3DF(1, 0.5f, 0.2f));

			for (int i = 0; i < numSamples; i++) {
				float u = (float)rand() / RAND_MAX;
				float v = (float)rand() / RAND_MAX;
				float theta = 2.0f * 3.1415926f * u;
				float phi = acosf(2.0f * v - 1.0f);
				float x = radius * sinf(phi) * cosf(theta);
				float y = radius * sinf(phi) * sinf(theta);
				float z = radius * cosf(phi);
				points[i] = Vector3DF(center.x + x, center.y + y, center.z + z);
			}

			// 포인트 데이터 등록
			DataPtr pntpos, pntclr, dummy;
			gvdb.AllocData(pntpos, numSamples, sizeof(Vector3DF));
			gvdb.AllocData(pntclr, numSamples, sizeof(Vector3DF));
			gvdb.SetDataCPU(pntpos, numSamples, (char*)points.data(), 0, sizeof(Vector3DF));
			gvdb.SetDataCPU(pntclr, numSamples, (char*)colors.data(), 0, sizeof(Vector3DF));
			gvdb.SetPoints(pntpos, dummy, pntclr);

			// 서브셀 생성 및 밀도 계산
			int subcell_size = 4;
			float gather_radius = 6.0f;
			int scPntLen = 0;

			gvdb.InsertPointsSubcell(subcell_size, numSamples, gather_radius, Vector3DF(0, 0, 0), scPntLen);
			if (scPntLen == 0) {
				printf("Warning: No subcells created. Points likely outside of volume bounds.\n");
				return;
			}

			gvdb.GatherDensity(subcell_size, numSamples, gather_radius, Vector3DF(0, 0, 0),
				scPntLen, 0, 1, true);
			gvdb.UpdateApron();

			printf("Sphere SDF generated successfully. Points: %d, subcells: %d\n",
				numSamples, scPntLen);







			//// 모든 복셀 값을 0으로 초기화
			////gvdb.FillChannel(0, { 0.0f, 0.0f, 0.0f, 0.0f });

			//printf("[GVDB] 5000³ logical volume initialized.\n");

			//printf("--------------------------------------------------\n");
			//printf("[GVDB] Diagnostic Info\n");
			//printf("  Level 0 nodes : %d\n", gvdb.getNumNodes(0));
			//printf("  Level 1 nodes : %d\n", gvdb.getNumNodes(1));
			//printf("  Level 2 nodes : %d\n", gvdb.getNumNodes(2));
			//printf("--------------------------------------------------\n");

			//// Set volume params
			//gvdb.SetTransform(m_pretrans, m_scale, m_angs, m_trans);
			//gvdb.getScene()->SetSteps(.25f, 16, .25f);			// Set raycasting steps
			//gvdb.getScene()->SetExtinct(-1.0f, 1.0f, 0.0f);		// Set volume extinction
			//gvdb.getScene()->SetVolumeRange(0.1f, 0.0f, .5f);	// Set volume value range
			//gvdb.getScene()->SetCutoff(0.005f, 0.005f, 0.0f);
			//gvdb.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);
			//gvdb.getScene()->LinearTransferFunc(0.00f, 0.25f, Vector4DF(0, 0, 0, 0), Vector4DF(1, 0, 0, 0.05f));
			//gvdb.getScene()->LinearTransferFunc(0.25f, 0.50f, Vector4DF(1, 0, 0, 0.05f), Vector4DF(1, .5f, 0, 0.1f));
			//gvdb.getScene()->LinearTransferFunc(0.50f, 0.75f, Vector4DF(1, .5f, 0, 0.1f), Vector4DF(1, 1, 0, 0.15f));
			//gvdb.getScene()->LinearTransferFunc(0.75f, 1.00f, Vector4DF(1, 1, 0, 0.15f), Vector4DF(1, 1, 1, 0.2f));
			//gvdb.CommitTransferFunc();


			// Create Camera 
			Camera3D* cam = new Camera3D;
			cam->setFov(50.0);
			cam->setOrbit(Vector3DF(30, 45, 0), Vector3DF(2500, 2500, 2500), 10000, 1.0);
			gvdb.getScene()->SetCamera(cam);

			// Create Light
			Light* lgt = new Light;
			lgt->setOrbit(Vector3DF(299, 57.3f, 0), Vector3DF(132, -20, 50), 200, 1.0);
			gvdb.getScene()->SetLight(0, lgt);

			// Add render buffer
			printf("Creating screen buffer. %d x %d\n", w, h);
			gvdb.AddRenderBuf(0, w, h, 4);

			// Cleanup
			//cudaFree(d_tf);
		}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {

		m_angs.y += timeDelta * 0.01f;
		gvdb.SetTransform(m_pretrans, m_scale, m_angs, m_trans);

		// Render volume
		gvdb.TimerStart();
		gvdb.Render(SHADE_VOLUME, 0, 0);
		float rtime = gvdb.TimerStop();

		// Copy render buffer into opengl texture
		// This function does a gpu-gpu device copy from the gvdb cuda output buffer
		// into the opengl texture, avoiding the cpu readback found in ReadRenderBuf

		auto entity = Feather.GetEntityByName("gvdb plane");
		auto texture = Feather.GetComponent<Texture>(entity);
		texture->Bind();
		gvdb.ReadRenderTexGL(0, texture->GetTextureID());
		checkGL("After ReadRenderTexGL");
		//texture->Unbind();
		
		});

	Feather.Run();

	gvdb.Clear();

	Feather.Terminate();

	return 0;
}
