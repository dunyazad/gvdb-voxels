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

#include <Eigen/Sparse>
using SparseMatrixf = Eigen::SparseMatrix<float>;
using VecXf = Eigen::VectorXf;

using VD = VisualDebugging;

#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

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
	Feather.SetConsoleWindowIndex(1);
	Feather.SetMainWindowIndex(2);

	auto w = Feather.GetFeatherWindow();

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

		unsigned int OCTREE_DEPTH = 12;

		//inpterpolatedColors = Color::InterpolateColors({ Color::blue(), Color::yellow(), Color::black(), Color::green(), Color::red() }, 16);
		auto inpterpolatedColors = Color::InterpolateColors({ Color::blue(), Color::red() }, OCTREE_DEPTH);

		//cuInstance.Test();

			HostPointCloud<PointCloudProperty> h_input;
			h_input.DeserializePLY("D:\\Debug\\PLY\\input.ply");

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			/*	{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);

				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}

			{
				TS(H2D);
				CUDA_SYNC();
				DevicePointCloud<PointCloudProperty> d_input = h_input;
				CUDA_SYNC();
				TE(H2D);
			
				TS(D2H);
				CUDA_SYNC();
				HostPointCloud<PointCloudProperty> h_result = d_input;
				CUDA_SYNC();
				TE(D2H);
			}*/

			auto entity = Feather.CreateEntity("PointCloud");
			auto aabb = ApplyPointCloudToEntity(entity, h_input);

			auto center = (aabb.min + aabb.max) * 0.5f;
			auto axisLength = aabb.max - aabb.min;
			auto maxLength = fmaxf(fmaxf(axisLength.x, axisLength.y), axisLength.z);
			auto halfLength = maxLength * 0.5f;

			printf("maxLength : %f\n", maxLength);

			aabb.min.x = center.x - halfLength;
			aabb.min.y = center.y - halfLength;
			aabb.min.z = center.z - halfLength;

			aabb.max.x = center.x + halfLength;
			aabb.max.y = center.y + halfLength;
			aabb.max.z = center.z + halfLength;

			{
				TS(BuildOCtree_CPU);

				std::vector<float3> inputPositions(h_input.numberOfPoints);
				//std::vector<float3> inputNormals(h_input.numberOfPoints);
				memcpy(inputPositions.data(), h_input.positions, sizeof(float3) * h_input.numberOfPoints);
				//memcpy(inputNormals.data(), h_input.normals, sizeof(float3) * h_input.numberOfPoints);

				HybridOctree octree(OCTREE_DEPTH);
				octree.Build(inputPositions, { XYZ(aabb.min), XYZ(aabb.max) }, 1, OCTREE_DEPTH);

				TE(BuildOCtree_CPU);

				//octree.Draw();

				//for (size_t i = 0; i <= OCTREE_DEPTH; i++)
				//{
				//	std::string name = "octree_" + std::to_string(i);
				//	VD::AddToSelectionList(name);
				//}

#pragma region QueryPoints
//				{
//					TS(QueryPoint);
//					auto queryResult = octree.QueryPoints(inputPositions);
//					TE(QueryPoint);
//					for (auto& r : queryResult)
//					{
//						if (0 > r) continue;
//
//						auto& node = octree.host_nodes[r];
//						VD::AddWiredBox(
//							"queried_nodes",
//							{ glm::vec3(XYZ(node.bounds.min)), glm::vec3(XYZ(node.bounds.max)) },
//							Color::green()
//						);
//					}
//				}
#pragma endregion


#pragma region QueryRegion
//				{
//					TS(QueryRange);
//					std::vector<unsigned int> outFlatIndices;
//					std::vector<unsigned int> outCounts;
//					cuAABB aabb = { {-5.0f, -5.0f, -5.0f}, {5.0f, 5.0f, 5.0f} };
//					octree.QueryRange({ aabb }, 500000, outFlatIndices, outCounts);
//					TE(QueryRange);
//
//					size_t offset = 0;
//					for (size_t q = 0; q < outCounts.size(); ++q)
//					{
//						unsigned int count = outCounts[q];
//						for (unsigned int i = 0; i < count; ++i)
//						{
//							unsigned int nodeIdx = outFlatIndices[offset + i];
//							if (nodeIdx < octree.host_nodes.size())
//							{
//								const auto& node = octree.host_nodes[nodeIdx];
//
//								for (unsigned int pointIdx : node.indices)
//								{
//									const auto& p = h_input.positions[pointIdx];
//									if (aabb.contains(p))
//									{
//										VD::AddSphere("queried_points", { XYZ(p) }, 0.02f, Color::red());
//									}
//								}
//							}
//						}
//						offset += count;
//					}
//
//					VD::AddWiredBox("AABB", { {-5.0f, -5.0f, -5.0f}, {5.0f, 5.0f, 5.0f} }, Color::red());
//				}
#pragma endregion


#pragma region QueryRadius
				{
					TS(QueryRadius);
					std::vector<unsigned int> outFlatIndices;
					std::vector<unsigned int> outCounts;
					octree.QueryRadius({ { 0.0f, 0.0f, 0.0f } }, { 5.0f }, 500000, outFlatIndices, outCounts);
					TE(QueryRadius);

					size_t offset = 0;
					for (size_t q = 0; q < outCounts.size(); ++q)
					{
						unsigned int count = outCounts[q];
						for (unsigned int i = 0; i < count; ++i)
						{
							unsigned int nodeIdx = outFlatIndices[offset + i];
							if (nodeIdx < octree.host_nodes.size())
							{
								const auto& node = octree.host_nodes[nodeIdx];

								for (unsigned int pointIdx : node.indices)
								{
									const auto& p = h_input.positions[pointIdx];
									if (5.0f > sqrtf(p.x * p.x + p.y * p.y + p.z * p.z))
									{
										VD::AddSphere("queried_points", { XYZ(p) }, 0.1f, Color::red());
									}
								}
							}
						}
						offset += count;
					}
				}
#pragma endregion

//#pragma region QueryNearestNode
//				{
//					std::vector<float3> inputPositions(h_input.numberOfPoints);
//					memcpy(inputPositions.data(), h_input.positions, sizeof(float3)* h_input.numberOfPoints);
//					for (size_t i = 0; i < inputPositions.size(); i++)
//					{
//						auto& p = inputPositions[i];
//						p.z += 10.0f;
//					}
//
//					TS(QueryNearestNode);
//					std::vector<unsigned int> outFlatIndices;
//					std::vector<unsigned int> outCounts;
//					auto indices = octree.QueryNearestPoint(inputPositions, inputPositions);
//					TE(QueryNearestNode);
//
//					size_t offset = 0;
//					for (size_t i = 0; i < indices.size(); i++)
//					{
//						unsigned int nodeIdx = indices[i];
//						if (nodeIdx < octree.host_nodes.size())
//						{
//							const auto& node = octree.host_nodes[nodeIdx];
//							for (unsigned int pointIdx : node.indices)
//							{
//								const auto& p = h_input.positions[pointIdx];
//								auto n = normalize(inputPositions[i] - p);
//								auto np = p + n;
//								
//								//VD::AddSphere("queried_points", { XYZ(p) }, 0.1f, Color::red());
//								VD::AddLine("queried_line", { XYZ(p) }, { XYZ(np) }, Color::red(), Color::red());
//							}
//						}
//					}
//				}
//#pragma endregion

			}

			TS(Octree);
			HPOctree octree;
			octree.Initialize(h_input.positions, h_input.numberOfPoints, { {XYZ(aabb.min)}, {XYZ(aabb.max)}}, OCTREE_DEPTH);
			TE(Octree);

			//{
			//	auto domain_aabb = octree.domainAABB;
			//	auto domain_length = octree.domain_length;

			//	std::vector<HPOctreeNode> h_nodes(octree.numberOfNodes);
			//	CUDA_COPY_D2H(h_nodes.data(), octree.d_nodes, sizeof(HPOctreeNode) * octree.numberOfNodes);

			//	std::function<void(unsigned int, unsigned int)> DrawNode;
			//	DrawNode = [&](unsigned int nodeIndex, unsigned int depth) {
			//		const HPOctreeNode& node = h_nodes[nodeIndex];
			//		const uint64_t& code = node.key.ToCode();
			//		auto aabb = node.key.GetAABB(domain_aabb, domain_length);
			//		std::string name = "HPOctreeKey_" + std::to_string(node.key.d);
			//		VD::AddWiredBox(name, { glm::vec3(XYZ(aabb.min)), glm::vec3(XYZ(aabb.max)) }, Color::blue());

			//		if (node.childCount > 0)
			//		{
			//			for (unsigned int i = 0; i < node.childCount; ++i)
			//			{
			//				DrawNode(node.firstChildIndex + i, depth + 1);
			//			}
			//		}
			//		};

			//	DrawNode(0, 0);

			//	for (size_t i = 0; i <= octree.maxDepth; i++)
			//	{
			//		std::string name = "HPOctreeKey_" + std::to_string(i);
			//		VD::AddToSelectionList(name);
			//	}
			//}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
