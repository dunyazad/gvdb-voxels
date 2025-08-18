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

CUDAInstance cuInstance;

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

//const string resource_file_name = "diagonal";
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
		tm = glm::translate(tm, position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(0.125f));
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

	printf("numPoints: %d, numFaces: %d\n", cuInstance.h_mesh.numberOfPoints, cuInstance.h_mesh.numberOfFaces);
	for (unsigned int i = 0; i < cuInstance.h_mesh.numberOfFaces; ++i)
	{
		auto tri = cuInstance.h_mesh.faces[i];
		if (tri.x >= cuInstance.h_mesh.numberOfPoints ||
			tri.y >= cuInstance.h_mesh.numberOfPoints ||
			tri.z >= cuInstance.h_mesh.numberOfPoints)
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

int main(int argc, char** argv)
{
	//PLYFormat ply;
	//for (size_t i = 0; i < 1000; i++)
	//{
	//	ply.AddPoint(-50.0f + (float)i * 0.1f, -50.0f + (float)i * 0.1f, -50.0f + (float)i * 0.1f);
	//	ply.AddNormal(0.0f, 1.0f, 0.0f);
	//}
	////ply.Serialize("../../res/3D/diagonal.ply");
	//
	//auto aabbMin = make_float3(get<0>(ply.GetAABBMin()), get<1>(ply.GetAABBMin()), get<2>(ply.GetAABBMin()));
	//auto aabbMax = make_float3(get<0>(ply.GetAABBMax()), get<1>(ply.GetAABBMax()), get<2>(ply.GetAABBMax()));

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
				for (unsigned int i = 0; i < cuInstance.h_mesh.numberOfPoints; ++i)
				{
					auto position = glm::vec3(XYZ(cuInstance.h_mesh.positions[i]));

					stringstream ss;
					ss << i;
					VD::AddText("OneRingVertices", ss.str(), position, Color::white());
				}
			}
			else if (GLFW_KEY_TAB == event.keyCode)
			{
				if (event.action == 1)
				{
					auto entity = Feather.GetEntityByName("MarchingCubesMesh");
					auto renderable = Feather.GetComponent<Renderable>(entity);
					auto as = renderable->GetActiveShaderIndex();
					renderable->SetActiveShaderIndex((as + 1) % renderable->GetShaders().size());
				}
			}
			else if (GLFW_KEY_SPACE == event.keyCode)
			{
				if (event.action == 1)
				{
					//cudaInstance.d_mesh.RadiusLaplacianSmoothing(0.5f, 10, 0.05f);
					cuInstance.d_mesh.LaplacianSmoothing(5, 1.0f, false);
					cuInstance.interop.UploadFromDevice(cuInstance.d_mesh);

					VD::Clear("AABB");
					VD::AddWiredBox("AABB", { {XYZ(cuInstance.d_mesh.min)}, {XYZ(cuInstance.d_mesh.max)} }, Color::blue());
				}
			}
			else if (GLFW_KEY_ENTER == event.keyCode)
			{
				printf("Finding Border\n");

				auto& mesh = cuInstance.h_mesh;

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
			else if (GLFW_KEY_INSERT == event.keyCode)
			{
				if (event.action == 1)
					/*
					{ // AABBs
						static unsigned int count = 0;
						Feather.RemoveEventCallback<FrameEvent>(appMain);
						count = 0;

						static vector<cuAABB> aabbs;
						if (aabbs.empty())
						{
							aabbs = cudaInstance.d_mesh.GetAABBs();
						}
						Feather.CreateEventCallback<FrameEvent>(appMain, [&](Entity entity, const FrameEvent& event) {
							for (size_t i = 0; i < 10000; i++)
							{
								if (count < aabbs.size())
								{
									auto& aabb = aabbs[count++];
									VD::AddWiredBox("temp", { {XYZ(aabb.min)}, {XYZ(aabb.max)} }, Color::green());
								}
								if (count == aabbs.size())
								{
									aabbs.clear();
									Feather.RemoveEventCallback<FrameEvent>(appMain);
									break;
								}
							}
							});
					}
					*/
				{
					static unsigned int count = 0;
					Feather.RemoveEventCallback<FrameEvent>(appMain);
					count = 0;

					static vector<uint64_t> mortonCodes;
					if (mortonCodes.empty())
					{
						mortonCodes = cuInstance.d_mesh.GetMortonCodes();
						std::sort(mortonCodes.begin(), mortonCodes.end());
					}
					Feather.CreateEventCallback<FrameEvent>(appMain, [&](Entity entity, const FrameEvent& event) {
						float3 aabb_extent = cuInstance.d_mesh.max - cuInstance.d_mesh.min;
						float max_extent = fmaxf(aabb_extent.x, fmaxf(aabb_extent.y, aabb_extent.z));
						float voxelSize = max_extent / ((1 << 21) - 2); // safety margin

						for (size_t i = 0; i < 10000; i++)
						{
							if (count < mortonCodes.size())
							{
								auto mortonCode = mortonCodes[count++];
								auto position = Morton64ToFloat3(mortonCode, cuInstance.d_mesh.min, voxelSize);
								VD::AddWiredBox("temp", { XYZ(position) }, { 0.0f, 1.0f, 0.0f }, glm::vec3(voxelSize * 100.0f), Color::green());
							}
							if (count == mortonCodes.size())
							{
								mortonCodes.clear();
								Feather.RemoveEventCallback<FrameEvent>(appMain);
								break;
							}
						}
						});
				}
			}
			else if (GLFW_KEY_HOME == event.keyCode)
			{
				if (event.action == 1)
				{
					cuInstance.d_mesh.BuildFaceNodeHashMap();
					////////cudaInstance.d_mesh.DebugPrintFaceNodeHashMap();
					//////auto unlinked = cuInstance.d_mesh.FindUnlinkedFaceNodes();
					//////printf("unlinked : %d\n", unlinked.size());

					//////vector<float3> positions(cuInstance.d_mesh.numberOfPoints);
					//////vector<uint3> faces(cuInstance.d_mesh.numberOfFaces);
					//////CUDA_COPY_D2H(positions.data(), cuInstance.d_mesh.positions, sizeof(float3) * cuInstance.d_mesh.numberOfPoints);
					//////CUDA_COPY_D2H(faces.data(), cuInstance.d_mesh.faces, sizeof(uint3)* cuInstance.d_mesh.numberOfFaces);
					//////CUDA_SYNC();

					//////for (auto& i : unlinked)
					//////{
					//////	auto& f = faces[i];
					//////	auto& p0 = positions[f.x];
					//////	auto& p1 = positions[f.y];
					//////	auto& p2 = positions[f.z];

					//////	VD::AddTriangle("unlinked", { XYZ(p0) }, { XYZ(p1) }, { XYZ(p2) }, Color::red());

					//////	//printf("[%d] %d, %d, %d\n", i, f.x, f.y, f.z);
					//////	//printf("[%d] %.4f, %.4f, %.4f - %.4f, %.4f, %.4f - %.4f, %.4f, %.4f\n", i, XYZ(p0), XYZ(p1), XYZ(p2));
					//////}

					////////auto result = cuInstance.d_mesh.GetFaceNodePositions();
					////////for (auto& p : result)
					////////{
					////////	VD::AddWiredBox("HashMap", { XYZ(p) }, { 0.0f, 0.1f, 0.0f }, { 0.1f, 0.1f, 0.1f }, Color::red());
					////////}


					vector<float3> pcpositions(cuInstance.d_input.numberOfPoints);
					vector<float3> positions(cuInstance.d_mesh.numberOfPoints);
					vector<uint3> faces(cuInstance.d_mesh.numberOfFaces);
					CUDA_COPY_D2H(pcpositions.data(), cuInstance.d_input.positions, sizeof(float3)* cuInstance.d_input.numberOfPoints);
					CUDA_COPY_D2H(positions.data(), cuInstance.d_mesh.positions, sizeof(float3)* cuInstance.d_mesh.numberOfPoints);
					CUDA_COPY_D2H(faces.data(), cuInstance.d_mesh.faces, sizeof(uint3) * cuInstance.d_mesh.numberOfFaces);
					CUDA_SYNC();
					std::vector<float3> closestPoints;
					//auto indices = cuInstance.d_mesh.FindNearestTriangleIndices(cuInstance.d_input.positions, cuInstance.d_input.numberOfPoints);
					auto indices = cuInstance.d_mesh.FindNearestTriangleIndicesAndClosestPoints(
						cuInstance.d_input.positions, cuInstance.d_input.numberOfPoints, 2, closestPoints);
					printf("indices : %d\n", indices.size());
					for (size_t i = 0; i < cuInstance.d_input.numberOfPoints; i++)
					{
						auto& p = pcpositions[i];

						unsigned int triIdx = indices[i];
						if (triIdx == UINT32_MAX)
						{
							VD::AddBox("pc", { XYZ(p) }, { 0.0f, 0.1f, 0.0f }, glm::vec3(0.05f), Color::red());
							continue;
						}

						auto& f = faces[triIdx];
						auto& p0 = positions[f.x];
						auto& p1 = positions[f.y];
						auto& p2 = positions[f.z];
						auto centroid = (p0 + p1 + p2) / 3.0f;

						auto& closestPoint = closestPoints[i];

						VD::AddBox("pc", { XYZ(p) }, { 0.0f, 0.1f, 0.0f }, glm::vec3(0.005f), Color::white());
						//VD::AddLine("Nearest", { XYZ(p) }, { XYZ(centroid) }, Color::red());
						VD::AddLine("Nearest", { XYZ(p) }, { XYZ(closestPoint) }, Color::yellow());
					}
				}
			}
			else if (GLFW_KEY_END == event.keyCode)
			{
				if (event.action == 1)
				{
					CUDA_TS(BuildOctree);

					CUDA_TS(BuildOctreeInitialize);

					auto center = (cuInstance.h_mesh.min + cuInstance.h_mesh.max) * 0.5f;

					auto lx = cuInstance.h_mesh.max.x - cuInstance.h_mesh.min.x;
					auto ly = cuInstance.h_mesh.max.y - cuInstance.h_mesh.min.y;
					auto lz = cuInstance.h_mesh.max.z - cuInstance.h_mesh.min.z;

					//auto center = (aabbMin + aabbMax) * 0.5f;

					//auto lx = aabbMax.x - aabbMin.x;
					//auto ly = aabbMax.y - aabbMin.y;
					//auto lz = aabbMax.z - aabbMin.z;

					auto maxLength = std::max({ lx, ly, lz });
					float3 bbMin = center - make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);
					float3 bbMax = center + make_float3(maxLength * 0.5f, maxLength * 0.5f, maxLength * 0.5f);

					//float voxelSize = 0.1f;
					float voxelSize = 0.0125f;
					unsigned int maxDepth = 0;
					float unitLength = maxLength;
					while (unitLength > voxelSize)
					{
						unitLength *= 0.5f;
						maxDepth++;
					}

					printf("unitLength : %f\n", unitLength);
					printf("maxDepth : %d\n", maxDepth);

					Octree octree;
					octree.Initialize(
						cuInstance.h_mesh.positions,
						cuInstance.h_mesh.numberOfPoints,
						cuInstance.h_mesh.min,
						cuInstance.h_mesh.max,
						maxDepth);

					//octree.Initialize(
					//	(float3*)ply.GetPoints().data(),
					//	ply.GetPoints().size() / 3,
					//	make_float3(get<0>(ply.GetAABBMin()), get<1>(ply.GetAABBMin()), get<2>(ply.GetAABBMin())),
					//	make_float3(get<0>(ply.GetAABBMax()), get<1>(ply.GetAABBMax()), get<2>(ply.GetAABBMax())),
					//	maxDepth
					//);

					CUDA_TE(BuildOctreeInitialize);

					{
						for (const auto& node : octree.nodes)
						{
							// node.level(=이 노드의 깊이) 기준으로 중심 계산
							const auto p = octree.ToPosition(node.key, bbMin, bbMax);

							// leaf 셀 한 변 길이(예: 전체 길이 / 2^maxDepth)가 unitLength라고 가정
							const unsigned k = (maxDepth >= node.level) ? (maxDepth - node.level) : 0u;
							const float scale = std::ldexp(unitLength, static_cast<int>(k)); // unitLength * 2^k

							stringstream ss;
							ss << "octree_" << node.level;

							//if (k == 1)
							{
								VD::AddWiredBox(
									ss.str(),
									{ XYZ(p) },
									{ 0.0f, 1.0f, 0.0f },
									glm::vec3(scale),            // AddWiredBox가 "전체 길이"를 받는다고 가정
									Color::green()
								);
							}
						}
					}

					for (size_t i = 0; i <= maxDepth; i++)
					{
						stringstream ss;
						ss << "octree_" << i;

						VD::AddToSelectionList(ss.str());
					}

					//{
					//unsigned int numberOfEntries = 0;
					//CUDA_COPY_D2H(&numberOfEntries, octree.mortonCodes.info.numberOfEntries, sizeof(unsigned int));

					//vector<HashMapEntry<uint64_t, unsigned int>> entries(octree.mortonCodes.info.capacity);
					//CUDA_COPY_D2H(entries.data(), octree.mortonCodes.info.entries, sizeof(HashMapEntry<uint64_t, unsigned int>)* octree.mortonCodes.info.capacity);

					//CUDA_SYNC();

					//for (size_t i = 0; i < octree.mortonCodes.info.capacity; i++)
					//{
					//	auto& entry = entries[i];
					//	if (UINT64_MAX == entry.key) continue;
					//	//printf("MortonCode: %llu, Count: %d\n", entry.key, entry.value);

					//	Octree::PointFromCode_Voxel(entry.key, center, voxelSize, Octree::GridOffset());
					//	auto depth = Octree::UnpackDepth(entry.key);
					//	auto p = Octree::PointFromCode_Voxel(entry.key, center, 0.1f, Octree::GridOffset());
					//	//printf("%f, %f, %f\n", XYZ(p));
					//	if (depth == 8)
					//	{
					//		printf("%f, %f, %f\n", XYZ(p));
					//		VD::AddWiredBox("octree", { XYZ(p) }, { 0.0f, 1.0f, 0.0f }, glm::vec3(1.0f), Color::red());
					//	}
					//	else
					//	{
					//		VD::AddWiredBox("octree", { XYZ(p) }, { 0.0f, 1.0f, 0.0f }, glm::vec3(0.1f * (float)maxDepth / (float)depth), Color::green());
					//	}
					//}
					//}

					//{
					//	vector<OctreeNode> octreeNodes(octree.numberOfNodes);
					//	CUDA_COPY_D2H(octreeNodes.data(), octree.nodes, sizeof(OctreeNode) * octree.numberOfNodes);

					//	map<uint64_t, int> temp;

					//	for (auto& n : octreeNodes)
					//	{
					//		//printf("MortonCode: %llu, Level: %d\n", n.mortonCode, n.level);

					//		temp[n.mortonCode]++;

					//		auto p = Octree::PointFromCode_Voxel(n.mortonCode, center, 0.1f, Octree::GridOffset());
					//		//printf("%f, %f, %f\n", XYZ(p));
					//		VD::AddWiredBox("octree", { XYZ(p) }, { 0.0f, 1.0f, 0.0f }, glm::vec3(0.1f), Color::green());
					//	}

					//	printf("temp.size() : %d\n", temp.size());
					//}

					octree.Terminate();

					CUDA_TE(BuildOctree);
				}
			}
			else if (GLFW_KEY_UP == event.keyCode)
			{
				if (event.action == 1)
				{
					VD::ShowNextSelection();
				}
			}
			else if (GLFW_KEY_DOWN == event.keyCode)
			{
				if (event.action == 1)
				{
					VD::ShowPreviousSelection();
				}
			}
			else if (GLFW_KEY_0 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_10");
			}
			else if (GLFW_KEY_1 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_1");
			}
			else if (GLFW_KEY_2 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_2");
			}
			else if (GLFW_KEY_3 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_3");
			}
			else if (GLFW_KEY_4 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_4");
			}
			else if (GLFW_KEY_5 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_5");
			}
			else if (GLFW_KEY_6 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_6");
			}
			else if (GLFW_KEY_7 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_7");
			}
			else if (GLFW_KEY_8 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_8");
			}
			else if (GLFW_KEY_9 == event.keyCode)
			{
			if (event.action == 1) VD::ToggleVisibility("octree_9");
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
					cuInstance.d_mesh.BuildFaceNodeHashMap();

					//auto curvatures = cuInstance.d_mesh.GetFaceCurvatures();
					//auto curvatures = cuInstance.d_mesh.GetFaceCurvaturesInRadius(1.0f);

					//printf("curvatures.size() : %d\n", curvatures.size());

					//printf("cuInstance.d_mesh.numberOfFaces : %d\n", cuInstance.d_mesh.numberOfFaces);

					//float minc = 1e10, maxc = -1e10;
					//for (float c : curvatures) {
					//	minc = std::min(minc, c);
					//	maxc = std::max(maxc, c);
					//}
					//printf("Curvature: min=%f, max=%f\n", minc, maxc);

					//float sum = 0.0f, sq_sum = 0.0f;
					//for (float c : curvatures)
					//{
					//	sum += c;
					//	sq_sum += c * c;
					//}
					//float mean = sum / curvatures.size();
					//float variance = (sq_sum / curvatures.size()) - (mean * mean);
					//float stddev = sqrtf(variance);

					//vector<float3> positions(cuInstance.d_mesh.numberOfPoints);
					//vector<uint3> faces(cuInstance.d_mesh.numberOfFaces);
					//CUDA_COPY_D2H(positions.data(), cuInstance.d_mesh.positions, sizeof(float3) * cuInstance.d_mesh.numberOfPoints);
					//CUDA_COPY_D2H(faces.data(), cuInstance.d_mesh.faces, sizeof(uint3) * cuInstance.d_mesh.numberOfFaces);
					//CUDA_SYNC();

					//for (size_t i = 0; i < curvatures.size(); i++)
					//{
					//	float c = curvatures[i];
					//	float z = (c - mean) / (stddev + 1e-8f); // Z-score
					//	float ratio = (z + 2.0f) * 0.25f; // z=-2 -> 0, z=+2 -> 1
					//	ratio = std::clamp(ratio, 0.0f, 1.0f);
					//	auto color = Color::Lerp(Color::blue(), Color::red(), ratio);

					//	auto& f = faces[i];
					//	auto& p0 = positions[f.x];
					//	auto& p1 = positions[f.y];
					//	auto& p2 = positions[f.z];

					//	VD::AddTriangle("Curvatures", { XYZ(p0) }, { XYZ(p1) }, { XYZ(p2) }, color);
					//}
				}
			}
			else if (GLFW_KEY_PAGE_DOWN == event.keyCode)
			{
				cuInstance.h_mesh.CopyFromDevice(cuInstance.d_mesh);
				cuInstance.h_mesh.SerializePLY("../../res/3D/MarchingCubes.ply");
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

				/* Host
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
				}
				*/

				if (cuInstance.d_mesh.PickFace(
					make_float3(ray.origin.x, ray.origin.y, ray.origin.z),
					make_float3(ray.direction.x, ray.direction.y, ray.direction.z),
					hitIndex, outHit))
				{
					vector<float3> positions(cuInstance.d_mesh.numberOfPoints);
					vector<uint3> faces(cuInstance.d_mesh.numberOfFaces);
					CUDA_COPY_D2H(positions.data(), cuInstance.d_mesh.positions, sizeof(float3) * cuInstance.d_mesh.numberOfPoints);
					CUDA_COPY_D2H(faces.data(), cuInstance.d_mesh.faces, sizeof(uint3) * cuInstance.d_mesh.numberOfFaces);
					CUDA_SYNC();

					auto& f = faces[hitIndex];
					auto& p0 = positions[f.x];
					auto& p1 = positions[f.y];
					auto& p2 = positions[f.z];

					auto v0 = glm::vec3(XYZ(p0));
					auto v1 = glm::vec3(XYZ(p1));
					auto v2 = glm::vec3(XYZ(p2));

					auto normal = glm::trianglenormal(v0, v1, v2);

					//VD::AddTriangle("Picked", v0, v1, v2, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });
					//VD::AddSphere("PickedS", v0, normal, 0.025f, { 1.0f, 0.0f, 0.0f, 1.0f });
					//VD::AddSphere("PickedS", v1, normal, 0.025f, { 0.0f, 1.0f, 0.0f, 1.0f });
					//VD::AddSphere("PickedS", v2, normal, 0.025f, { 0.0f, 0.0f, 1.0f, 1.0f });
					auto hitPosition = ray.origin + glm::normalize(ray.direction) * outHit;
					//VD::AddSphere("PickedPosition", hitPosition, normal, 0.025f, { 0.0f, 0.0f, 1.0f, 1.0f });

					//VD::AddLine("PickingRay", ray.origin, hitPosition, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f });

					if (GLFW_MOD_CONTROL == event.mods)
					{
						camera->SetTarget(hitPosition);
						camera->SetEye(hitPosition + (-ray.direction) * manipulator->GetRadius());
					}

#pragma region Triangle Points
					//auto d0 = glm::distance2(hitPosition, v0);
					//auto d1 = glm::distance2(hitPosition, v1);
					//auto d2 = glm::distance2(hitPosition, v2);
					//unsigned int vi = -1;
					//glm::vec3 nearest = v0;
					//glm::vec3 nearestNormal = v0;
					//if (d0 < d1 && d0 < d2)
					//{
					//	nearest = v0;
					//	nearestNormal = glm::vec3(XYZ(cuInstance.h_mesh.normals[f.x]));
					//	vi = f.x;
					//}
					//else if (d1 < d0 && d1 < d2)
					//{
					//	nearest = v1;
					//	nearestNormal = glm::vec3(XYZ(cuInstance.h_mesh.normals[f.y]));
					//	vi = f.y;
					//}
					//else if (d2 < d0 && d2 < d1)
					//{
					//	nearest = v2;
					//	nearestNormal = glm::vec3(XYZ(cuInstance.h_mesh.normals[f.z]));
					//	vi = f.z;
					//}

					//VD::AddWiredBox("Nearest", nearest, nearestNormal, { 0.05f, 0.05, 0.05 }, { 1.0f, 1.0f, 1.0f, 1.0f });
#pragma endregion

					//printf("[d_mesh] hitIndex : %d, outHit : %f\n", hitIndex, outHit);

					//printf("OneRing Center : %d\n", vi);

					//float radius = 1.0f;

					////auto vis = cudaInstance.d_mesh.GetOneRingVertices(vi);
					//auto vis = cuInstance.d_mesh.GetVerticesInRadius(vi, radius);
					//for (auto& vi : vis)
					//{
					//	printf("%d, ", vi);
					//	glm::vec3 position = glm::vec3(XYZ(cuInstance.h_mesh.positions[vi]));
					//	VD::AddWiredBox("OneRing", position, normal, { 0.025f, 0.025, 0.025 }, Color::green());

					//	stringstream ss;
					//	ss << vi;
					//	//VD::AddText("OneRingVertices", ss.str(), position, Color::white());
					//}
					//printf("\n");
					//VD::AddSphere("Range", nearest, nearestNormal, radius, glm::vec4(1.0f, 1.0f, 1.0f, 0.3f));

					////cudaInstance.d_mesh.RadiusLaplacianSmoothing(radius, 1);
					//cuInstance.interop.UploadFromDevice(cuInstance.d_mesh);

					cuInstance.d_mesh.BuildFaceNodeHashMap();
					auto oneRingFaces = cuInstance.d_mesh.GetOneRingFaces(hitIndex);

					for (auto& faceIndex : oneRingFaces)
					{
						auto& face = faces[faceIndex];
						auto& p0 = positions[face.x];
						auto& p1 = positions[face.y];
						auto& p2 = positions[face.z];

						auto v0 = glm::vec3(XYZ(p0));
						auto v1 = glm::vec3(XYZ(p1));
						auto v2 = glm::vec3(XYZ(p2));

						//VD::AddTriangle("One Ring Faces", v0, v1, v2, Color::red());
					}

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

			auto result = cuInstance.ProcessPointCloud(h_pointCloud);
			//ApplyPointCloudToEntity(entity, result);
			//cuInstance.d_input = result;

			auto meshEntity = Feather.CreateEntity("MarchingCubesMesh");
			//ApplyHalfEdgeMeshToEntity(meshEntity, cudaInstance.h_mesh);
			auto meshRenderable = Feather.CreateComponent<Renderable>(meshEntity);
			meshRenderable->Initialize(Renderable::GeometryMode::Triangles);
			cuInstance.interop.Initialize(meshRenderable);
			meshRenderable->AddShader(Feather.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
			meshRenderable->AddShader(Feather.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
			meshRenderable->AddShader(Feather.CreateShader("Flat", File("../../res/Shaders/Flat.vs"), File("../../res/Shaders/Flat.gs"), File("../../res/Shaders/Flat.fs")));
			meshRenderable->SetActiveShaderIndex(0);
			cuInstance.interop.UploadFromDevice(cuInstance.d_mesh);

			printf("meshEntity : %d\n", meshEntity);

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

			result.Terminate();
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

			//auto width = w->GetWidth();
			//auto height = w->GetHeight();

			//Entity cam = Feather.GetEntityByName("Camera");
			//auto pcam = Feather.GetComponent<PerspectiveCamera>(cam);

			//auto projection = pcam->GetProjectionMatrix();
			//auto view = pcam->GetViewMatrix();

			VD::AddWiredBox("AABB", { { x, y, z }, { X, Y, Z } }, Color::blue());
		}
		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
