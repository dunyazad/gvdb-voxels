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

using namespace libRxTx;

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

// --- KD-Tree Implementation Start ---
struct KDPoint {
	glm::vec3 position;
	size_t originalIndex;
};

struct KDNode {
	KDPoint point;
	KDNode* left = nullptr;
	KDNode* right = nullptr;
	int axis;

	KDNode(const KDPoint& pt, int ax) : point(pt), axis(ax), left(nullptr), right(nullptr) {}
};

class SimpleKDTree {
public:
	SimpleKDTree() : root(nullptr) {}
	~SimpleKDTree() { Clear(root); }

	void Build(const std::vector<glm::vec3>& rawPoints) {
		Clear(root);
		points.clear();
		points.reserve(rawPoints.size());

		for (size_t i = 0; i < rawPoints.size(); ++i) {
			points.push_back({ rawPoints[i], i });
		}

		root = BuildRecursive(points.begin(), points.end(), 0);
		printf("[KDTree] Built with %zu points.\n", points.size());
	}

	void Search(const glm::vec3& query, int k, std::vector<size_t>& outIndices, std::vector<float>& outDistSq) {
		if (!root || k <= 0) return;

		// Max-heap to store the k nearest neighbors found so far.
		// Pair: <Squared Distance, Original Index>
		std::priority_queue<std::pair<float, size_t>> pq;

		SearchRecursive(root, query, k, pq);

		// Extract results from PQ
		outIndices.resize(pq.size());
		outDistSq.resize(pq.size());

		// PQ pops the largest element first, so we fill from back to front
		for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
			outIndices[i] = pq.top().second;
			outDistSq[i] = pq.top().first;
			pq.pop();
		}
	}

	void SearchRadius(const glm::vec3& query, float radius, std::vector<size_t>& outIndices) {
		if (!root || radius <= 0.0f) return;

		outIndices.clear();
		float radiusSq = radius * radius;
		SearchRadiusRecursive(root, query, radiusSq, outIndices);
	}

private:
	KDNode* root;
	std::vector<KDPoint> points;

	void Clear(KDNode* node) {
		if (!node) return;
		Clear(node->left);
		Clear(node->right);
		delete node;
	}

	KDNode* BuildRecursive(std::vector<KDPoint>::iterator start, std::vector<KDPoint>::iterator end, int depth) {
		if (start >= end) return nullptr;

		int axis = depth % 3;
		size_t len = std::distance(start, end);
		auto mid = start + len / 2;

		std::nth_element(start, mid, end, [axis](const KDPoint& a, const KDPoint& b) {
			return a.position[axis] < b.position[axis];
			});

		KDNode* node = new KDNode(*mid, axis);
		node->left = BuildRecursive(start, mid, depth + 1);
		node->right = BuildRecursive(mid + 1, end, depth + 1);

		return node;
	}

	void SearchRecursive(KDNode* node, const glm::vec3& query, int k, std::priority_queue<std::pair<float, size_t>>& pq) {
		if (!node) return;

		// Calculate squared distance between query and current node
		float dx = query.x - node->point.position.x;
		float dy = query.y - node->point.position.y;
		float dz = query.z - node->point.position.z;
		float distSq = dx * dx + dy * dy + dz * dz;

		// Update priority queue
		if (pq.size() < k) {
			pq.push({ distSq, node->point.originalIndex });
		}
		else if (distSq < pq.top().first) {
			pq.pop();
			pq.push({ distSq, node->point.originalIndex });
		}

		// Determine which side to traverse first
		float diff = query[node->axis] - node->point.position[node->axis];
		KDNode* nearNode = diff < 0 ? node->left : node->right;
		KDNode* farNode = diff < 0 ? node->right : node->left;

		// Visit near side
		SearchRecursive(nearNode, query, k, pq);

		// Visit far side only if necessary
		// If we haven't found k items yet, OR if the plane distance is within our current search radius
		if (pq.size() < k || (diff * diff) < pq.top().first) {
			SearchRecursive(farNode, query, k, pq);
		}
	}

	void SearchRadiusRecursive(KDNode* node, const glm::vec3& query, float radiusSq, std::vector<size_t>& outIndices) {
		if (!node) return;

		// Calculate squared distance
		float dx = query.x - node->point.position.x;
		float dy = query.y - node->point.position.y;
		float dz = query.z - node->point.position.z;
		float distSq = dx * dx + dy * dy + dz * dz;

		// Collect point if within radius
		if (distSq <= radiusSq) {
			outIndices.push_back(node->point.originalIndex);
		}

		// Determine traversal order
		float diff = query[node->axis] - node->point.position[node->axis];
		float diffSq = diff * diff;

		KDNode* nearNode = diff < 0 ? node->left : node->right;
		KDNode* farNode = diff < 0 ? node->right : node->left;

		// Always search the near side
		SearchRadiusRecursive(nearNode, query, radiusSq, outIndices);

		// Search the far side only if the plane intersects the search radius
		if (diffSq <= radiusSq) {
			SearchRadiusRecursive(farNode, query, radiusSq, outIndices);
		}
	}
};
// --- KD-Tree Implementation End ---

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

	//RxTx udp(Protocol::UDP);
	//udp.Init();
	//udp.SetMode(UdpMode::Broadcast);
	//udp.Bind(5000); // 자동 NIC 선택
	//udp.OnReceive([](const std::string& msg, const std::string& ip) {
	//	std::cout << "[UDP Received] " << msg << std::endl;
	//	});
	//udp.Start();
	
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
	}
#pragma endregion

	Feather.AddOnInitializeCallback([&]() {

		CUDAMain();
		
#if 0
		{
			PLYFormat ply;
			ply.Deserialize("D:\\Debug\\PLY\\inputA.ply");
			auto [minx, miny, minz] = ply.GetAABBMin();

			std::vector<glm::vec3> pointCloud;
			size_t pointCount = ply.GetPoints().size() / 3;
			pointCloud.reserve(pointCount);

			for (size_t i = 0; i < pointCount; ++i)
			{
				float x = ply.GetPoints()[i * 3 + 0];
				float y = ply.GetPoints()[i * 3 + 1];
				float z = ply.GetPoints()[i * 3 + 2];

				if (-10.0f < x && x < 10.0f &&
					-10.0f < y && y < 10.0f &&
					-10.0f < z && z < 10.0f)
				{
					pointCloud.emplace_back(x, y, z);
				}
			}

			TS(build_KDTree);
			SimpleKDTree kdtree;
			kdtree.Build(pointCloud);
			TE(build_KDTree);

			std::vector<size_t> numberOfNeighbors(pointCount);
			std::vector<size_t> borderPoints(pointCount);
			TS(KNN);
			for (auto& p : pointCloud)
			{
				std::vector<size_t> indices;
				kdtree.SearchRadius(p, 0.2f, indices);

				numberOfNeighbors.push_back(indices.size());

				if (25 > indices.size())
				{
					borderPoints.push_back(1);
				}
				else
				{
					borderPoints.push_back(0);
				}
			}
			TE(KNN);

			for (auto& p : pointCloud)
			{
				std::vector<size_t> indices;
				kdtree.SearchRadius(p, 0.3f, indices);

				numberOfNeighbors.push_back(indices.size());

				size_t threshold = 25;
				if (threshold > indices.size())
				{
					VD::AddSphere("points", p, 0.05f, Color::red());
				}
				else
				{
					VD::AddSphere("points", p, 0.05f, Color::white());
				}
			}
		}
#endif // 0


		});

	Feather.AddOnUpdateCallback([&](f32 timeDelta) {
		});

	Feather.Run();

	Feather.Terminate();

	return 0;
}
