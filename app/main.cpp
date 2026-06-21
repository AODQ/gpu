#include "shaders/scene_shared.h"
#include "shaders/resolve_pc.h"
#include "asset_library.hpp"
#include "scene_desc.hpp"

#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <srat/camera.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <string>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>

static float sScrollDelta = 0.0f;
static int sDebugMode = 0;

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

static void tick(char const * label, Clock::time_point & t) {
	printf("  [%.1f ms] %s\n", Ms(Clock::now() - t).count(), label);
	t = Clock::now();
}

static constexpr f32 kDegToRad = 0.017453292519943295f;

struct LoadedModel
{
	mor::Scene scene;
	mor::GpuScene gpuScene;
	u32 modelId;
};

static f32m44 make_model_matrix(SceneInstance const & inst)
{
	return (
		f32m44_translate(inst.position.x, inst.position.y, inst.position.z)
		* f32m44_rotate_y(inst.rotation.y * kDegToRad)
		* f32m44_rotate_x(inst.rotation.x * kDegToRad)
		* f32m44_rotate_z(inst.rotation.z * kDegToRad)
		* f32m44_scale(inst.scale, inst.scale, inst.scale)
	);
}

int32_t main(int32_t const argc, char const * const * const argv) {
	char const * gltfPath = nullptr;
	char const * scenePath = nullptr;
	char const * screenshotPath = nullptr;
	for (i32 i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc) {
			screenshotPath = argv[++i];
		} else {
			std::filesystem::path const p(argv[i]);
			if (p.extension() == ".json") {
				scenePath = argv[i];
			} else {
				gltfPath = argv[i];
			}
		}
	}
	bool const headless = screenshotPath != nullptr;

	if (headless && !gltfPath) {
		printf("usage (headless): cull <path.gltf> --screenshot <out.png>\n");
		return 1;
	}

	static constexpr u32 kScreenW = 1280u;
	static constexpr u32 kScreenH = 720u;

	Clock::time_point t = Clock::now();

	if (headless) {
		vkof::init_headless(kScreenW, kScreenH);
	} else {
		vkof::init();
	}
	if (headless) { tick("vk init", t); }

	std::filesystem::path const appDir = (
		std::filesystem::canonical(
			std::filesystem::path(__FILE__).parent_path()
		)
	);
	std::filesystem::path const shaderDir = appDir / "shaders";
	std::filesystem::path const libDir = appDir.parent_path() / "lib";
	std::filesystem::path const assetsDir = appDir.parent_path() / "assets";
	std::filesystem::path const repoDir = appDir.parent_path();

	AssetLibrary const assetLib = asset_library_create(assetsDir);

	std::filesystem::path sceneFilePath = (
		scenePath ? std::filesystem::path(scenePath) : std::filesystem::path{}
	);
	SceneDesc sceneDesc = scenePath
		? scene_desc_load(sceneFilePath)
		: SceneDesc{};

	std::string const meshPath = (shaderDir / "scene.mesh").string();
	std::string const fragPath = (shaderDir / "scene.frag").string();
	std::string const libDirStr = libDir.string();

	vkof::TransientImage const visibilityTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r32ui,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1,
		.isDoubleBuffered = !headless,
	});
	vkof::TransientImage const colorTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1,
		.isDoubleBuffered = !headless,
	});
	vkof::TransientImage const depthTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::d24_unorm_s8_uint,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1,
		.isDoubleBuffered = !headless,
	});

	char const * const kIncludePaths[] = { libDirStr.c_str() };
	static constexpr vkof::ImageFormat kColorFmts[] = {
		vkof::ImageFormat::r32ui,
	};
	vkof::Pipeline const visibilityPipeline = (
		vkof::pipeline_graphics_create({
			.pathMesh = meshPath.c_str(),
			.pathFragment = fragPath.c_str(),
			.attachmentColorFormats = (
				srat::slice<vkof::ImageFormat const>(kColorFmts, 1u)
			),
			.attachmentDepthStencilFormat = vkof::ImageFormat::d24_unorm_s8_uint,
			.depthTest = vkof::DepthTest::write_on_test_on,
			.cullMode = vkof::CullMode::back,
			.blendMode = vkof::BlendMode::none,
			.includePaths = srat::slice { kIncludePaths, 1u },
		})
	);
	vkof::Pipeline const resolvePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (shaderDir / "resolve.comp").string().c_str(),
			.includePaths = srat::slice { kIncludePaths, 1u },
		})
	);
	if (headless) { tick("pipeline (shader compile)", t); }

	srat::CameraOrbit cam = {
		.target = { 0.0f, 0.0f, 0.0f },
		.distance = 5.0f,
		.azimuth = 0.0f,
		.elevation = 0.3f,
		.fovY = 2.0f,
		.aspect = 1.0f,
		.near = 0.01f,
		.far = 1000.0f,
	};

	bool const singleModelMode = (gltfPath != nullptr && scenePath == nullptr);
	if (singleModelMode) {
		scene_desc_add_instance(
			sceneDesc,
			std::filesystem::canonical(gltfPath).string()
		);
	}

	std::unordered_map<std::string, size_t> loadedModels;
	std::vector<LoadedModel> modelList;

	static constexpr u32 kMaxModels = 256u;
	vkof::Buffer const modelsIndirectBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect) * kMaxModels,
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	auto const fnUploadModelsIndirectBuffer = [&]() {
		std::vector<GpuResolveModelIndirect> descs(modelList.size());
		for (size_t i = 0u; i < modelList.size(); ++i) {
			mor::Buffers const bufs = mor::scene_gpu_buffers(modelList[i].gpuScene);
			descs[i] = GpuResolveModelIndirect {
				.meshlets = bufs.meshlets,
				.materials = bufs.materials,
			};
		}
		if (descs.empty()) { return; }
		vkof::buffer_upload({
			.buffer = modelsIndirectBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(descs.data()),
				descs.size() * sizeof(GpuResolveModelIndirect)
			),
		});
	};

	auto const fnVerifyModelLoaded = [&](std::string const & filename) {
		if (loadedModels.count(filename)) { return; }
		std::filesystem::path p(filename);
		if (p.is_relative()) { p = repoDir / p; }
		if (!std::filesystem::exists(p)) {
			printf("[warn] model not found: %s\n", p.c_str());
			return;
		}
		mor::Scene const scene = mor::scene_create();
		mor::scene_load_gltf(scene, p.c_str());
		mor::GpuScene const sceneGpu = mor::scene_gpu_upload(scene);
		u32 const modelId = modelList.size();
		loadedModels[filename] = modelId;
		modelList.push_back({
			.scene = scene,
			.gpuScene = sceneGpu,
			.modelId = modelId,
		});
		fnUploadModelsIndirectBuffer();
	};

	for (SceneInstance const & inst : sceneDesc.instances) {
		fnVerifyModelLoaded(inst.filename);
	}

	GLFWwindow * const window = vkof::window();

	glfwSetScrollCallback(window, [](GLFWwindow * w, double xoff, double yoff) {
		ImGui_ImplGlfw_ScrollCallback(w, xoff, yoff);
		if (!ImGui::GetIO().WantCaptureMouse) {
			sScrollDelta += (float)yoff;
		}
	});

	double prevMouseX = 0.0;
	double prevMouseY = 0.0;
	glfwGetCursorPos(window, &prevMouseX, &prevMouseY);

	while (!glfwWindowShouldClose(window)) {
		if (!glfwGetWindowAttrib(window, GLFW_FOCUSED)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}

		double curMouseX, curMouseY;
		glfwGetCursorPos(window, &curMouseX, &curMouseY);
		f32 const dx = (f32)(curMouseX - prevMouseX);
		f32 const dy = (f32)(curMouseY - prevMouseY);
		prevMouseX = curMouseX;
		prevMouseY = curMouseY;

		if (!ImGui::GetIO().WantCaptureMouse) {
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
				srat::camera_orbit_rotate(cam, -dx * 0.005f, dy * 0.005f);
			}
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
				srat::camera_orbit_pan(
					cam,
					dx * 0.01f * cam.distance,
					-dy * 0.01f * cam.distance
				);
			}
			srat::camera_orbit_zoom(cam, -sScrollDelta * 0.1f * cam.distance);
		}
		sScrollDelta = 0.0f;

		int winW, winH;
		glfwGetWindowSize(window, &winW, &winH);
		cam.aspect = (f32)winW / (f32)winH;

		for (SceneInstance const & inst : sceneDesc.instances) {
			fnVerifyModelLoaded(inst.filename);
		}

		bool const rightClick = (
			!ImGui::GetIO().WantCaptureMouse
			&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
		);
		GpuGlobalPC const globalPC {
			.time = (f32)glfwGetTime(),
			.probeX = (i32)curMouseX,
			.probeY = (i32)curMouseY,
			.probeActive = rightClick ? 1u : 0u,
			.debugMode = (u32)sDebugMode,
			.cameraPos = srat::camera_orbit_eye(cam),
			._pad = {},
			.viewProj = srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam),
		};

		vkof::imgui_begin();

		if (char const * uri = asset_library_imgui(assetLib)) {
			std::string const relPath = std::filesystem::relative(
				std::filesystem::path(uri), repoDir
			).string();
			scene_desc_add_instance(sceneDesc, relPath);
			fnVerifyModelLoaded(relPath);
		}

		if (!singleModelMode) {
			i32 const focusIdx = scene_desc_imgui(sceneDesc, sceneFilePath);
			if (focusIdx >= 0) {
				cam.target = sceneDesc.instances[focusIdx].position;
			}
		}

		ImGui::Begin("scene");
		ImGui::Text("fps: %.1f", ImGui::GetIO().Framerate);
		ImGui::Separator();
		float fovDeg = cam.fovY * (180.0f / 3.14159265f);
		if (ImGui::SliderFloat("fov", &fovDeg, 10.0f, 120.0f, "%.0f deg")) {
			cam.fovY = fovDeg * (3.14159265f / 180.0f);
		}
		if (ImGui::Button("reset camera")) {
			cam.target = { 0.0f, 0.0f, 0.0f };
			cam.distance = 5.0f;
			cam.azimuth = 0.0f;
			cam.elevation = 0.3f;
		}
		// these match MOR_DEBUG_VIEW* in mor_shared.h
		static constexpr char const * kDebugModes[] = {
			"none",
			"meshlet index",
			"model index",
			"triangle index",
		};
		ImGui::Combo(
			"debug", &sDebugMode, kDebugModes, IM_ARRAYSIZE(kDebugModes)
		);
		ImGui::End();

		if (char const * msg = vkof::probe_message()) {
			ImGui::BeginTooltip();
			ImGui::Text("%s", msg);
			ImGui::EndTooltip();
		}

		{
			vkof::RenderNode const drawNode = vkof::render_node_create({
				.queue = vkof::CommandQueue::graphics,
			});
			static constexpr f32 kVisibilityClearColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
			static constexpr f32 kDepthClear = 1.0f;
			vkof::render_node_attachment_color({
				.node = drawNode,
				.image = visibilityTarget,
				.loadOp = vkof::RenderNodeLoadOp::clear,
				.mipLevel = 0,
				.colorIndex = 0,
				.clearColor = srat::slice<f32 const>(kVisibilityClearColor, 4),
			});
			vkof::render_node_attachment_depth({
				.node = drawNode,
				.image = depthTarget,
				.loadOp = vkof::RenderNodeLoadOp::clear,
				.mipLevel = 0,
				.clearDepth = srat::slice<f32 const>(&kDepthClear, 1),
			});
			vkof::render_node_callback({
				.node = drawNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					for (SceneInstance const & inst : sceneDesc.instances) {
						auto const it = loadedModels.find(inst.filename);
						if (it == loadedModels.end()) { continue; }
						LoadedModel const & model = modelList[it->second];
						mor::Buffers const bufs = (
							mor::scene_gpu_buffers(model.gpuScene)
						);
						u32 const count = (
							mor::scene_gpu_meshlet_count(model.gpuScene)
						);
						GpuSceneDrawPC const drawPC {
							.modelId = model.modelId,
							.meshlets = bufs.meshlets,
							.positions = bufs.positions,
							.instances = bufs.instances,
							.meshletVerts = bufs.meshletVerts,
							.meshletTris = bufs.meshletTris,
							.modelMatrix = make_model_matrix(inst),
						};
						vkof::cmd_draw({
							.cmd = cmd,
							.pipeline = visibilityPipeline,
							.pushconstant = srat::slice<u8 const>(
								reinterpret_cast<u8 const *>(&drawPC), sizeof(drawPC)
							),
							.vertexCount = count,
							.instanceCount = 1,
						});
					}
				},
			});

			u32 const visibilityHandle = (
				vkof::transient_image_storage_handle({
					.image = visibilityTarget,
					.mipLevel = 0,
				})
			);
			u32 const colorHandle = (
				vkof::transient_image_storage_handle({
					.image = colorTarget,
					.mipLevel = 0,
				})
			);

			// TODO try compute queue (will require synchronization)
			vkof::RenderNode const resolveNode = vkof::render_node_create({
				.queue = vkof::CommandQueue::graphics,
			});
			vkof::render_node_add_image({
				.node = resolveNode,
				.image = visibilityTarget,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_add_image({
				.node = resolveNode,
				.image = colorTarget,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_callback({
				.node = resolveNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					GpuResolvePC const resolvePC {
						.visibilityImageHandle = visibilityHandle,
						.outputImageHandle = colorHandle,
						.models = vkof::buffer_virtual_address(modelsIndirectBuffer),
					};
					ImGui::Begin("Resolve PC");
					ImGui::Text("vis handle: %u", visibilityHandle);
					ImGui::Text("color handle: %u", colorHandle);
					ImGui::Text("models buffer: %lu", resolvePC.models);
					ImGui::End();
					vkof::cmd_dispatch(vkof::CmdDispatch {
						.cmd = cmd,
						.pipeline = resolvePipeline,
						.pushconstant = srat::slice<u8 const>(
							reinterpret_cast<u8 const *>(&resolvePC), sizeof(resolvePC)
						),
						.groupCountX = (kScreenW + 15) / 16,
						.groupCountY = (kScreenH + 15) / 16,
						.groupCountZ = 1,
					});
				}
			});

			vkof::RenderNode const nodes[] = { drawNode, resolveNode, };
			vkof::render_graph_execute({
				.nodes = srat::slice<vkof::RenderNode const>(nodes, 2u),
				.rootPushconstant = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(&globalPC), sizeof(globalPC)
				),
				.finalImage = colorTarget,
			});
			vkof::render_node_destroy(drawNode);
			vkof::render_node_destroy(resolveNode);
		}
	}

	vkof::device_wait_idle();
	for (auto & [filename, modelIndex] : loadedModels) {
		LoadedModel const & model = modelList[modelIndex];
		mor::scene_destroy(model.scene);
		mor::scene_gpu_destroy(model.gpuScene);
	}
	loadedModels.clear();
	modelList.clear();
	vkof::buffer_destroy(modelsIndirectBuffer);
	vkof::pipeline_destroy(visibilityPipeline);
	vkof::pipeline_destroy(resolvePipeline);
	mor::sampler_cache_destroy();
	vkof::shutdown();
	return 0;
}
