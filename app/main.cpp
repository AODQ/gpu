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
static bool sMipOverrideActive = false;
static float sMipLodOverride = 0.0f;
static float sMipLodBias = 0.0f;
static float sAnisotropy = 16.0f;
static float sAnisotropyPending = 16.0f;
static float sExposure = 1.0f;
static bool sFpMode = false;
static float sFlyCamSpeed = 0.05f;
static u32 sSelectedObject = 0xFFFFFFFFu;
static std::vector<u32> sDrawToInstIdx;

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
	vkof::AccelerationStructureBlas blas;
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
	{ tick("vk init", t); }

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
	vkof::Pipeline const debugProbePipeline = (
		vkof::debug_sphere_pipeline_create({
			.pathFrag = (
				(shaderDir / "ddgi_probe_vis.frag").string().c_str()
			)
		})
	);
	{ tick("pipeline (shader compile)", t); }

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
	srat::CameraFirstPerson fpCam = {
		.position = { 0.0f, 1.0f, 5.0f },
		.yaw = 0.0f,
		.pitch = 0.0f,
		.fovY = cam.fovY,
		.aspect = cam.aspect,
		.near = cam.near,
		.far = cam.far,
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
	vkof::Buffer const debugPcBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuDebugPC),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::AccelerationStructureTlas const tlas = (
		vkof::tlas_create({ .maxInstances = kMaxModels })
	);

	static constexpr u32 kMaxLights = 64u;
	vkof::Buffer const lightsBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuLight) * kMaxLights,
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	auto const fnVerifyModelLoaded = [&](std::string const & filename) {
		if (loadedModels.count(filename)) { return; }
		std::filesystem::path p(filename);
		if (p.is_relative()) { p = repoDir / p; }
		if (!std::filesystem::exists(p)) {
			printf("[warn] model not found: %s\n", p.c_str());
			return;
		}
		mor::Scene const scene = mor::scene_create();
		auto const tLoad = Clock::now();
		mor::scene_load_gltf(scene, p.c_str());
		auto const tUpload = Clock::now();
		mor::GpuScene const sceneGpu = mor::scene_gpu_upload(scene);
		auto const tBlas = Clock::now();
		mor::Buffers const bufsForBlas = mor::scene_gpu_buffers(sceneGpu);
		vkof::AccelerationStructureBlas const blas = (
			vkof::blas_create({
				.positionVa = bufsForBlas.positions,
				.vertexCount = bufsForBlas.vertexCount,
				.indexVa = bufsForBlas.flatIndices,
				.triangleCount = bufsForBlas.triangleCount,
			})
		);
		auto const tDone = Clock::now();
		printf(
			"[model] %s: load %.1fms, upload %.1fms, blas %.1fms\n",
			p.filename().c_str(),
			Ms(tUpload - tLoad).count(),
			Ms(tBlas - tUpload).count(),
			Ms(tDone - tBlas).count()
		);
		u32 const modelId = modelList.size();
		loadedModels[filename] = modelId;
		modelList.emplace_back(LoadedModel {
			.scene = scene,
			.gpuScene = sceneGpu,
			.modelId = modelId,
			.blas = blas,
		});
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
	bool prevFpLookActive = false;

	while (!glfwWindowShouldClose(window)) {
		static Clock::time_point sPrevFrameTime = Clock::now();
		f32 const dt = std::min(
			(f32)(Ms(Clock::now() - sPrevFrameTime).count() * 0.001f),
			0.05f
		);
		sPrevFrameTime = Clock::now();

		if (!glfwGetWindowAttrib(window, GLFW_FOCUSED)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		glfwPollEvents();

		{
			static bool sPrevEsc = false;
			bool const currEsc = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
			if (currEsc && !sPrevEsc) {
				if (sFpMode) {
					sFpMode = false;
					cam.azimuth = -fpCam.yaw;
					cam.elevation = -fpCam.pitch;
					cam.target = (
						fpCam.position
						+ srat::camera_fp_forward(fpCam) * cam.distance
					);
				} else {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
				}
			}
			sPrevEsc = currEsc;
		}
		{
			static bool sPrevTab = false;
			bool const currTab = (
				glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS
				&& !ImGui::GetIO().WantCaptureKeyboard
			);
			if (currTab && !sPrevTab) {
				sFpMode = !sFpMode;
				if (sFpMode) {
					fpCam.position = srat::camera_orbit_eye(cam);
					fpCam.yaw = -cam.azimuth;
					fpCam.pitch = -cam.elevation;
					fpCam.fovY = cam.fovY;
					fpCam.aspect = cam.aspect;
					fpCam.near = cam.near;
					fpCam.far = cam.far;
				} else {
					cam.azimuth = -fpCam.yaw;
					cam.elevation = -fpCam.pitch;
					cam.target = (
						fpCam.position
						+ srat::camera_fp_forward(fpCam) * cam.distance
					);
				}
			}
			sPrevTab = currTab;
		}

		double curMouseX, curMouseY;
		glfwGetCursorPos(window, &curMouseX, &curMouseY);
		f32 const dx = (f32)(curMouseX - prevMouseX);
		f32 const dy = (f32)(curMouseY - prevMouseY);
		prevMouseX = curMouseX;
		prevMouseY = curMouseY;

		f32 sceneBboxExtent = cam.distance * 2.0f;
		{
			f32v3 bboxMin = { 1e30f, 1e30f, 1e30f };
			f32v3 bboxMax = { -1e30f, -1e30f, -1e30f };
			for (SceneInstance const & inst : sceneDesc.instances) {
				bboxMin.x = std::min(bboxMin.x, inst.position.x);
				bboxMin.y = std::min(bboxMin.y, inst.position.y);
				bboxMin.z = std::min(bboxMin.z, inst.position.z);
				bboxMax.x = std::max(bboxMax.x, inst.position.x);
				bboxMax.y = std::max(bboxMax.y, inst.position.y);
				bboxMax.z = std::max(bboxMax.z, inst.position.z);
			}
			sceneBboxExtent = std::max(
				{
					bboxMax.x - bboxMin.x,
					bboxMax.y - bboxMin.y,
					bboxMax.z - bboxMin.z,
					cam.distance * 2.0f,
				}
			);
		}
		bool const fpLookActive = (
			sFpMode
			&& !ImGui::GetIO().WantCaptureMouse
			&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
		);
		if (!fpLookActive && prevFpLookActive) {
			glfwGetCursorPos(window, &prevMouseX, &prevMouseY);
		}
		if (fpLookActive) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			srat::camera_fp_look(fpCam, dx * 0.005f, -dy * 0.005f);
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		if (sFpMode && !ImGui::GetIO().WantCaptureKeyboard) {
			f32 const speed = sFlyCamSpeed * sceneBboxExtent * dt;
			f32v3 localDelta = {};
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { localDelta.z += speed; }
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { localDelta.z -= speed; }
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { localDelta.x += speed; }
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { localDelta.x -= speed; }
			if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { localDelta.y += speed; }
			if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { localDelta.y -= speed; }
			srat::camera_fp_move(fpCam, localDelta);
		} else if (!sFpMode && !ImGui::GetIO().WantCaptureMouse) {
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
				srat::camera_orbit_rotate(cam, -dx * 0.005f, dy * 0.005f);
			}
			if (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS
			) {
				srat::camera_orbit_pan(
					cam,
					dx * 0.01f * cam.distance,
					-dy * 0.01f * cam.distance
				);
			}
			srat::camera_orbit_zoom(cam, -sScrollDelta * 0.1f * cam.distance);
		}
		sScrollDelta = 0.0f;
		prevFpLookActive = fpLookActive;

		int winW, winH;
		glfwGetWindowSize(window, &winW, &winH);
		cam.aspect = (f32)winW / (f32)winH;
		fpCam.aspect = cam.aspect;

		for (SceneInstance const & inst : sceneDesc.instances) {
			fnVerifyModelLoaded(inst.filename);
		}

		bool const rightClick = (
			!ImGui::GetIO().WantCaptureMouse
			&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
		);
		GpuDebugPC const debugPC {
			.debugMode = (i32)sDebugMode,
			.probeX = (i32)curMouseX,
			.probeY = (i32)curMouseY,
			.probeActive = rightClick ? 1u : 0u,
			.mipLodBias = sMipLodBias,
			.mipOverrideActive = sMipOverrideActive ? 1u : 0u,
			.mipLodOverride = sMipLodOverride,
		};
		vkof::buffer_upload({
			.buffer = debugPcBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&debugPC), sizeof(debugPC)
			),
		});
		GpuGlobalPC const globalPC {
			.time = (f32)glfwGetTime(),
			.cameraPos = sFpMode ? fpCam.position : srat::camera_orbit_eye(cam),
			.lightCount = (u32)sceneDesc.lights.size(),
			.lightsVa = vkof::buffer_virtual_address(lightsBuffer),
			.exposure = sExposure,
			.selectedObject = sSelectedObject,
			.viewProj = (
				sFpMode
				? (srat::camera_fp_proj(fpCam) * srat::camera_fp_view(fpCam))
				: (srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam))
			),
			.debug = vkof::buffer_virtual_address(debugPcBuffer),
			.shadowsEnabled = (tlas.id != 0u && !modelList.empty()) ? 1u : 0u,
			._reserved = {},
		};

		vkof::imgui_begin();

		if (char const * uri = asset_library_imgui(assetLib)) {
			std::string const relPath = std::filesystem::relative(
				std::filesystem::path(uri), repoDir
			).string();
			scene_desc_add_instance(
				sceneDesc,
				relPath,
				sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
			);
			fnVerifyModelLoaded(relPath);
		}

		if (!singleModelMode) {
			i32 selectedInstIdx = -1;
			if (
				sSelectedObject != 0xFFFFFFFFu
				&& sSelectedObject < (u32)sDrawToInstIdx.size()
			) {
				selectedInstIdx = (i32)sDrawToInstIdx[sSelectedObject];
			}
			SceneDescImguiResult const listResult = (
				scene_desc_imgui(sceneDesc, sceneFilePath, selectedInstIdx)
			);
			if (listResult.focusIdx >= 0) {
				cam.target = sceneDesc.instances[listResult.focusIdx].position;
			}
			if (listResult.selectedIdx >= 0) {
				u32 const newInstIdx = (u32)listResult.selectedIdx;
				sSelectedObject = 0xFFFFFFFFu;
				for (u32 di = 0u; di < (u32)sDrawToInstIdx.size(); ++di) {
					if (sDrawToInstIdx[di] == newInstIdx) {
						sSelectedObject = di;
						break;
					}
				}
			}
		}

		ImGui::Begin("scene");
		ImGui::Text("fps: %.1f", ImGui::GetIO().Framerate);
		ImGui::Separator();
		if (
			sSelectedObject != 0xFFFFFFFFu
			&& sSelectedObject < (u32)sDrawToInstIdx.size()
		) {
			u32 const instIdx = sDrawToInstIdx[sSelectedObject];
			SceneInstance & selInst = sceneDesc.instances[instIdx];
			std::string const selLabel = (
				std::filesystem::path(selInst.filename).stem().string()
			);
			ImGui::Text("selected: %s", selLabel.c_str());
			ImGui::SameLine();
			if (ImGui::Button("deselect")) {
				sSelectedObject = 0xFFFFFFFFu;
			}
			ImGui::DragFloat3("position", &selInst.position.x, 0.01f);
			ImGui::DragFloat3("rotation", &selInst.rotation.x, 0.5f);
			ImGui::DragFloat("scale", &selInst.scale, 0.01f, 0.001f, 1000.0f);
			ImGui::Separator();
		}

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
		if (ImGui::Checkbox("fly cam [Tab / ESC]", &sFpMode)) {
			if (sFpMode) {
				fpCam.position = srat::camera_orbit_eye(cam);
				fpCam.yaw = -cam.azimuth;
				fpCam.pitch = -cam.elevation;
				fpCam.fovY = cam.fovY;
				fpCam.aspect = cam.aspect;
				fpCam.near = cam.near;
				fpCam.far = cam.far;
			} else {
				cam.azimuth = -fpCam.yaw;
				cam.elevation = -fpCam.pitch;
				cam.target = (
					fpCam.position
					+ srat::camera_fp_forward(fpCam) * cam.distance
				);
			}
		}
		if (sFpMode) {
			ImGui::SliderFloat(
				"fly speed", &sFlyCamSpeed, 0.001f, 2.0f, "%.4f",
				ImGuiSliderFlags_Logarithmic
			);
			ImGui::TextDisabled("right-click + drag to look");
		}
		// these match MOR_DEBUG_VIEW* in mor_shared.h
		static constexpr char const * kDebugModes[] = {
			"none",
			"meshlet index",
			"model index",
			"triangle index",
			"mip heatmap",
		};
		ImGui::Combo(
			"debug", &sDebugMode, kDebugModes, IM_ARRAYSIZE(kDebugModes)
		);
		ImGui::Separator();
		ImGui::Checkbox("override mip LOD", &sMipOverrideActive);
		if (sMipOverrideActive) {
			ImGui::SliderFloat("mip level", &sMipLodOverride, 0.0f, 12.0f, "%.1f");
		} else {
			ImGui::SliderFloat("mip LOD bias", &sMipLodBias, -4.0f, 4.0f, "%.1f");
		}
		ImGui::DragFloat("exposure", &sExposure, 0.01f, 0.0f, 10.0f, "%.2f");
		ImGui::SliderFloat("anisotropy", &sAnisotropyPending, 1.0f, 16.0f, "%.0f");
		if (ImGui::IsItemDeactivatedAfterEdit() && sAnisotropyPending != sAnisotropy) {
			sAnisotropy = sAnisotropyPending;
			vkof::device_wait_idle();
			for (auto const & [filename, modelIdx] : loadedModels) {
				mor::scene_set_anisotropy(
					modelList[modelIdx].scene,
					modelList[modelIdx].gpuScene,
					sAnisotropy
				);
			}
		}
		ImGui::End();

		ImGui::Begin("lights");
		if (ImGui::Button("add light")) {
			sceneDesc.lights.emplace_back(GpuLight {
				.position = (
					sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
				),
				.radius = 10.0f,
				.color = { 50.0f, 50.0f, 50.0f },
			});
		}
		i32 removeLightIdx = -1;
		for (u32 i = 0u; i < (u32)sceneDesc.lights.size(); ++i) {
			GpuLight & light = sceneDesc.lights[i];
			ImGui::PushID((int)i);
			char hdrLabel[32];
			snprintf(hdrLabel, sizeof(hdrLabel), "light %u", i);
			ImGui::SetNextItemAllowOverlap();
			bool const open = ImGui::CollapsingHeader(hdrLabel);
			float const focusW = (
				ImGui::CalcTextSize("focus").x
				+ ImGui::GetStyle().FramePadding.x * 2.0f
			);
			float const removeW = (
				ImGui::CalcTextSize("remove").x
				+ ImGui::GetStyle().FramePadding.x * 2.0f
			);
			float const spacing = ImGui::GetStyle().ItemSpacing.x;
			ImGui::SameLine(
				ImGui::GetContentRegionMax().x - focusW - removeW - spacing
			);
			if (ImGui::SmallButton("focus")) {
				cam.target = light.position;
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("remove")) {
				removeLightIdx = (i32)i;
			}
			if (open) {
				ImGui::DragFloat3("pos", &light.position.x, 0.1f);
				ImGui::DragFloat("radius", &light.radius, 0.1f, 0.01f, 1000.0f);
				f32v3 & col = light.color;
				float maxComp = std::max({col.x, col.y, col.z, 0.0001f});
				f32v3 norm = { col.x / maxComp, col.y / maxComp, col.z / maxComp };
				bool changed = false;
				if (ImGui::ColorEdit3("hue", &norm.x)) { changed = true; }
				if (
					ImGui::DragFloat(
						"intensity", &maxComp, 0.5f, 0.01f, 2000.0f, "%.1f"
					)
				) {
					changed = true;
				}
				if (changed) {
					col = { norm.x * maxComp, norm.y * maxComp, norm.z * maxComp };
				}
			}
			ImGui::PopID();
		}
		if (removeLightIdx >= 0) {
			sceneDesc.lights.erase(sceneDesc.lights.begin() + removeLightIdx);
		}
		ImGui::End();

		ImGui::Begin("ddgi volumes");
		if (ImGui::Button("add volume")) {
			sceneDesc.ddgiVolumes.emplace_back(DdgiVolume {
				.origin = (
					sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
				),
				.probeSpacing = { 1.0f, 1.0f, 1.0f },
				.probeCounts = { 8u, 4u, 8u },
			});
		}
		i32 removeVolIdx = -1;
		for (u32 i = 0u; i < (u32)sceneDesc.ddgiVolumes.size(); ++i) {
			DdgiVolume & vol = sceneDesc.ddgiVolumes[i];
			ImGui::PushID((int)i);
			char volLabel[32];
			snprintf(volLabel, sizeof(volLabel), "volume %u", i);
			ImGui::SetNextItemAllowOverlap();
			bool const open = ImGui::CollapsingHeader(volLabel);
			float const focusW = (
				ImGui::CalcTextSize("focus").x
				+ ImGui::GetStyle().FramePadding.x * 2.0f
			);
			float const removeW = (
				ImGui::CalcTextSize("remove").x
				+ ImGui::GetStyle().FramePadding.x * 2.0f
			);
			float const spacing = ImGui::GetStyle().ItemSpacing.x;
			ImGui::SameLine(
				ImGui::GetContentRegionMax().x - focusW - removeW - spacing
			);
			if (ImGui::SmallButton("focus")) {
				f32v3 const volMax = vol.max();
				cam.target = {
					(vol.origin.x + volMax.x) * 0.5f,
					(vol.origin.y + volMax.y) * 0.5f,
					(vol.origin.z + volMax.z) * 0.5f,
				};
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("remove")) { removeVolIdx = (i32)i; }
			if (open) {
				ImGui::DragFloat3("origin", &vol.origin.x, 0.1f);
				ImGui::DragFloat3(
					"spacing", &vol.probeSpacing.x, 0.01f, 0.01f, 100.0f
				);
				int counts[3] = {
					(int)vol.probeCounts.x,
					(int)vol.probeCounts.y,
					(int)vol.probeCounts.z,
				};
				if (ImGui::DragInt3("probe counts", counts, 1.0f, 1, 64)) {
					vol.probeCounts.x = (u32)counts[0];
					vol.probeCounts.y = (u32)counts[1];
					vol.probeCounts.z = (u32)counts[2];
				}
			}
			ImGui::PopID();
		}
		if (removeVolIdx >= 0) {
			sceneDesc.ddgiVolumes.erase(
				sceneDesc.ddgiVolumes.begin() + removeVolIdx
			);
		}
		ImGui::End();

		if (!sceneDesc.lights.empty()) {
			vkof::buffer_upload({
				.buffer = lightsBuffer,
				.byteOffset = 0u,
				.data = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(sceneDesc.lights.data()),
					sceneDesc.lights.size() * sizeof(GpuLight)
				),
			});
		}

		if (char const * msg = vkof::probe_message()) {
			u32 pickedModel = 0u;
			if (sscanf(msg, "__MODEL__: %u", &pickedModel) == 1) {
				sSelectedObject = pickedModel;
			}
			ImGui::BeginTooltip();
			ImGui::Text("%s", msg);
			ImGui::EndTooltip();
		}

		// -- upload model indirects
		{
			std::vector<GpuResolveModelIndirect> drawDescs;
			for (SceneInstance const & inst : sceneDesc.instances) {
				auto const it = loadedModels.find(inst.filename);
				if (it == loadedModels.end()) { continue; }
				LoadedModel const & model = modelList[it->second];
				mor::Buffers const bufs = mor::scene_gpu_buffers(model.gpuScene);
				drawDescs.push_back(GpuResolveModelIndirect {
					.meshlets = bufs.meshlets,
					.materials = bufs.materials,
					.positions = bufs.positions,
					.instances = bufs.instances,
					.attributes = bufs.attributes,
					.meshletVerts = bufs.meshletVerts,
					.meshletTris = bufs.meshletTris,
					.modelMatrix = make_model_matrix(inst),
				});
			}
			if (!drawDescs.empty()) {
				vkof::buffer_upload({
					.buffer = modelsIndirectBuffer,
					.byteOffset = 0u,
					.data = srat::slice<u8 const>(
						reinterpret_cast<u8 const *>(drawDescs.data()),
						drawDescs.size() * sizeof(GpuResolveModelIndirect)
					),
				});
			}
		}

		// -- draw debug scenes
		for (GpuLight const & light : sceneDesc.lights) {
			vkof::debug_draw_sphere(
				/*center=*/light.position,
				/*radius=*/light.radius,
				/*color=*/light.color
			);
		}
		static std::vector<vkof::DebugSphere> probes;
		probes.clear();
		for (DdgiVolume const & vol : sceneDesc.ddgiVolumes) {
			vkof::debug_draw_box(
				/*min=*/vol.min(),
				/*max=*/vol.max(),
				/*color=*/{ 0.0f, 1.0f, 1.0f }
			);
			f32 const probeRadius = (
				std::min({
					vol.probeSpacing.x,
					vol.probeSpacing.y,
					vol.probeSpacing.z,
				}) * 0.2f
			);
			for (u32 pz = 0u; pz < vol.probeCounts.z; ++pz)
			for (u32 py = 0u; py < vol.probeCounts.y; ++py)
			for (u32 px = 0u; px < vol.probeCounts.x; ++px) {
				probes.emplace_back(vkof::DebugSphere {
					.position = {
						vol.origin.x + vol.probeSpacing.x * (f32)px,
						vol.origin.y + vol.probeSpacing.y * (f32)py,
						vol.origin.z + vol.probeSpacing.z * (f32)pz,
					},
					.radius = probeRadius,
				});
			}
		}
		if (!probes.empty()) {
			vkof::debug_draw_spheres(
				srat::slice<vkof::DebugSphere const>(
					probes.data(), probes.size()
				),
				debugProbePipeline
			);
		}

		// -- render graph
		{
			// -- visibility draw
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
					u32 drawIdx = 0u;
					sDrawToInstIdx.clear();
					for (u32 instIdx = 0u; instIdx < (u32)sceneDesc.instances.size(); ++instIdx) {
						SceneInstance const & inst = sceneDesc.instances[instIdx];
						auto const it = loadedModels.find(inst.filename);
						if (it == loadedModels.end()) { continue; }
						sDrawToInstIdx.emplace_back(instIdx);
						LoadedModel const & model = modelList[it->second];
						mor::Buffers const bufs = (
							mor::scene_gpu_buffers(model.gpuScene)
						);
						u32 const count = (
							mor::scene_gpu_meshlet_count(model.gpuScene)
						);
						GpuSceneDrawPC const drawPC {
							.modelId = drawIdx,
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
						++drawIdx;
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

			// -- TLAS build node
			vkof::RenderNode const tlasNode = vkof::render_node_create({
				.queue = vkof::CommandQueue::graphics,
			});
			vkof::render_node_callback({
				.node = tlasNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					std::vector<vkof::TlasInstance> tlasInstances;
					for (u32 di = 0u; di < (u32)sDrawToInstIdx.size(); ++di) {
						SceneInstance const & inst = (
							sceneDesc.instances[sDrawToInstIdx[di]]
						);
						auto const it = loadedModels.find(inst.filename);
						if (it == loadedModels.end()) { continue; }
						LoadedModel const & model = modelList[it->second];
						tlasInstances.emplace_back(vkof::TlasInstance {
							.blas = model.blas,
							.transform = make_model_matrix(inst),
							.instanceCustomIndex = di,
							.rayMask = 0xFFu,
						});
					}
					vkof::tlas_build(
						cmd,
						tlas,
						srat::slice<vkof::TlasInstance const>(
							tlasInstances.data(), tlasInstances.size()
						)
					);
				},
			});
			vkof::acceleration_structure_set_tlas(tlas);

			// -- resolve node
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

			// -- execute render graph
			vkof::RenderNode const nodes[] = { drawNode, tlasNode, resolveNode, };
			vkof::render_graph_execute({
				.nodes = srat::slice<vkof::RenderNode const>(nodes, 3u),
				.rootPushconstant = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(&globalPC), sizeof(globalPC)
				),
				.finalImage = colorTarget,
				.debugDrawViewProj = globalPC.viewProj,
				.debugDrawDepth = depthTarget,
			});
			vkof::render_node_destroy(drawNode);
			vkof::render_node_destroy(tlasNode);
			vkof::render_node_destroy(resolveNode);
		}
	}

	vkof::device_wait_idle();
	for (auto & [filename, modelIndex] : loadedModels) {
		LoadedModel const & model = modelList[modelIndex];
		mor::scene_destroy(model.scene);
		mor::scene_gpu_destroy(model.gpuScene);
		vkof::blas_destroy(model.blas);
	}
	loadedModels.clear();
	modelList.clear();
	vkof::buffer_destroy(modelsIndirectBuffer);
	vkof::buffer_destroy(lightsBuffer);
	vkof::pipeline_destroy(visibilityPipeline);
	vkof::pipeline_destroy(resolvePipeline);
	vkof::pipeline_destroy(debugProbePipeline);
	mor::sampler_cache_destroy();
	vkof::shutdown();
	return 0;
}
