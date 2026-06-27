#include "shaders/scene_shared.h"
#include "shaders/resolve_pc.h"
#include "shared/ddgi-shared.h"
#include "shared/light_shared.h"
#include "asset_library.hpp"
#include <scene/scene.hpp>

#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <srat/camera.hpp>

#include <imgui.h>
#include <ImGuizmo.h>
#include <backends/imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <variant>

static float sScrollDelta = 0.0f;
static int sDebugMode = 0;
static bool sMipOverrideActive = false;
static float sMipLodOverride = 0.0f;
static float sMipLodBias = 0.0f;
static float sAnisotropy = 16.0f;
static float sAnisotropyPending = 16.0f;
static float sExposure = 1.0f;
static float sSkyTurbidity = 1.7f;
static float sSkyTimeScale = 0.05f;
static float sSunIntensity = 1.0f;
static bool sFpMode = false;
static float sFlyCamSpeed = 0.05f;
static std::optional<u32> sSelected;
static std::vector<u32> sDrawToEntityIdx;
static ImGuizmo::OPERATION sGizmoOp = ImGuizmo::TRANSLATE;
static bool sShowDdgiProbes = false;

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

static void tick(char const * label, Clock::time_point & t) {
	printf("  [%.1f ms] %s\n", Ms(Clock::now() - t).count(), label);
	t = Clock::now();
}

// -----------------------------------------------------------------------------
// -- ddgi
// -----------------------------------------------------------------------------

struct DdgiVolume
{
	vkof::Image imageIrradiance;
	vkof::Image imageDepth;
	vkof::Sampler commonSampler;
	u32 irradianceStorageHandle;
	u32 depthStorageHandle;
	u32 irradianceSamplerHandle;
	u32 depthSamplerHandle;
	ImTextureID irradianceImguiId;
	ImTextureID depthImguiId;
};

DdgiVolume ddgi_volume_create(u32v3 const probeCounts)
{
	DdgiVolume vol = {
		.imageIrradiance = {},
		.imageDepth = {},
		.commonSampler = {},
		.irradianceStorageHandle = 0u,
		.depthStorageHandle = 0u,
		.irradianceSamplerHandle = 0u,
		.depthSamplerHandle = 0u,
		.irradianceImguiId = nullptr,
		.depthImguiId = nullptr,
	};

	// irradiance uses 6x6 octahedron with 1-pixel border on each side = 8x8
	// depth uses 14x14 octahedron with 1-pixel border on each side = 16x16
	static constexpr u32 skIrradianceRes = 6u + 2u;
	static constexpr u32 skDepthRes = 14u + 2u;

	vol.imageIrradiance = vkof::image_create({
		.width = probeCounts.x * skIrradianceRes,
		.height = probeCounts.y * skIrradianceRes,
		.depth = probeCounts.z,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});

	vol.imageDepth = vkof::image_create({
		.width = probeCounts.x * skDepthRes,
		.height = probeCounts.y * skDepthRes,
		.depth = probeCounts.z,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});

	vol.commonSampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::linear,
		.minFilter = vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
		.mipmapMode = vkof::SamplerMipmapMode::nearest,
	});
	vol.irradianceStorageHandle = vkof::image_storage3d_handle({
		.image = vol.imageIrradiance,
		.mipLevel = 0u,
	});
	vol.depthStorageHandle = vkof::image_storage3d_handle({
		.image = vol.imageDepth,
		.mipLevel = 0u,
	});
	vol.irradianceSamplerHandle = vkof::image_sampler3d_handle({
		.image = vol.imageIrradiance,
		.sampler = vol.commonSampler,
	});
	vol.depthSamplerHandle = vkof::image_sampler3d_handle({
		.image = vol.imageDepth,
		.sampler = vol.commonSampler,
	});

	return vol;
}

void ddgi_volume_destroy(DdgiVolume & dv)
{
	if (dv.irradianceImguiId) { vkof::image_imgui_id_destroy(dv.irradianceImguiId); }
	if (dv.depthImguiId) { vkof::image_imgui_id_destroy(dv.depthImguiId); }
	vkof::image_destroy(dv.imageIrradiance);
	vkof::image_destroy(dv.imageDepth);
	vkof::sampler_destroy(dv.commonSampler);
	dv = {};
}

// -----------------------------------------------------------------------------
// -- specular temporal history
// -----------------------------------------------------------------------------

struct SpecularHistory
{
	vkof::Image normalImage {};
	vkof::Image specularImage[2] {};
	vkof::Image momentImage[2] {};
	vkof::Sampler commonSampler {};
	u32 normalSamplerHandle { 0u };
	u32 normalStorageHandle { 0u };
	u32 specularSamplerHandle[2] { 0u, 0u };
	u32 specularStorageHandle[2] { 0u, 0u };
	u32 momentSamplerHandle[2] { 0u, 0u };
	u32 momentStorageHandle[2] { 0u, 0u };
	ImTextureID normalImguiId { nullptr };
	ImTextureID specularImguiId[2] { nullptr, nullptr };
	ImTextureID momentImguiId[2] { nullptr, nullptr };
};

SpecularHistory specular_history_create(u32 const width, u32 const height)
{
	SpecularHistory sh = {};

	sh.normalImage = vkof::image_create({
		.width = width,
		.height = height,
		.depth = 1u,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	for (u32 i = 0u; i < 2u; ++i) {
		sh.specularImage[i] = vkof::image_create({
			.width = width,
			.height = height,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		sh.momentImage[i] = vkof::image_create({
			.width = width,
			.height = height,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
	}

	sh.commonSampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::linear,
		.minFilter = vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
		.mipmapMode = vkof::SamplerMipmapMode::nearest,
	});

	sh.normalSamplerHandle = vkof::image_sampler_handle({
		.image = sh.normalImage,
		.sampler = sh.commonSampler,
	});
	sh.normalStorageHandle = vkof::image_storage_handle({
		.image = sh.normalImage,
		.mipLevel = 0u,
	});
	for (u32 i = 0u; i < 2u; ++i) {
		sh.specularSamplerHandle[i] = vkof::image_sampler_handle({
			.image = sh.specularImage[i],
			.sampler = sh.commonSampler,
		});
		sh.specularStorageHandle[i] = vkof::image_storage_handle({
			.image = sh.specularImage[i],
			.mipLevel = 0u,
		});
		sh.momentSamplerHandle[i] = vkof::image_sampler_handle({
			.image = sh.momentImage[i],
			.sampler = sh.commonSampler,
		});
		sh.momentStorageHandle[i] = vkof::image_storage_handle({
			.image = sh.momentImage[i],
			.mipLevel = 0u,
		});
	}

	return sh;
}

void specular_history_destroy(SpecularHistory & sh)
{
	if (sh.normalImguiId) {
		vkof::image_imgui_id_destroy(sh.normalImguiId);
	}
	for (u32 i = 0u; i < 2u; ++i) {
		if (sh.specularImguiId[i]) {
			vkof::image_imgui_id_destroy(sh.specularImguiId[i]);
		}
		if (sh.momentImguiId[i]) {
			vkof::image_imgui_id_destroy(sh.momentImguiId[i]);
		}
	}
	vkof::image_destroy(sh.normalImage);
	for (u32 i = 0u; i < 2u; ++i) {
		vkof::image_destroy(sh.specularImage[i]);
		vkof::image_destroy(sh.momentImage[i]);
	}
	vkof::sampler_destroy(sh.commonSampler);
	sh = {};
}

struct DdgiCascade
{
	DdgiVolume volume {};
	f32v3 snappedOrigin {};
	u32v3 scrollOffset {};
	u32v3 frameInvalidStart {};
	u32v3 frameInvalidCount {};
};

struct DdgiSystem
{
	DdgiCascade cascades[skMaxDdgiCascades] {};
	u32 cascadeCount { 3u };
	u32v3 probeCounts { 16u, 16u, 16u };
	f32 probeSpacing { 2.0f };
	f32 cascadeScale { 8.0f };
	bool dirty { true };
	bool frozen { false };
};
static DdgiSystem sDdgi {};
static SpecularHistory sSpecularHistory {};
static u32 sFrameIndex = 0u;


// -----------------------------------------------------------------------------
// -- model
// -----------------------------------------------------------------------------

struct LoadedModel
{
	mor::Scene scene;
	mor::GpuScene gpuScene;
	u32 modelId;
	vkof::AccelerationStructureBlas blas;
	f32v3 boundsMin;
	f32v3 boundsMax;
};

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

	bool const singleModelMode = (gltfPath != nullptr && scenePath == nullptr);
	std::filesystem::path sceneFilePath = [&]() -> std::filesystem::path {
		if (scenePath) { return std::filesystem::path(scenePath); }
		if (singleModelMode) {
			std::filesystem::path const gltfFsPath = (
				std::filesystem::canonical(gltfPath)
			);
			return (
				gltfFsPath.parent_path()
				/ (gltfFsPath.stem().string() + ".json")
			);
		}
		return {};
	}();
	scene::Desc sceneDesc = !sceneFilePath.empty()
		? scene::desc_load(sceneFilePath)
		: scene::Desc{};
	bool const freshSingleModel = singleModelMode && sceneDesc.entities.empty();

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
	vkof::Pipeline const ddgiTraceIrradiancePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(shaderDir / "ddgi-trace-irradiance.comp").string().c_str()
			),
			.includePaths = srat::slice { kIncludePaths, 1u },
		})
	);
	vkof::Pipeline const ddgiTraceDepthPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(shaderDir / "ddgi-trace-depth.comp").string().c_str()
			),
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
				(shaderDir / "ddgi-debug-probe.frag").string().c_str()
			),
			.includePaths = srat::slice { kIncludePaths, 1u },
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

	if (freshSingleModel) {
		scene::desc_add_entity(
			sceneDesc,
			scene::EntityInstance {
				.filename = std::filesystem::canonical(gltfPath).string(),
				.rotation = {},
				.scale = 1.0f,
			},
			{}
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
	vkof::Buffer const ddgiGridBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuDdgiCascades),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::AccelerationStructureTlas const tlas = (
		vkof::tlas_create({ .maxInstances = kMaxModels })
	);

	for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
		sDdgi.cascades[ci].volume = ddgi_volume_create(sDdgi.probeCounts);
	}
	sDdgi.dirty = false;
	sSpecularHistory = specular_history_create(kScreenW, kScreenH);

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
		f32v3 boundsMin, boundsMax;
		mor::scene_bounds(scene, boundsMin, boundsMax);
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
			.boundsMin = boundsMin,
			.boundsMax = boundsMax,
		});
	};

	for (scene::Entity const & entity : sceneDesc.entities) {
		if (auto * ei = std::get_if<scene::EntityInstance>(&entity.data)) {
			fnVerifyModelLoaded(ei->filename);
		}
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
		++sFrameIndex;

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
		if (!ImGui::GetIO().WantCaptureKeyboard) {
			static bool sPrevT = false, sPrevR = false, sPrevGrave = false;
			bool const currT = glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS;
			bool const currR = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
			bool const currGrave = glfwGetKey(window, GLFW_KEY_GRAVE_ACCENT) == GLFW_PRESS;
			if (currT && !sPrevT) { sGizmoOp = ImGuizmo::TRANSLATE; }
			if (currR && !sPrevR) { sGizmoOp = ImGuizmo::ROTATE; }
			if (currGrave && !sPrevGrave) { sSelected = std::nullopt; }
			sPrevT = currT; sPrevR = currR; sPrevGrave = currGrave;
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
			for (scene::Entity const & entity : sceneDesc.entities) {
				if (!std::holds_alternative<scene::EntityInstance>(entity.data)) { continue; }
				bboxMin.x = std::min(bboxMin.x, entity.position.x);
				bboxMin.y = std::min(bboxMin.y, entity.position.y);
				bboxMin.z = std::min(bboxMin.z, entity.position.z);
				bboxMax.x = std::max(bboxMax.x, entity.position.x);
				bboxMax.y = std::max(bboxMax.y, entity.position.y);
				bboxMax.z = std::max(bboxMax.z, entity.position.z);
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
			&& (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS
				|| glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
			)
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

		for (scene::Entity const & entity : sceneDesc.entities) {
			if (auto * ei = std::get_if<scene::EntityInstance>(&entity.data)) {
				fnVerifyModelLoaded(ei->filename);
			}
		}

		bool const rightClick = (
			!ImGui::GetIO().WantCaptureMouse
			&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
		);
		bool const leftClickSelect = (
			!sFpMode
			&& !ImGui::GetIO().WantCaptureMouse
			&& ImGui::IsMouseReleased(ImGuiMouseButton_Left)
			&& !ImGui::IsMouseDragging(ImGuiMouseButton_Left)
		);
		f32 const sunAngle = (f32)glfwGetTime() * sSkyTimeScale;
		f32v3 const sunDir = f32v3_normalize({
			cosf(sunAngle),
			sinf(sunAngle) * 0.8f + 0.2f,
			0.3f,
		});
		GpuDebugPC const debugPC {
			.debugMode = (i32)sDebugMode,
			.probeX = (i32)curMouseX,
			.probeY = (i32)curMouseY,
			.probeActive = (rightClick || leftClickSelect) ? 1u : 0u,
			.mipLodBias = sMipLodBias,
			.mipOverrideActive = sMipOverrideActive ? 1u : 0u,
			.mipLodOverride = sMipLodOverride,
			.sunDir = sunDir,
			.skyTurbidity = sSkyTurbidity,
			.sunIntensity = sSunIntensity,
		};
		vkof::buffer_upload({
			.buffer = debugPcBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&debugPC), sizeof(debugPC)
			),
		});

		// -- build gpu lights from entities
		std::vector<GpuLight> gpuLights;
		for (scene::Entity const & entity : sceneDesc.entities) {
			if (auto * el = std::get_if<scene::EntityLight>(&entity.data)) {
				gpuLights.push_back(GpuLight {
					.position = entity.position,
					.radius = el->radius,
					.color = el->color,
				});
			}
		}
		if (!gpuLights.empty()) {
			vkof::buffer_upload({
				.buffer = lightsBuffer,
				.byteOffset = 0u,
				.data = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(gpuLights.data()),
					gpuLights.size() * sizeof(GpuLight)
				),
			});
		}

		// -- reinit ddgi if probe counts changed
		bool const ddgiReinit = sDdgi.dirty;
		if (sDdgi.dirty) {
			vkof::device_wait_idle();
			for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
				ddgi_volume_destroy(sDdgi.cascades[ci].volume);
				sDdgi.cascades[ci].volume = ddgi_volume_create(sDdgi.probeCounts);
				sDdgi.cascades[ci].scrollOffset = {};
			}
			sDdgi.dirty = false;
		}

		// -- snap ddgi origins and update toroidal scroll offsets per cascade
		{
			f32v3 const cameraPos = (
				sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
			);

			auto const updateAxis = [](
				u32 & scroll,
				u32 & invStart,
				u32 & invCount,
				i32 const delta,
				u32 const count
			) {
				invStart = 0u;
				invCount = 0u;
				if (delta == 0) { return; }
				u32 const absDelta = (u32)i32_min(i32_abs(delta), (i32)count);
				u32 const prevScroll = scroll;
				scroll = (u32)(
					(((i32)scroll + delta) % (i32)count + (i32)count) % (i32)count
				);
				invStart = (delta > 0) ? prevScroll : scroll;
				invCount = absDelta;
			};

			f32 snapSpacingScale = 1.0f;
			for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
				DdgiCascade & cascade = sDdgi.cascades[ci];
				f32 const s = sDdgi.probeSpacing * snapSpacingScale;
				f32v3 const half = {
					f32(sDdgi.probeCounts.x) * s * 0.5f,
					f32(sDdgi.probeCounts.y) * s * 0.5f,
					f32(sDdgi.probeCounts.z) * s * 0.5f,
				};
				f32v3 const newSnapped = {
					std::floor((cameraPos.x - half.x) / s) * s,
					std::floor((cameraPos.y - half.y) / s) * s,
					std::floor((cameraPos.z - half.z) / s) * s,
				};

				// -- compute invalid probe slice from per-axis delta
				if (ddgiReinit) {
					cascade.frameInvalidStart = { 0u, 0u, 0u };
					cascade.frameInvalidCount = sDdgi.probeCounts;
					cascade.snappedOrigin = newSnapped;
				} else if (sDdgi.frozen) {
					cascade.frameInvalidStart = {};
					cascade.frameInvalidCount = {};
				} else {
					i32 const dx = (
						(i32)f32_round(
							(newSnapped.x - cascade.snappedOrigin.x) / s
						)
					);
					i32 const dy = (
						(i32)f32_round(
							(newSnapped.y - cascade.snappedOrigin.y) / s
						)
					);
					i32 const dz = (
						(i32)f32_round(
							(newSnapped.z - cascade.snappedOrigin.z) / s
						)
					);
					updateAxis(
						cascade.scrollOffset.x,
						cascade.frameInvalidStart.x,
						cascade.frameInvalidCount.x,
						dx, sDdgi.probeCounts.x
					);
					updateAxis(
						cascade.scrollOffset.y,
						cascade.frameInvalidStart.y,
						cascade.frameInvalidCount.y,
						dy, sDdgi.probeCounts.y
					);
					updateAxis(
						cascade.scrollOffset.z,
						cascade.frameInvalidStart.z,
						cascade.frameInvalidCount.z,
						dz, sDdgi.probeCounts.z
					);
					cascade.snappedOrigin = newSnapped;
				}
				snapSpacingScale *= sDdgi.cascadeScale;
			}
		}

		// -- build ddgi cascade data and upload
		GpuDdgiCascades gpuCascades { .grids = {}, .count = sDdgi.cascadeCount };
		{
			f32 uploadSpacingScale = 1.0f;
			for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
				DdgiCascade const & cascade = sDdgi.cascades[ci];
				DdgiVolume const & dv = cascade.volume;
				f32 const s = sDdgi.probeSpacing * uploadSpacingScale;
				f32v3 const sp { s, s, s };
				gpuCascades.grids[ci] = {
					.origin = cascade.snappedOrigin,
					.probeCounts = sDdgi.probeCounts,
					.probeSpacing = sp,
					.irradianceStorageHandle = dv.irradianceStorageHandle,
					.depthStorageHandle = dv.depthStorageHandle,
					.irradianceSamplerHandle = dv.irradianceSamplerHandle,
					.depthSamplerHandle = dv.depthSamplerHandle,
					.scrollOffset = cascade.scrollOffset,
					.invalidStart = cascade.frameInvalidStart,
					.invalidCount = cascade.frameInvalidCount,
				};
				uploadSpacingScale *= sDdgi.cascadeScale;
			}
			vkof::buffer_upload({
				.buffer = ddgiGridBuffer,
				.byteOffset = 0u,
				.data = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(&gpuCascades),
					sizeof(gpuCascades)
				),
			});
		}

		// -- compute selected draw index for gpu highlighting (uses previous frame's sDrawToEntityIdx)
		u32 selectedDrawIdx = 0xFFFFFFFFu;
		if (sSelected.has_value()) {
			u32 const selEntityIdx = *sSelected;
			if (
				selEntityIdx < (u32)sceneDesc.entities.size()
				&& std::holds_alternative<scene::EntityInstance>(
					sceneDesc.entities[selEntityIdx].data
				)
			) {
				for (u32 di = 0u; di < (u32)sDrawToEntityIdx.size(); ++di) {
					if (sDrawToEntityIdx[di] == selEntityIdx) {
						selectedDrawIdx = di;
						break;
					}
				}
			}
		}

		GpuGlobalPC const globalPC {
			.time = (f32)glfwGetTime(),
			.cameraPos = sFpMode ? fpCam.position : srat::camera_orbit_eye(cam),
			.lightCount = (u32)gpuLights.size(),
			.shadowsEnabled = (tlas.id != 0u && !modelList.empty()) ? 1u : 0u,
			.lightsVa = vkof::buffer_virtual_address(lightsBuffer),
			.exposure = sExposure,
			.selectedObject = selectedDrawIdx,
			.viewProj = (
				sFpMode
				? (srat::camera_fp_proj(fpCam) * srat::camera_fp_view(fpCam))
				: (srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam))
			),
			.debug = vkof::buffer_virtual_address(debugPcBuffer),
			.ddgiGrid = vkof::buffer_virtual_address(ddgiGridBuffer),
			.models = vkof::buffer_virtual_address(modelsIndirectBuffer),
		};

		vkof::imgui_begin();

		{
			ImGuiIO const & io = ImGui::GetIO();
			ImGuizmo::BeginFrame();
			ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
			ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);
		}

		if (char const * uri = asset_library_imgui(assetLib)) {
			std::string const relPath = std::filesystem::relative(
				std::filesystem::path(uri), repoDir
			).string();
			scene::desc_add_entity(
				sceneDesc,
				scene::EntityInstance {
					.filename = relPath,
					.rotation = {},
					.scale = 1.0f,
				},
				sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
			);
			fnVerifyModelLoaded(relPath);
		}

		if (char const * msg = vkof::probe_message()) {
			u32 pickedModel = 0u;
			if (sscanf(msg, "__MODEL__: %u", &pickedModel) == 1) {
				if (rightClick && pickedModel < (u32)sDrawToEntityIdx.size()) {
					sSelected = sDrawToEntityIdx[pickedModel];
				}
			}
			ImGui::BeginTooltip();
			ImGui::Text("%s", msg);
			ImGui::EndTooltip();
		}

		// -- entity pending for removal (set inside scene window, executed after)
		std::optional<u32> removeEntityIdx;

		ImGui::Begin("scene");

		if (sceneFilePath.empty()) {
			static char sSaveAsInput[512] = {};
			ImGui::SetNextItemWidth(200.0f);
			ImGui::InputText("##saveas", sSaveAsInput, sizeof(sSaveAsInput));
			ImGui::SameLine();
			if (ImGui::Button("Save As") && sSaveAsInput[0] != '\0') {
				sceneFilePath = sSaveAsInput;
				scene::desc_save(sceneDesc, sceneFilePath);
			}
		} else {
			if (ImGui::Button("Save")) {
				scene::desc_save(sceneDesc, sceneFilePath);
			}
			ImGui::SameLine();
			if (ImGui::Button("Load")) {
				sceneDesc = scene::desc_load(sceneFilePath);
				sSelected = std::nullopt;
			}
		}
		ImGui::Separator();

		ImGui::Text("fps: %.1f", ImGui::GetIO().Framerate);

		// -- selected entity panel
		if (
			sSelected.has_value()
			&& *sSelected < (u32)sceneDesc.entities.size()
		) {
			ImGui::Separator();
			scene::Entity & selEntity = sceneDesc.entities[*sSelected];

			// -- focus / deselect / remove buttons
			{
				static bool sPrevF = false;
				bool const currF = (
					!ImGui::GetIO().WantCaptureKeyboard
					&& glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS
				);
				bool const focusPressed = ImGui::Button("focus") || (currF && !sPrevF);
				sPrevF = currF;
				if (focusPressed) {
					cam.target = scene::entity_focus_target(selEntity);
					if (sFpMode) {
						fpCam.position = (
							cam.target
							- srat::camera_fp_forward(fpCam) * cam.distance
						);
					}
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("deselect")) { sSelected = std::nullopt; }
			ImGui::SameLine();
			if (ImGui::Button("remove")) {
				removeEntityIdx = *sSelected;
				sSelected = std::nullopt;
			}

			// -- position (shared across all entity types)
			ImGui::DragFloat3("position", &selEntity.position.x, 0.01f);

			// -- type-specific fields
			std::visit(
				[&](auto & d) {
					using T = std::decay_t<decltype(d)>;
					if constexpr (std::is_same_v<T, scene::EntityInstance>) {
						ImGui::DragFloat3("rotation", &d.rotation.x, 0.5f);
						ImGui::SameLine();
						if (ImGui::Button("reset")) { d.rotation = {}; }
						ImGui::DragFloat("scale", &d.scale, 0.01f, 0.001f, 1000.0f);
					} else if constexpr (std::is_same_v<T, scene::EntityLight>) {
						ImGui::DragFloat("radius", &d.radius, 0.1f, 0.01f, 1000.0f);
						f32v3 & col = d.color;
						float maxComp = std::max({col.x, col.y, col.z, 0.0001f});
						f32v3 norm = {
							col.x / maxComp,
							col.y / maxComp,
							col.z / maxComp,
						};
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
							col = {
								norm.x * maxComp,
								norm.y * maxComp,
								norm.z * maxComp,
							};
						}
					}
				},
				selEntity.data
			);

			// -- gizmo: translate for all; also rotate for instances (T/R keys)
			{
				f32m44 const viewMat = (
					sFpMode
					? srat::camera_fp_view(fpCam)
					: srat::camera_orbit_view(cam)
				);
				f32m44 projMat = (
					sFpMode
					? srat::camera_fp_proj(fpCam)
					: srat::camera_orbit_proj(cam)
				);
				// undo Vulkan Y-flip so ImGuizmo (OpenGL convention) renders correctly
				projMat.m[5] = -projMat.m[5];
				ImGuizmo::OPERATION const gizmoOp = (
					std::holds_alternative<scene::EntityInstance>(selEntity.data)
					? sGizmoOp
					: ImGuizmo::TRANSLATE
				);
				f32m44 matrix;
				if (auto * ei = std::get_if<scene::EntityInstance>(&selEntity.data)) {
					matrix = scene::entity_model_matrix(selEntity, *ei);
				} else {
					matrix = f32m44_translate(
						selEntity.position.x,
						selEntity.position.y,
						selEntity.position.z
					);
				}
				if (ImGuizmo::Manipulate(
					viewMat.m.ptr(),
					projMat.m.ptr(),
					gizmoOp,
					ImGuizmo::WORLD,
					matrix.m.ptr()
				)) {
					f32 const * m = matrix.m.ptr();
					selEntity.position = { m[12], m[13], m[14] };
					if (auto * ei = std::get_if<scene::EntityInstance>(&selEntity.data)) {
						if (sGizmoOp == ImGuizmo::ROTATE) {
							f32 const invS = (
								1.0f / sqrtf(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])
							);
							ei->rotation.x = (
								(180.0f/3.14159265f) * asinf(-m[9] * invS)
							);
							ei->rotation.y = (
								(180.0f/3.14159265f) * atan2f(m[8] * invS, m[10] * invS)
							);
							ei->rotation.z = (
								(180.0f/3.14159265f) * atan2f(m[1] * invS, m[5] * invS)
							);
						}
					}
				}
			}

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
		ImGui::SameLine();
		ImGui::Checkbox("ddgi probes", &sShowDdgiProbes);
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
		static constexpr char const * kDebugModes[] = {
			"none",
			"meshlet index",
			"model index",
			"triangle index",
			"mip heatmap",
			"albedo",
			"world normal",
			"roughness",
			"metallic",
			"emissive",
			"uv",
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
		ImGui::SliderFloat("turbidity", &sSkyTurbidity, 1.0f, 10.0f, "%.1f");
		ImGui::DragFloat("sun speed", &sSkyTimeScale, 0.001f, 0.0f, 1.0f, "%.3f");
		ImGui::DragFloat("sun intensity", &sSunIntensity, 0.01f, 0.0f, 10.0f, "%.2f");
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

		// -- entity lists
		if (!singleModelMode) {
			ImGui::Separator();

			// -- instances
			ImGui::SetNextItemAllowOverlap();
			bool const instOpen = ImGui::CollapsingHeader("instances");
			if (instOpen) {
				u32 instDisplayIdx = 0u;
				for (u32 i = 0u; i < (u32)sceneDesc.entities.size(); ++i) {
					auto * ei = std::get_if<scene::EntityInstance>(
						&sceneDesc.entities[i].data
					);
					if (!ei) { continue; }
					ImGui::PushID((int)i);
					std::string const label = scene::entity_label(
						sceneDesc.entities[i], instDisplayIdx
					);
					bool const sel = sSelected.has_value() && *sSelected == i;
					if (sel) { ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive)); }
					if (ImGui::Button(label.c_str())) { sSelected = i; }
					if (sel) { ImGui::PopStyleColor(); }
					ImGui::PopID();
					++instDisplayIdx;
					(void)instDisplayIdx;
				}
			}

			// -- lights
			ImGui::SetNextItemAllowOverlap();
			bool const lightsOpen = ImGui::CollapsingHeader("lights");
			{
				float const addW = (
					ImGui::CalcTextSize("+").x
					+ ImGui::GetStyle().FramePadding.x * 2.0f
				);
				ImGui::SameLine(ImGui::GetContentRegionMax().x - addW);
				if (ImGui::SmallButton("+##addlight")) {
					scene::desc_add_entity(
						sceneDesc,
						scene::EntityLight {
							.color = { 50.0f, 50.0f, 50.0f },
							.radius = 10.0f,
						},
						sFpMode ? fpCam.position : srat::camera_orbit_eye(cam)
					);
				}
			}
			if (lightsOpen) {
				u32 lightDisplayIdx = 0u;
				for (u32 i = 0u; i < (u32)sceneDesc.entities.size(); ++i) {
					if (!std::holds_alternative<scene::EntityLight>(
						sceneDesc.entities[i].data
					)) { continue; }
					ImGui::PushID((int)i);
					std::string const label = scene::entity_label(
						sceneDesc.entities[i], lightDisplayIdx
					);
					bool const sel = sSelected.has_value() && *sSelected == i;
					if (sel) { ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive)); }
					if (ImGui::Button(label.c_str())) { sSelected = i; }
					if (sel) { ImGui::PopStyleColor(); }
					ImGui::PopID();
					++lightDisplayIdx;
				}
			}

			// -- ddgi settings
			if (ImGui::CollapsingHeader("ddgi")) {
				int counts[3] = {
					(int)sDdgi.probeCounts.x,
					(int)sDdgi.probeCounts.y,
					(int)sDdgi.probeCounts.z,
				};
				if (ImGui::DragInt3("probe counts", counts, 1.0f, 1, 64)) {
					sDdgi.probeCounts.x = (u32)std::max(1, counts[0]);
					sDdgi.probeCounts.y = (u32)std::max(1, counts[1]);
					sDdgi.probeCounts.z = (u32)std::max(1, counts[2]);
					sDdgi.dirty = true;
				}
				int cascadeCountI = (int)sDdgi.cascadeCount;
				bool const cascadeDirty = ImGui::DragInt(
					"cascade count",
					&cascadeCountI,
					1.0f, 1, (int)skMaxDdgiCascades
				);
				if (cascadeDirty) {
					sDdgi.cascadeCount = (u32)i32_max(1, cascadeCountI);
					sDdgi.dirty = true;
				}
				ImGui::DragFloat(
					"probe spacing", &sDdgi.probeSpacing, 0.05f, 0.1f, 100.0f
				);
				ImGui::DragFloat(
					"cascade scale", &sDdgi.cascadeScale, 0.1f, 1.1f, 16.0f
				);
				ImGui::Checkbox("freeze probes", &sDdgi.frozen);
			}
		}

		ImGui::End();

		// -- execute deferred entity removal
		if (removeEntityIdx.has_value()) {
			u32 const removeIdx = *removeEntityIdx;
			if (removeIdx < (u32)sceneDesc.entities.size()) {
				sceneDesc.entities.erase(
					sceneDesc.entities.begin() + (ptrdiff_t)removeIdx
				);
			}
		}

		{
			ImGui::Begin("ddgi atlases");
			DdgiVolume & dv = sDdgi.cascades[0].volume;
			if (!dv.irradianceImguiId) {
				dv.irradianceImguiId = (
					vkof::image_imgui_id({
						.image = dv.imageIrradiance,
						.sampler = dv.commonSampler,
					})
				);
				dv.depthImguiId = (
					vkof::image_imgui_id({
						.image = dv.imageDepth,
						.sampler = dv.commonSampler,
					})
				);
			}
			f32 const w = ImGui::GetContentRegionAvail().x;
			ImGui::Text("irradiance");
			ImGui::Image(dv.irradianceImguiId, ImVec2(w, w * 0.5f));
			ImGui::Text("depth");
			ImGui::Image(dv.depthImguiId, ImVec2(w, w * 0.5f));
			ImGui::End();
		}

		{
			ImGui::Begin("temporal");
			if (!sSpecularHistory.normalImguiId) {
				sSpecularHistory.normalImguiId = (
					vkof::image_imgui_id({
						.image = sSpecularHistory.normalImage,
						.sampler = sSpecularHistory.commonSampler,
					})
				);
				for (u32 i = 0u; i < 2u; ++i) {
					sSpecularHistory.specularImguiId[i] = (
						vkof::image_imgui_id({
							.image = sSpecularHistory.specularImage[i],
							.sampler = sSpecularHistory.commonSampler,
						})
					);
					sSpecularHistory.momentImguiId[i] = (
						vkof::image_imgui_id({
							.image = sSpecularHistory.momentImage[i],
							.sampler = sSpecularHistory.commonSampler,
						})
					);
				}
			}
			u32 const readIdx = sFrameIndex % 2u;
			f32 const w = ImGui::GetContentRegionAvail().x;
			f32 const h = w * (f32)kScreenH / (f32)kScreenW;
			ImGui::Text("prevFrameNormal");
			ImGui::Image(sSpecularHistory.normalImguiId, ImVec2(w, h));
			ImGui::Text("prevFrameSpecular");
			ImGui::Image(sSpecularHistory.specularImguiId[readIdx], ImVec2(w, h));
			ImGui::Text("prevFrameMoment");
			ImGui::Image(sSpecularHistory.momentImguiId[readIdx], ImVec2(w, h));
			ImGui::End();
		}

		// -- upload model indirects
		{
			std::vector<GpuResolveModelIndirect> drawDescs;
			for (scene::Entity const & entity : sceneDesc.entities) {
				auto * ei = std::get_if<scene::EntityInstance>(&entity.data);
				if (!ei) { continue; }
				auto const it = loadedModels.find(ei->filename);
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
					.flatIndices = bufs.flatIndices,
					.flatMeshlets = bufs.flatMeshlets,
					.modelMatrix = scene::entity_model_matrix(entity, *ei),
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

		// -- debug draws
		for (scene::Entity const & entity : sceneDesc.entities) {
			if (auto * el = std::get_if<scene::EntityLight>(&entity.data)) {
				vkof::debug_draw_sphere(
					/*center=*/entity.position,
					/*radius=*/el->radius,
					/*color=*/el->color
				);
				vkof::debug_draw_sphere(
					/*center=*/entity.position,
					/*radius=*/0.1f,
					/*color=*/{ 1.0f, 1.0f, 0.2f }
				);
			}
		}
		static std::vector<vkof::DebugSphere> probes;
		probes.clear();
		if (sShowDdgiProbes) {
			f32 probeSpacingScale = 1.0f;
			for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
				f32 const s = sDdgi.probeSpacing * probeSpacingScale;
				f32v3 const origin = sDdgi.cascades[ci].snappedOrigin;
				f32v3 const gridMax = {
					origin.x + s * f32(sDdgi.probeCounts.x),
					origin.y + s * f32(sDdgi.probeCounts.y),
					origin.z + s * f32(sDdgi.probeCounts.z),
				};
				vkof::debug_draw_box(origin, gridMax, { 0.0f, 1.0f, 1.0f });
				f32 const probeRadius = s * 0.2f;
				probeSpacingScale *= sDdgi.cascadeScale;
				for (u32 pz = 0u; pz < sDdgi.probeCounts.z; ++pz)
				for (u32 py = 0u; py < sDdgi.probeCounts.y; ++py)
				for (u32 px = 0u; px < sDdgi.probeCounts.x; ++px) {
					probes.emplace_back(vkof::DebugSphere {
						.position = {
							origin.x + s * ((f32)px + 0.5f),
							origin.y + s * ((f32)py + 0.5f),
							origin.z + s * ((f32)pz + 0.5f),
						},
						.radius = probeRadius,
					});
				}
			}
		}
		if (!probes.empty() && sShowDdgiProbes) {
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
					sDrawToEntityIdx.clear();
					for (u32 entityIdx = 0u; entityIdx < (u32)sceneDesc.entities.size(); ++entityIdx) {
						scene::Entity const & entity = sceneDesc.entities[entityIdx];
						auto * ei = std::get_if<scene::EntityInstance>(&entity.data);
						if (!ei) { continue; }
						auto const it = loadedModels.find(ei->filename);
						if (it == loadedModels.end()) { continue; }
						sDrawToEntityIdx.emplace_back(entityIdx);
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
							.modelMatrix = scene::entity_model_matrix(entity, *ei),
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
					for (u32 di = 0u; di < (u32)sDrawToEntityIdx.size(); ++di) {
						scene::Entity const & entity = (
							sceneDesc.entities[sDrawToEntityIdx[di]]
						);
						auto * ei = std::get_if<scene::EntityInstance>(&entity.data);
						if (!ei) { continue; }
						auto const it = loadedModels.find(ei->filename);
						if (it == loadedModels.end()) { continue; }
						LoadedModel const & model = modelList[it->second];
						tlasInstances.emplace_back(vkof::TlasInstance {
							.blas = model.blas,
							.transform = scene::entity_model_matrix(entity, *ei),
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

			// -- prepare resolve node (declared before ddgi nodes so it can receive persistent image reads)
			vkof::RenderNode const resolveNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);

			// -- ddgi trace node
			vkof::RenderNode const ddgiNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			{
				for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
					DdgiVolume const & dv = sDdgi.cascades[ci].volume;
					vkof::render_node_add_persistent_image({
						.node = ddgiNode,
						.image = dv.imageIrradiance,
						.access = vkof::RenderNodeAccess::write,
					});
					vkof::render_node_add_persistent_image({
						.node = ddgiNode,
						.image = dv.imageDepth,
						.access = vkof::RenderNodeAccess::write,
					});
					vkof::render_node_add_persistent_image({
						.node = resolveNode,
						.image = dv.imageIrradiance,
						.access = vkof::RenderNodeAccess::read,
					});
					vkof::render_node_add_persistent_image({
						.node = resolveNode,
						.image = dv.imageDepth,
						.access = vkof::RenderNodeAccess::read,
					});
				}

				vkof::render_node_callback({
					.node = ddgiNode,
					.callback = [
						gpuCascades,
						&ddgiTraceIrradiancePipeline,
						&ddgiTraceDepthPipeline
					](vkof::CommandBuffer const & cmd) {
						for (u32 ci = 0u; ci < gpuCascades.count; ++ci) {
							GpuDdgiGrid const & tracePC = gpuCascades.grids[ci];
							vkof::cmd_dispatch(vkof::CmdDispatch {
								.cmd = cmd,
								.pipeline = ddgiTraceIrradiancePipeline,
								.pushconstant = srat::slice<u8 const>(
									reinterpret_cast<u8 const *>(&tracePC),
									sizeof(tracePC)
								),
								.groupCountX = tracePC.probeCounts.x,
								.groupCountY = (
									tracePC.probeCounts.y * tracePC.probeCounts.z
								),
								.groupCountZ = 1u,
							});
							vkof::cmd_dispatch(vkof::CmdDispatch {
								.cmd = cmd,
								.pipeline = ddgiTraceDepthPipeline,
								.pushconstant = srat::slice<u8 const>(
									reinterpret_cast<u8 const *>(&tracePC),
									sizeof(tracePC)
								),
								.groupCountX = tracePC.probeCounts.x,
								.groupCountY = (
									tracePC.probeCounts.y * tracePC.probeCounts.z
								),
								.groupCountZ = 1u,
							});
						}
					},
				});
			}

			// -- resolve node
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
			vkof::RenderNode const allNodes[] = {
				drawNode, tlasNode, ddgiNode, resolveNode,
			};
			vkof::render_graph_execute({
				.nodes = srat::slice<vkof::RenderNode const>(allNodes, 4u),
				.rootPushconstant = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(&globalPC), sizeof(globalPC)
				),
				.finalImage = colorTarget,
				.debugDrawViewProj = globalPC.viewProj,
				.debugDrawDepth = depthTarget,
			});
			vkof::render_node_destroy(drawNode);
			vkof::render_node_destroy(tlasNode);
			vkof::render_node_destroy(ddgiNode);
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
	for (u32 ci = 0u; ci < sDdgi.cascadeCount; ++ci) {
		ddgi_volume_destroy(sDdgi.cascades[ci].volume);
	}
	specular_history_destroy(sSpecularHistory);
	vkof::buffer_destroy(modelsIndirectBuffer);
	vkof::buffer_destroy(lightsBuffer);
	vkof::pipeline_destroy(visibilityPipeline);
	vkof::pipeline_destroy(resolvePipeline);
	vkof::pipeline_destroy(debugProbePipeline);
	mor::sampler_cache_destroy();
	vkof::shutdown();
	return 0;
}
