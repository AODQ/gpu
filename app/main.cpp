#include "shaders/scene_shared.h"

#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <srat/camera.hpp>

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <cstdio>

static float sScrollDelta = 0.0f;

int32_t main(int32_t const argc, char const * const * const argv) {
	if (argc < 2) {
		printf("usage: cull <path.gltf>\n");
		return 1;
	}

	vkof::init();

	mor::Scene * const scene = mor::scene_create();
	mor::scene_load_gltf(scene, argv[1]);
	mor::GpuScene const gpuScene    = mor::scene_gpu_upload(scene);
	u32 const meshletCount          = mor::scene_gpu_meshlet_count(gpuScene);
	u32 const instanceCount         = mor::scene_instance_count(scene);
	u32 const vertexCount           = mor::scene_vertex_count(scene);
	mor::Buffers const sceneBufs    = mor::scene_gpu_buffers(gpuScene);
	mor::scene_destroy(scene);

	std::filesystem::path const appDir = (
		std::filesystem::canonical(std::filesystem::path(__FILE__).parent_path())
	);
	std::filesystem::path const shaderDir = appDir / "shaders";
	std::filesystem::path const libDir = appDir.parent_path() / "lib";

	std::string const meshPath = (shaderDir / "scene.mesh").string();
	std::string const fragPath = (shaderDir / "scene.frag").string();
	std::string const libDirStr = libDir.string();

	vkof::TransientImage const colorTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1,
		.isDoubleBuffered = true,
	});
	vkof::TransientImage const depthTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::d24_unorm_s8_uint,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1,
		.isDoubleBuffered = true,
	});

	char const * const kIncludePaths[] = { libDirStr.c_str() };
	static vkof::ImageFormat const kColorFmts[] = {
		vkof::ImageFormat::r8g8b8a8_unorm,
	};
	vkof::Pipeline const pipeline = vkof::pipeline_graphics_create({
		.pathMesh = meshPath.c_str(),
		.pathFragment = fragPath.c_str(),
		.attachmentColorFormats = srat::slice<vkof::ImageFormat const>(kColorFmts, 1),
		.attachmentDepthStencilFormat = vkof::ImageFormat::d24_unorm_s8_uint,
		.depthTest = vkof::DepthTest::write_on_test_on,
		.cullMode = vkof::CullMode::back,
		.blendMode = vkof::BlendMode::none,
		.includePaths = { kIncludePaths, 1 },
	});

	GLFWwindow * const window = vkof::window();

	glfwSetScrollCallback(window, [](GLFWwindow *, double, double yoff) {
		sScrollDelta += (float)yoff;
	});

	srat::CameraOrbit cam = {
		.target    = { 0.0f, 0.0f, 0.0f },
		.distance  = 5.0f,
		.azimuth   = 0.0f,
		.elevation = 0.3f,
		.fovY      = 1.0f,
		.aspect    = 1.0f,
		.near      = 0.01f,
		.far       = 1000.0f,
	};

	double prevMouseX = 0.0;
	double prevMouseY = 0.0;
	glfwGetCursorPos(window, &prevMouseX, &prevMouseY);

	while (!glfwWindowShouldClose(window)) {
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
				srat::camera_orbit_pan(cam, dx * 0.01f * cam.distance, -dy * 0.01f * cam.distance);
			}
			srat::camera_orbit_zoom(cam, -sScrollDelta * 0.1f * cam.distance);
		}
		sScrollDelta = 0.0f;

		int winW, winH;
		glfwGetWindowSize(window, &winW, &winH);
		cam.aspect = (f32)winW / (f32)winH;

		f32m44 const viewProj = (
			srat::camera_orbit_proj(cam)
			* srat::camera_orbit_view(cam)
		);

		GlobalPC const globalPC {
			.time = (f32)glfwGetTime(),
			._pad = {},
		};
		SceneDrawPC const drawPC {
			.meshlets     = sceneBufs.meshlets,
			.instances    = sceneBufs.instances,
			.positions    = sceneBufs.positions,
			.attributes   = sceneBufs.attributes,
			.meshletVerts = sceneBufs.meshletVerts,
			.meshletTris  = sceneBufs.meshletTris,
			.viewProj     = viewProj,
		};

		vkof::imgui_begin();
		ImGui::Begin("scene");
		ImGui::Text("fps:       %.1f", ImGui::GetIO().Framerate);
		ImGui::Text("instances: %u",   instanceCount);
		ImGui::Text("meshlets:  %u",   meshletCount);
		ImGui::Text("vertices:  %u",   vertexCount);
		if (ImGui::Button("reset camera")) {
			cam.target    = { 0.0f, 0.0f, 0.0f };
			cam.distance  = 5.0f;
			cam.azimuth   = 0.0f;
			cam.elevation = 0.3f;
		}
		ImGui::End();

		vkof::RenderNode const drawNode = vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
		});
		static f32 const kClearColor[] = { 0.05f, 0.05f, 0.1f, 1.0f };
		static f32 const kDepthClear   = 1.0f;
		vkof::render_node_attachment_color({
			.node       = drawNode,
			.image      = colorTarget,
			.loadOp     = vkof::RenderNodeLoadOp::clear,
			.mipLevel   = 0,
			.colorIndex = 0,
			.clearColor = srat::slice<f32 const>(kClearColor, 4),
		});
		vkof::render_node_attachment_depth({
			.node       = drawNode,
			.image      = depthTarget,
			.loadOp     = vkof::RenderNodeLoadOp::clear,
			.mipLevel   = 0,
			.clearDepth = srat::slice<f32 const>(&kDepthClear, 1),
		});
		vkof::render_node_callback({
			.node     = drawNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				vkof::cmd_draw({
					.cmd          = cmd,
					.pipeline     = pipeline,
					.pushconstant = srat::slice<u8 const>(
						reinterpret_cast<u8 const *>(&drawPC), sizeof(drawPC)
					),
					.vertexCount   = meshletCount,
					.instanceCount = 1,
				});
			},
		});

		vkof::RenderNode const nodes[] = { drawNode };
		vkof::render_graph_execute({
			.nodes            = srat::slice<vkof::RenderNode const>(nodes, 1),
			.rootPushconstant = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&globalPC), sizeof(globalPC)
			),
			.finalImage = colorTarget,
		});

		vkof::render_node_destroy(drawNode);
	}

	vkof::device_wait_idle();
	mor::scene_gpu_destroy(gpuScene);
	vkof::pipeline_destroy(pipeline);
	vkof::shutdown();
	return 0;
}
