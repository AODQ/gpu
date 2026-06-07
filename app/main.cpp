#include "gfx.hpp"

#include "cull.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <cmath>
#include <filesystem>

inline f32v3 cross(f32v3 const a, f32v3 const b) {
	return {
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x
	};
}

inline f32 dot(f32v3 const a, f32v3 const b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline f32m44 lookat(f32v3 const & eye, f32v3 const & center, f32v3 const & up) {
	f32v3 const f = normalize(center - eye);
	f32v3 const s = normalize(cross(f, up));
	f32v3 const u = cross(s, f);

	f32m44 result {};
	result.m[0] = s.x;
	result.m[4] = s.y;
	result.m[8] = s.z;
	result.m[12] = -dot(s, eye);
	result.m[1] = u.x;
	result.m[5] = u.y;
	result.m[9] = u.z;
	result.m[13] = -dot(u, eye);
	result.m[2] = -f.x;
	result.m[6] = -f.y;
	result.m[10] = -f.z;
	result.m[14] = dot(f, eye);
	result.m[3] = 0.0f;
	result.m[7] = 0.0f;
	result.m[11] = 0.0f;
	result.m[15] = 1.0f;

	return result;
}

inline f32m44 perspective(f32 fovY, f32 aspect, f32 near, f32 far) {
	f32 const tanHalfFovY = tanf(fovY / 2.0f);

	f32m44 result {};
	result.m[0] = 1.0f / (aspect * tanHalfFovY);
	result.m[5] = 1.0f / tanHalfFovY;
	result.m[10] = -(far + near) / (far - near);
	result.m[11] = -1.0f;
	result.m[14] = -(2.0f * far * near) / (far - near);
	return result;
}

struct Camera {
	f32v3 target { 0.0f, -1.0f, 0.0f };
	f32 distance { 10.0f };
	f32 yaw { 0.0f };
	f32 pitch { 0.0f };
	f32 lastMouseX { 0.0f };
	f32 lastMouseY { 0.0f };
	f32 scrollDelta { 0.0f };
};

void glfwScrollCallback(
	GLFWwindow * const window,
	[[maybe_unused]] double const xoffset,
	double const yoffset
) {
	Camera * camera = (Camera *)glfwGetWindowUserPointer(window);
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	camera->scrollDelta = (f32)yoffset;
}

f32m44 cameraUpdate(
	GLFWwindow * const window,
	Camera & camera,
	f32v3 & cameraForward,
	f32v3 & cameraPosWorld
) {
	double mouseX, mouseY;
	glfwGetCursorPos(window, &mouseX, &mouseY);

	f32 const deltaX = (f32)(mouseX - camera.lastMouseX);
	f32 const deltaY = (f32)(mouseY - camera.lastMouseY);
	camera.lastMouseX = (f32)mouseX;
	camera.lastMouseY = (f32)mouseY;

	if (!ImGui::GetIO().WantCaptureMouse) {
		// left click: orbit
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
			camera.yaw += deltaX * 0.01f;
			camera.pitch += deltaY * 0.01f;
			camera.pitch = std::clamp(camera.pitch, -1.5f, 1.5f);
		}

		// right click: pan
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
			f32v3 right { cosf(camera.yaw), 0.0f, sinf(camera.yaw) };
			f32v3 up { 0.0f, -1.0f, 0.0f };
			camera.target.x += right.x * deltaX * 0.02f;
			camera.target.z -= right.z * deltaX * 0.02f;
			camera.target.y -= up.y * deltaY * 0.02f;
		}

		// scroll: zoom
		camera.distance -= (f32)camera.scrollDelta * 0.5f;
		camera.distance = std::clamp(camera.distance, 1.0f, 100.0f);
		camera.scrollDelta = 0.0f;
	}

	f32 cosYaw = cosf(camera.yaw);
	f32 sinYaw = sinf(camera.yaw);
	f32 cosPitch = cosf(camera.pitch);
	f32 sinPitch = sinf(camera.pitch);

	f32v3 eye {
		camera.target.x + camera.distance * cosPitch * sinYaw,
		camera.target.y + camera.distance * sinPitch,
		camera.target.z + camera.distance * cosPitch * cosYaw
	};

	cameraForward = normalize(camera.target - eye);
	cameraPosWorld = eye;

	return lookat(eye, camera.target, { 0.0f, -1.0f, 0.0f });
}

int32_t main() {
	gfx::Device context = gfx::device_init();
	VkDevice const device = context.vkDevice();
	gfx::imgui_init(context);

	// // -- load first pipeline
	// auto const pipelineLayout = [&]() -> VkPipelineLayout {
	// 	VkPushConstantRange pushConstantRange {
	// 		.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
	// 		.offset = 0,
	// 		.size = sizeof(f32)*4*4*2 + sizeof(f32) + sizeof(u32),
	// 	};
	// 	VkPipelineLayoutCreateInfo const pipelineLayoutCreateInfo {
	// 		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	// 		.pNext = nullptr,
	// 		.flags = 0,
	// 		.setLayoutCount = 0,
	// 		.pSetLayouts = nullptr,
	// 		.pushConstantRangeCount = 1,
	// 		.pPushConstantRanges = &pushConstantRange,
	// 	};
	// 	VkPipelineLayout pipelineLayout;
	// 	VkAssert(
	// 		vkCreatePipelineLayout(
	// 			device,
	// 			&pipelineLayoutCreateInfo,
	// 			nullptr,
	// 			&pipelineLayout
	// 		)
	// 	);
	// 	return pipelineLayout;
	// }();

	// std::optional<gfx::Pipeline> pipelineGrass;
	// static int reloadPopup { 0 };
	// std::string errorPipelineGrass;
	// auto const fnReloadPipeline = [&]() -> void {
	// 	pipelineGrass.emplace(
	// 		gfx::pipeline_create(
	// 			context,
	// 			pipelineLayout,
	// 			pwd_dir() / "shaders/triangle.mesh.glsl",
	// 			pwd_dir() / "shaders/triangle.frag.glsl",
	// 			/*alphaBlending=*/ false,
	// 			(pipelineGrass.has_value() ? &pipelineGrass.value() : nullptr)
	// 		)
	// 	);
	// 	errorPipelineGrass = gfx::last_pipline_compile_error();
	// 	pipelinePlane.emplace(
	// 		gfx::pipeline_create(
	// 			context,
	// 			pipelineLayout,
	// 			pwd_dir() / "shaders/plane.mesh.glsl",
	// 			pwd_dir() / "shaders/triangle.frag.glsl",
	// 			/*alphaBlending=*/ true,
	// 			(pipelinePlane.has_value() ? &pipelinePlane.value() : nullptr)
	// 		)
	// 	);
	// 	errorPipelinePlane = gfx::last_pipline_compile_error();
	// 	reloadPopup = 120;
	// };
	// fnReloadPipeline();

	// -- command pool and per-frame data
	gfx::CommandPool const commandPool = gfx::command_pool_create(context);
	std::vector<gfx::Frame> perFrame;
	perFrame.resize(context.swapchain.requested_min_image_count);
	for (auto & frame : perFrame) {
		frame = gfx::frame_create(context, commandPool);
	}
	auto const vkCmdDrawMeshTasksEXT = (
		(PFN_vkCmdDrawMeshTasksEXT)
		vkGetDeviceProcAddr(
			device,
			"vkCmdDrawMeshTasksEXT"
		)
	);

	Cull::init(context, commandPool.pool);

	// generate lookat / perspective matrices for the push constant
	Camera camera {};
	glfwSetWindowUserPointer(context.window, &camera);
	glfwSetScrollCallback(context.window, glfwScrollCallback);
	i32 frameIdx { 0 };
	f32 frameTime = 0.0f;
	f64 lastTime = glfwGetTime();
	while (!glfwWindowShouldClose(context.window)) {
		glfwPollEvents();

		if (glfwGetKey(context.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(context.window, true);
		}

		f32v3 cameraForward;
		f32v3 cameraPosWorld;
		f32m44 const view = (
			cameraUpdate(context.window, camera, cameraForward, cameraPosWorld)
		);
		f32m44 const proj = perspective(
			90.0f * (3.14159265f / 180.0f),
			1.77777f,
			0.1f,
			10000.0f
		);
		f32m44 const viewProj = proj * view;

		gfx::Frame & frame = perFrame[frameIdx % perFrame.size()];

		// -- wait for previous in-flight frame to finish
		vkWaitForFences(
			device,
			1,
			&frame.fenceCommandBufferInFlight,
			/*waitAll=*/ VK_TRUE,
			/*timeout=*/ 1u * 1000u * 1000u
		);
		vkResetFences(device, 1, &frame.fenceCommandBufferInFlight);

		// -- acquire swapchain image
		u32 imageIdx;
		vkAcquireNextImageKHR(
			device,
			context.swapchain,
			/*timeout=*/ UINT64_MAX,
			frame.semaphoreSwapchainImageAvailable,
			/*fence=*/ VK_NULL_HANDLE,
			&imageIdx
		);

		// -- record command buffer
		vkResetCommandBuffer(frame.commandBuffer, /*flags=*/ 0);
		VkCommandBufferBeginInfo beginInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = 0,
			.pInheritanceInfo = nullptr,
		};
		vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

		// -- setup image
		VkImageView const imageView = (
			context.swapchain.get_image_views().value()[imageIdx]
		);

		// layout transition
		VkImageMemoryBarrier imageBarrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = context.swapchain.get_images().value()[imageIdx],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			frame.commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageBarrier
		);

		VkImageMemoryBarrier const depthImageBarrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = context.swapchainDepthImages[imageIdx],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			frame.commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &depthImageBarrier
		);

		// -- setup render

		{
			f32m44 const cullProj = (
				perspective(
				45.0f * (3.14159265f / 180.0f),
				1.77777f,
				0.1f,
				10000.0f
				)
			);
			f32m44 const cullViewProj = cullProj * view;
			Cull::update(
				context, frame.commandBuffer,
				/*viewProj=*/ cullViewProj,
				/*cameraForward=*/ cameraForward,
				/*cameraPosWorld=*/ cameraPosWorld
			);
		}

		{
			VkRenderingAttachmentInfo const colorAttachment {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = context.swapchain.get_image_views().value()[imageIdx],
				.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = VkClearValue { .color = { 0.2f, 0.2f, 0.8f, 1.0f } },
			};

			VkRenderingAttachmentInfo const depthAttachment {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = context.swapchainDepthImageViews[imageIdx],
				.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = VkClearValue { .depthStencil = { 1.0f, 0 } },
			};
			VkRenderingInfo renderingInfo {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.pNext = nullptr,
				.flags = 0,
				.renderArea = VkRect2D {
					.offset = { 0, 0 },
					.extent = context.swapchain.extent,
				},
				.layerCount = 1,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachments = &colorAttachment,
				.pDepthAttachment = &depthAttachment,
				.pStencilAttachment = nullptr,
			};
			vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);
		}

		{ // -- imgui
			gfx::imgui_new_frame();
			ImGui::Begin("info");
			ImGui::Text("fps: %.2f", ImGui::GetIO().Framerate);
			ImGui::End();
		}

		{
			ImGui::Begin("HiZ");
			ImGui::Text(
				"visible instances: %u / %u",
				Cull::lastVisibleCount(),
				Cull::totalInstanceCount()
			);
			ImGui::End();
		}

		{
			ImGui::Begin("shader compile output");
			// if (!errorPipelineGrass.empty()) {
			// 	ImGui::Text("Grass pipeline error:");
			// 	ImGui::TextWrapped("%s", errorPipelineGrass.c_str());
			// } else {
			// 	ImGui::Text("Grass pipeline compiled successfully.");
			// }
			ImGui::End();
		}

		VkViewport viewport {
			.x = 0,
			.y = 0,
			.width = (f32)context.swapchain.extent.width,
			.height = (f32)context.swapchain.extent.height,
			.minDepth = 0,
			.maxDepth = 1,
		};
		vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);
		{
			auto const scissor = VkRect2D {
				.offset = { 0, 0 },
				.extent = context.swapchain.extent,
			};
			vkCmdSetScissor(
				frame.commandBuffer,
				/*firstScissor=*/ 0,
				/*scissorCount=*/ 1,
				/*pScissors=*/ &scissor
			);
		}

		// dispatch mesh shader workgroup
		// {
		// 	vkCmdBindPipeline(
		// 		frame.commandBuffer,
		// 		VK_PIPELINE_BIND_POINT_GRAPHICS,
		// 		pipelineGrass->pipeline
		// 	);
		// 	vkCmdPushConstants(
		// 		frame.commandBuffer,
		// 		pipelineGrass->layout,
		// 		VK_SHADER_STAGE_MESH_BIT_EXT,
		// 		/*offset=*/ 0,
		// 		sizeof(viewProj),
		// 		&viewProj
		// 	);
		// 	f32 const time = (f32)glfwGetTime();
		// 	vkCmdPushConstants(
		// 		frame.commandBuffer,
		// 		pipelineGrass->layout,
		// 		VK_SHADER_STAGE_MESH_BIT_EXT,
		// 		/*offset=*/ sizeof(viewProj),
		// 		sizeof(time),
		// 		&time
		// 	);
		// 	static int chunksPerAxisX = 32;
		// 	static int chunksPerAxisY = 32;
		// 	static int chunksPerAxisZ = 32;
		// 	ImGui::Text("Chunks per axis:");
		// 	ImGui::SliderInt("##chunksPerAxisX", &chunksPerAxisX, 1, 256);
		// 	ImGui::SliderInt("##chunksPerAxisY", &chunksPerAxisY, 1, 64);
		// 	ImGui::SliderInt("##chunksPerAxisZ", &chunksPerAxisZ, 1, 256);
		// 	u32 chunksPerAxis[3] = {
		// 		(u32)chunksPerAxisX,
		// 		(u32)chunksPerAxisY,
		// 		(u32)chunksPerAxisZ,
		// 	};
		// 	vkCmdPushConstants(
		// 		frame.commandBuffer,
		// 		pipelineGrass->layout,
		// 		VK_SHADER_STAGE_MESH_BIT_EXT,
		// 		/*offset=*/ sizeof(viewProj) + sizeof(time) + sizeof(u32)*3,
		// 		sizeof(chunksPerAxis),
		// 		chunksPerAxis
		// 	);
		// 	u32 const dispatchCount = (
		// 		chunksPerAxisX * chunksPerAxisY * chunksPerAxisZ
		// 	);
		// 	vkCmdDrawMeshTasksEXT(frame.commandBuffer, dispatchCount, 1, 1);
		// }

		// if (reloadPopup > 0) {
		// 	ImGui::Text("Reloaded shaders!");
		// 	--reloadPopup;
		// }

		Cull::draw(
			context,
			frame.commandBuffer,
			/*viewProj=*/ viewProj,
			/*cameraForward=*/ cameraForward,
			/*cameraPosWorld=*/ cameraPosWorld
		);
		vkCmdEndRendering(frame.commandBuffer);

		Cull::resolveDepth(
			context,
			frame.commandBuffer,
			context.swapchainDepthImages[imageIdx],
			context.swapchainDepthImageViews[imageIdx]
		);

		// -- transition depth image for debug presentation
		Cull::imageHizTransition(frame.commandBuffer);
		static std::vector<VkDescriptorSet> imguiDescriptorSet {};
		if (imguiDescriptorSet.size() == 0) {
			auto const & imageViews = Cull::imageHiz();
			for (auto const & iv : imageViews) {
				imguiDescriptorSet.push_back(
					ImGui_ImplVulkan_AddTexture(
						context.samplerNearest,
						iv,
						VK_IMAGE_LAYOUT_GENERAL
					)
				);
			}
		}

		ImGui::Begin("HiZ debug");
		for (size_t i = 0; i < imguiDescriptorSet.size(); ++i) {
			ImGui::Text("Mip level %zu", i);
			ImGui::Image(
				imguiDescriptorSet[i],
				ImVec2(
					640.0f*0.5,
					480.0f*0.5
				)
			);
		}
		ImGui::End();

		{
			VkRenderingAttachmentInfo const colorAttachment {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = context.swapchain.get_image_views().value()[imageIdx],
				.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {},
			};
			VkRenderingAttachmentInfo const depthAttachment {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = context.swapchainDepthImageViews[imageIdx],
				.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {},
			};

			VkRenderingInfo renderingInfo {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.pNext = nullptr,
				.flags = 0,
				.renderArea = VkRect2D {
					.offset = { 0, 0 },
					.extent = context.swapchain.extent,
				},
				.layerCount = 1,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachments = &colorAttachment,
				.pDepthAttachment = &depthAttachment,
				.pStencilAttachment = nullptr,
			};
			vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);
		}

		{ // -- imgui end
			ImGui::Render();
			gfx::imgui_render(frame.commandBuffer);
		}
		vkCmdEndRendering(frame.commandBuffer);

		// -- transition image for presentation

		VkImageMemoryBarrier const presentBarrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = 0,
			.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = context.swapchain.get_images().value()[imageIdx],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			frame.commandBuffer,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &presentBarrier
		);

		VkAssert(vkEndCommandBuffer(frame.commandBuffer));

		// -- submit

		VkPipelineStageFlags const waitStage = (
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		);

		VkSubmitInfo const submit {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &frame.semaphoreSwapchainImageAvailable,
			.pWaitDstStageMask = &waitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &frame.commandBuffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &frame.semaphoreRenderFinished,
		};
		VkAssert(
			vkQueueSubmit(
				context.graphicsQueue,
				1,
				&submit,
				frame.fenceCommandBufferInFlight
			)
		);

		VkPresentInfoKHR const present {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &frame.semaphoreRenderFinished,
			.swapchainCount = 1,
			.pSwapchains = &context.swapchain.swapchain,
			.pImageIndices = &imageIdx,
			.pResults = nullptr,
		};
		VkAssert(vkQueuePresentKHR(context.graphicsQueue, &present));

		++frameIdx;

		// -- hot reload pipeline if shaders have changed
		// auto const meshShaderWriteTime = (
		// 	std::filesystem::last_write_time(
		// 		pwd_dir() / "shaders/triangle.mesh.glsl"
		// 	)
		// );
		// if (
		// 	   meshShaderWriteTime != pipelineGrass->lastWriteTimeMesh
		// ) {
		// 	printf("reloading shader timestamp: %.2fs\n", glfwGetTime());
		// 	vkDeviceWaitIdle(device);
		// 	fnReloadPipeline();
		// }
	}

	vkDeviceWaitIdle(device);

	Cull::shutdown(context);

	for (auto & frame : perFrame) {
		vkDestroySemaphore(
			device, frame.semaphoreSwapchainImageAvailable, nullptr
		);
		vkDestroySemaphore(device, frame.semaphoreRenderFinished, nullptr);
		vkDestroyFence(device, frame.fenceCommandBufferInFlight, nullptr);
	}

	vkDestroyCommandPool(device, commandPool.pool, nullptr);
	for (auto fb : context.swapchain.get_image_views().value()) {
		vkDestroyImageView(device, fb, nullptr);
	}
	vkb::destroy_swapchain(context.swapchain);
	// vkDestroyPipeline(device, pipelineGrass->pipeline, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(context.instance, context.surface, nullptr);
	vkb::destroy_instance(context.instance);
	glfwDestroyWindow(context.window);
	glfwTerminate();


	return 0;
}

