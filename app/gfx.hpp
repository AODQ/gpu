#pragma once

#include <cstdint>
#include <cmath>

#include <filesystem>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using isize = std::size_t;
using usize = std::size_t;
using f32 = float;
using f64 = double;

#define VkAssert(x) { \
	if ((x != VK_SUCCESS)) { \
		printf("assertion failed: %s [%d]\n", #x, x); \
		std::abort(); \
	} \
}

struct GLFWwindow;
struct ImGuiContext;

namespace gfx {

struct Device {
	GLFWwindow * window;

	vkb::Instance instance;
	vkb::PhysicalDevice physicalDevice;
	vkb::Swapchain swapchain;
	std::vector<VkImage> swapchainDepthImages;
	std::vector<VkImageView> swapchainDepthImageViews;
	VkSurfaceKHR surface;
	vkb::Device device;

	VmaAllocator allocator;

	[[nodiscard]] inline VkDevice vkDevice() const { return device; }

	VkSampler samplerNearest;

	VkQueue graphicsQueue;
	u32 graphicsQueueFamily;
};

Device device_init();

VkShaderModule shader_load(
	gfx::Device const & device,
	std::filesystem::path const & path
);

struct Pipeline {
	VkPipeline pipeline;
	VkPipelineLayout layout;
	VkShaderModule shaderMesh;
	VkShaderModule shaderFrag;
	std::filesystem::file_time_type lastWriteTimeMesh;
	std::filesystem::file_time_type lastWriteTimeFrag;
};

struct PipelineCompute {
	VkPipeline pipeline;
	VkPipelineLayout layout;
	VkShaderModule shaderComp;
	std::filesystem::file_time_type lastWriteTimeComp;
};

Pipeline pipeline_create(
	gfx::Device const & device,
	VkPipelineLayout const pipelineLayout,
	std::filesystem::path const & pathMeshOrVert,
	std::filesystem::path const & pathFrag,
	bool const alphaBlending = false,
	bool const isMeshPipeline = false,
	Pipeline const * hotReloadPipeline = nullptr
);

PipelineCompute pipeline_compute_create(
	gfx::Device const & device,
	VkPipelineLayout const pipelineLayout,
	std::filesystem::path const & pathComp,
	PipelineCompute const * hotReloadPipeline = nullptr
);

std::string const & last_pipline_compile_error();

struct CommandPool {
	VkCommandPool pool;
};

CommandPool command_pool_create(gfx::Device const & device);

struct Frame {
	VkCommandBuffer commandBuffer;
	VkSemaphore semaphoreSwapchainImageAvailable;
	VkSemaphore semaphoreRenderFinished;
	VkFence fenceCommandBufferInFlight;
};

Frame frame_create(
	gfx::Device const & device,
	CommandPool const & commandPool
);

struct Buffer {
	VkBuffer buffer;
	VmaAllocation allocation;
	VkDeviceAddress virtualAddress;
};

Buffer buffer_create(
	gfx::Device const & device,
	VkDeviceSize const size,
	VkBufferUsageFlags const usage,
	VmaMemoryUsage const memoryUsage
);

Buffer buffer_create_with_data(
	gfx::Device const & device,
	VkDeviceSize const size,
	VkBufferUsageFlags const usage,
	VmaMemoryUsage const memoryUsage,
	VkCommandPool const & commandPool,
	void const * const data,
	VkDeviceSize const dataSize
);

struct Image {
	VkImage handle;
	VmaAllocation allocation;
	VkFormat format;
	VkExtent2D extent;
	u32 mipLevels;
};

Image image_create(
	gfx::Device const & device,
	VkExtent2D const extent,
	VkFormat const format,
	VkImageUsageFlags const usage,
	VmaMemoryUsage const memoryUsage,
	u32 const mipLevels = 1
);
void image_destroy(
	gfx::Device const & device,
	Image const & image
);

std::vector<VkImageView> image_create_mip_views(
	gfx::Device const & device,
	Image const & image,
	VkComponentSwizzle const channelSwizzle = VK_COMPONENT_SWIZZLE_IDENTITY
);


// -- imgui
void imgui_init(gfx::Device const & device);
void imgui_shutdown();
void imgui_new_frame();
void imgui_render(VkCommandBuffer commandBuffer);

}

// -- math --

struct f32m44 {
	f32 m[16];

	[[nodiscard]] inline f32m44 operator*(f32m44 const & other) const {
		f32m44 result {};
		for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
		for (int k = 0; k < 4; ++k) {
			result.m[i*4 + j] += m[k*4 + j] * other.m[i*4 + k];
		};
		return result;
	}
};

struct f32v3 {
	f32 x, y, z;

	[[nodiscard]] f32v3 operator-(f32v3 const & other) const {
		return { x - other.x, y - other.y, z - other.z };
	}

	[[nodiscard]] f32v3 operator+(f32v3 const & other) const {
		return { x + other.x, y + other.y, z + other.z };
	}

	[[nodiscard]] f32v3 operator*(f32 const scalar) const {
		return { x * scalar, y * scalar, z * scalar };
	}
};

inline f32 length(f32v3 const v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

struct f32v4 {
	f32 x, y, z, w;
};

inline f32v3 normalize(f32v3 const v) {
	f32 len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	return { v.x/len, v.y/len, v.z/len };
}
