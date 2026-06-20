#include <vkof/vkof.hpp>
#include <srat/core-handle.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <stb_image_write.h>

#include <chrono>
#include <unordered_map>
#include <filesystem>

#define VkAssert(x) { \
	if ((x != VK_SUCCESS)) { \
		printf("assertion failed: %s [%d]\n", #x, x); \
		std::abort(); \
	} \
}

// -----------------------------------------------------------------------------
// -- private forwards
// -----------------------------------------------------------------------------

namespace {
	struct ImplTexture;
	struct ImplPipelineGraphics;
	struct ImplPipelineCompute;
	struct CompileResult;

	enum struct ShaderStage { fragment, compute, mesh };
	struct CompileResult {
		int exitCode;
		std::string output;
	};

	struct ImplCommandBuffer
	{
		VkCommandBuffer cmd;
		vkof::Pipeline boundPipeline;
		VkPipelineBindPoint bindPoint;
	};
}

static std::filesystem::file_time_type file_write_time(std::string const & p);
static VkImageView image_storage_view_for_mip(
	ImplTexture const & implTexture, u32 mipLevel
);
static char const * shader_stage_flag(ShaderStage const & stage);
static CompileResult system_capture(char const * cmd);
static VkShaderModule compile_and_load_shader_module(
	std::string const & glslPath,
	ShaderStage stage,
	std::vector<std::string> const & defines,
	std::vector<std::string> const & includePaths
);
static bool file_changed(
	std::string const & path,
	std::filesystem::file_time_type const & lastWriteTime
);
static void pipeline_hot_reload();
static void profiler_init();

static VkFormat to_vk_format(vkof::ImageFormat format) {
	switch (format) {
		case vkof::ImageFormat::r8g8b8a8_unorm:
			return VK_FORMAT_R8G8B8A8_UNORM;
		case vkof::ImageFormat::r8g8b8a8_srgb:
			return VK_FORMAT_R8G8B8A8_SRGB;
		case vkof::ImageFormat::r16g16b16a16_sfloat:
			return VK_FORMAT_R16G16B16A16_SFLOAT;
		case vkof::ImageFormat::r32_float:
			return VK_FORMAT_R32_SFLOAT;
		case vkof::ImageFormat::d24_unorm_s8_uint:
			return VK_FORMAT_D24_UNORM_S8_UINT;
		default:
			return VK_FORMAT_UNDEFINED;
	}
}

static VkCullModeFlagBits to_vk_cull(vkof::CullMode cullMode) {
	switch (cullMode) {
		case vkof::CullMode::front:
			return VK_CULL_MODE_FRONT_BIT;
		case vkof::CullMode::back:
			return VK_CULL_MODE_BACK_BIT;
		default:
			return VK_CULL_MODE_NONE;
	}
}

static void depth_test_flags(
	vkof::DepthTest const writeOnTestOff,
	VkBool32 & outDepthTestEnable,
	VkBool32 & outDepthWriteEnable
) {
	switch (writeOnTestOff) {
		case vkof::DepthTest::write_off_test_off:
			outDepthTestEnable = VK_FALSE;
			outDepthWriteEnable = VK_FALSE;
		break;
		case vkof::DepthTest::write_on_test_off:
			outDepthTestEnable = VK_FALSE;
			outDepthWriteEnable = VK_TRUE;
		break;
		case vkof::DepthTest::write_off_test_on:
			outDepthTestEnable = VK_TRUE;
			outDepthWriteEnable = VK_FALSE;
		break;
		case vkof::DepthTest::write_on_test_on:
			outDepthTestEnable = VK_TRUE;
			outDepthWriteEnable = VK_TRUE;
		break;
	}
}

namespace
{
	// derive the aspect mask from a format (color vs depth)
	VkImageAspectFlags format_aspect(VkFormat const format) {
		switch (format) {
			case VK_FORMAT_D32_SFLOAT:
			case VK_FORMAT_D16_UNORM:
				return VK_IMAGE_ASPECT_DEPTH_BIT;
			case VK_FORMAT_D24_UNORM_S8_UINT:
			case VK_FORMAT_D32_SFLOAT_S8_UINT:
				return (
					VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
				);
			default:
				return VK_IMAGE_ASPECT_COLOR_BIT;
		}
	}

	// usage depends on whether the format is depth or color
	VkImageUsageFlags format_usage(VkFormat const format) {
		VkImageUsageFlags usage = (
			  VK_IMAGE_USAGE_SAMPLED_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
			| VK_IMAGE_USAGE_TRANSFER_DST_BIT
		);
		bool const isDepth = (
			format_aspect(format) & VK_IMAGE_ASPECT_DEPTH_BIT
		);
		if (isDepth) {
			usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		} else {
			usage |= (
				  VK_IMAGE_USAGE_STORAGE_BIT
				| VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
			);
		}
		return usage;
	}

	struct ImplTexture
	{
		VkImage image;
		VmaAllocation allocation;
		VkImageView imageViewFull;
		std::vector<VkImageView> imageViewPerMip;
		u32 width;
		u32 height;
	};

	struct ImplBuffer
	{
		VkBuffer buffer;
		VmaAllocation allocation;
		u64 deviceAddress;  // BDA, present for every buffer
		void * mapped;      // host pointer, HostWritable only (else null)
		u64 byteCount;      // size of the allocation in bytes
	};

	struct ImplSampler
	{
		VkSampler sampler;
	};

	struct ImplTransientImage {
		vkof::ImageFormat format;
		f32 scaleWidth;
		f32 scaleHeight;
		u32 mipLevels;
		bool isDoubleBuffered;
		// pre-allocated image handles (index 0 always valid; index 1 only if
		// isDoubleBuffered).  stored in imagePool so all image_* helpers work.
		std::array<vkof::Image, 2> handles;
	};

	struct ImplTransientBuffer {
		bool isDoubleBuffered;
		// pre-allocated buffer handles (index 1 only if isDoubleBuffered)
		std::array<vkof::Buffer, 2> handles;
	};

	struct ImplRenderBarrier
	{
		vkof::TransientImage transientImage;
		vkof::RenderNodeAccess access;
	};

	struct ImplBufferBarrier
	{
		vkof::TransientBuffer transientBuffer;
		vkof::RenderNodeAccess access;
	};

	struct ImplColorAttachment
	{
		vkof::TransientImage image;
		vkof::RenderNodeLoadOp loadOp;
		u32 mipLevel;
		u32 colorIndex;
		f32 clearColor[4];
	};

	struct ImplDepthAttachment
	{
		vkof::TransientImage image;
		vkof::RenderNodeLoadOp loadOp;
		u32 mipLevel;
		f32 clearDepth;
	};

	struct ImplRenderNode
	{
		vkof::CommandQueue queue;
		std::vector<ImplRenderBarrier> imageBarriers;
		std::vector<ImplBufferBarrier> bufferBarriers;
		std::vector<ImplColorAttachment> colorAttachments;
		std::optional<ImplDepthAttachment> depthAttachment;
		std::function<void(vkof::CommandBuffer const &)> callback;
	};

	struct ImplPipelineGraphics
	{
		VkPipeline pipeline;
		std::vector<vkof::ImageFormat> colorAttachmentFormats;
		vkof::ImageFormat depthAttachmentFormat;
		vkof::DepthTest depthTest;
		vkof::CullMode cullMode;
		vkof::BlendMode blendMode;

		std::string pathFragment;
		std::string pathMesh;
		std::vector<std::string> defines;
		std::vector<std::string> includePaths;

		std::filesystem::file_time_type lastWriteTimeFragment;
		std::filesystem::file_time_type lastWriteTimeMesh;
	};

	struct ImplPipelineCompute
	{
		VkPipeline pipeline;
		VkPipelineLayout layout;
		std::string pathCompute;
		std::vector<std::string> defines;
		std::vector<std::string> includePaths;
		std::filesystem::file_time_type lastWriteTimeCompute;
	};

	struct PerFrameData {
		VkCommandBuffer cmdGraphics;
		VkSemaphore semaphoreAcquire;
		VkFence fence;
	};

	static constexpr u32 kFramesInFlight = 2;
	static constexpr u32 kMaxProfiledNodes = 16;

	struct NodeTiming {
		double cpuMs { 0.0 };
		double gpuMs { 0.0 };
	};

	struct Profiler {
		VkQueryPool queryPool { VK_NULL_HANDLE };
		double timestampPeriodMs { 0.0 };
		bool gpuSupported { false };
		// how many nodes ran in the last use of each frame slot (for readback)
		std::array<u32, kFramesInFlight> slotNodeCounts {};
		// stable display data: gpu updated on readback, cpu updated each frame
		std::array<NodeTiming, kMaxProfiledNodes> timings {};
		u32 displayCount { 0 };
		u32 frameNodeCount { 0 };
	};

	struct ImplDevice {
		GLFWwindow * window;
		vkb::Instance instance;
		vkb::PhysicalDevice physicalDevice;
		vkb::Device device;
		vkb::Swapchain swapchain;
		std::vector<VkImage> swapchainImages;
		std::vector<VkImageView> swapchainImageViews;
		std::vector<VkImage> swapchainDepthImages;
		std::vector<VkImageView> swapchainDepthImageViews;
		std::vector<VmaAllocation> swapchainDepthAllocs;
		// non-null only in headless mode; holds the fake swapchain image allocations
		VmaAllocation headlessSwapchainAlloc { VK_NULL_HANDLE };
		VmaAllocation headlessDepthAlloc { VK_NULL_HANDLE };

		u32 swapchainImageIndex;

		VkCommandPool commandPoolGraphics;
		VkCommandPool commandPoolCompute;

		VkPipelineLayout pipelineLayoutUniversal;

		VkSurfaceKHR surface;
		VkQueue queueGraphics;
		VkQueue queueCompute;
		VmaAllocator allocator;

		// per-frame-in-flight sync primitives
		std::array<PerFrameData, kFramesInFlight> frameData;
		u32 frameIndex { 0 };
		// per-swapchain-image render semaphores (must not be per-frame-slot;
		// reusing before WSI releases them triggers VUID-vkQueueSubmit2-semaphore-03868)
		std::vector<VkSemaphore> semaphoresRender;

		srat::HandlePool<vkof::Image, ImplTexture> imagePool;
		srat::HandlePool<vkof::Buffer, ImplBuffer> bufferPool;
		srat::HandlePool<vkof::Sampler, ImplSampler> samplerPool;
		srat::HandlePool<vkof::TransientImage, ImplTransientImage>
			transientImagePool;
		srat::HandlePool<vkof::TransientBuffer, ImplTransientBuffer>
			transientBufferPool;
		srat::HandlePool<vkof::Pipeline, ImplPipelineGraphics>
			pipelineGraphicsPool;
		srat::HandlePool<vkof::Pipeline, ImplPipelineCompute>
			pipelineComputePool;
		srat::HandlePool<vkof::CommandBuffer, ImplCommandBuffer>
			commandBufferPool;
		srat::HandlePool<vkof::RenderNode, ImplRenderNode>
			renderNodePool;

		VkDescriptorPool imguiDescriptorPool { VK_NULL_HANDLE };
		bool imguiFrameActive { false };
		Profiler profiler;

		// need a linear list of pipeline handles for hot reload iteration
		std::vector<vkof::Pipeline> pipelineGraphicsHandles {};
		std::vector<vkof::Pipeline> pipelineComputeHandles {};

		// transient resource handles for cleanup in shutdown
		std::vector<vkof::Image> transientImageCleanupHandles {};
		std::vector<vkof::Buffer> transientBufferCleanupHandles {};

	};
}
static std::filesystem::file_time_type file_write_time(std::string const & p) {
	std::error_code ec;
	auto const t = std::filesystem::last_write_time(p, ec);
	return ec ? std::filesystem::file_time_type {} : t;
}

static ImplDevice * sDevice;

static PFN_vkGetImageViewHandleNVX pfnGetImageViewHandleNVX;
static PFN_vkCmdDrawMeshTasksEXT pfnVkCmdDrawMeshTasksEXT;

// -----------------------------------------------------------------------------
// -- buffer
// -----------------------------------------------------------------------------

vkof::Buffer vkof::buffer_create(
	vkof::BufferCreateInfo const & ci
) {
	ImplBuffer implBuffer;
	VkBufferCreateInfo const bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = ci.byteCount,
		.usage = (
			  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
			| VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			| VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
			| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
			| VK_BUFFER_USAGE_TRANSFER_DST_BIT
		),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	bool const isHost = (ci.memory == vkof::BufferMemory::HostWritable);
	VmaAllocationCreateInfo const allocationCreateInfo = {
		.flags = (
			VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| (
				isHost
					? (VmaAllocationCreateFlags)(
						  VMA_ALLOCATION_CREATE_MAPPED_BIT
						| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
					)
					: (VmaAllocationCreateFlags)0
			)
		),
		.usage = (
			isHost
				? VMA_MEMORY_USAGE_AUTO_PREFER_HOST
				: VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		),
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VmaAllocationInfo allocationInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&bufferCreateInfo,
			&allocationCreateInfo,
			&implBuffer.buffer,
			&implBuffer.allocation,
			&allocationInfo
		)
	);

	// -- BDA, fetched for every buffer (the core of the binding model)
	VkBufferDeviceAddressInfo const addressInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.pNext = nullptr,
		.buffer = implBuffer.buffer,
	};
	implBuffer.deviceAddress = (
		vkGetBufferDeviceAddress(sDevice->device, &addressInfo)
	);

	// -- host mapped pointer, only for host-writable (VMA mapped it for us)
	implBuffer.mapped = (isHost ? allocationInfo.pMappedData : nullptr);
	implBuffer.byteCount = ci.byteCount;

	return sDevice->bufferPool.allocate(implBuffer);
}

void vkof::buffer_destroy(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer) {
		vmaDestroyBuffer(
			sDevice->allocator, implBuffer->buffer, implBuffer->allocation
		);
		sDevice->bufferPool.free(buffer);
	}
}

u64 vkof::buffer_virtual_address(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer) {
		return implBuffer->deviceAddress;
	}
	return 0;
}

srat::slice<u8> vkof::buffer_host_address(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer && implBuffer->mapped != nullptr) {
		return srat::slice<u8>(
			(u8 *)implBuffer->mapped, implBuffer->byteCount
		);
	}
	return srat::slice<u8>(nullptr, 0);
}

void vkof::buffer_upload(BufferUploadInfo const & uploadInfo)
{
	auto const dst = sDevice->bufferPool.get(uploadInfo.buffer);
	if (!dst || uploadInfo.data.size() == 0) {
		return;
	}

	// -- staging buffer (host-visible, mapped)
	VkBufferCreateInfo const stagingCi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = uploadInfo.data.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo const stagingAllocCi = {
		.flags = (
			  VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| VMA_ALLOCATION_CREATE_MAPPED_BIT
			| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		),
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer stagingBuffer;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo stagingInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&stagingCi,
			&stagingAllocCi,
			&stagingBuffer,
			&stagingAlloc,
			&stagingInfo
		)
	);
	std::memcpy(
		stagingInfo.pMappedData,
		uploadInfo.data.ptr(),
		uploadInfo.data.size()
	);

	// -- one-shot command buffer
	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));
	VkBufferCopy const region = {
		.srcOffset = 0,
		.dstOffset = uploadInfo.byteOffset,
		.size = uploadInfo.data.size(),
	};
	vkCmdCopyBuffer(cmd, stagingBuffer, dst->buffer, 1, &region);
	VkAssert(vkEndCommandBuffer(cmd));

	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));

	vkFreeCommandBuffers(
		sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
	);
	vmaDestroyBuffer(sDevice->allocator, stagingBuffer, stagingAlloc);
}

void vkof::buffer_download(BufferDownloadInfo const & info)
{
	auto const src = sDevice->bufferPool.get(info.buffer);
	if (!src || info.dst.size() == 0) {
		return;
	}

	// -- host-visible staging buffer (destination of the GPU copy)
	VkBufferCreateInfo const stagingCi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = info.dst.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo const stagingAllocCi = {
		.flags = (
			  VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| VMA_ALLOCATION_CREATE_MAPPED_BIT
			| VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
		),
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer stagingBuffer;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo stagingInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&stagingCi,
			&stagingAllocCi,
			&stagingBuffer,
			&stagingAlloc,
			&stagingInfo
		)
	);

	// -- one-shot command buffer: GPU buffer -> staging
	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	// memory barrier: make sure all prior writes are visible before the copy
	VkBufferMemoryBarrier const barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = src->buffer,
		.offset = info.byteOffset,
		.size = info.dst.size(),
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 1, &barrier, 0, nullptr
	);

	VkBufferCopy const region = {
		.srcOffset = info.byteOffset,
		.dstOffset = 0,
		.size = info.dst.size(),
	};
	vkCmdCopyBuffer(cmd, src->buffer, stagingBuffer, 1, &region);
	VkAssert(vkEndCommandBuffer(cmd));

	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));

	vkFreeCommandBuffers(
		sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
	);

	// -- copy staging -> caller buffer
	auto dst = info.dst;
	std::memcpy(dst.ptr(), stagingInfo.pMappedData, dst.size());

	vmaDestroyBuffer(sDevice->allocator, stagingBuffer, stagingAlloc);
}

// -----------------------------------------------------------------------------
// -- image
// -----------------------------------------------------------------------------

vkof::Image vkof::image_create(vkof::ImageCreateInfo const & createInfo)
{
	ImplTexture implTexture;
	VkFormat const vkFormat = to_vk_format(createInfo.format);
	VkImageCreateInfo const imageCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vkFormat,
		.extent = VkExtent3D {
			.width = createInfo.width,
			.height = createInfo.height,
			.depth = 1,
		},
		.mipLevels = createInfo.mipLevels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = format_usage(vkFormat),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VmaAllocationCreateInfo const allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkAssert(
		vmaCreateImage(
			sDevice->allocator,
			&imageCreateInfo,
			&allocationCreateInfo,
			&implTexture.image,
			&implTexture.allocation,
			nullptr
		)
	);
	implTexture.width = createInfo.width;
	implTexture.height = createInfo.height;

	// view: aspect must match the format
	VkImageViewCreateInfo const imageViewFullCi = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.image = implTexture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vkFormat,
		.components = VkComponentMapping {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		},
		.subresourceRange = VkImageSubresourceRange {
			.aspectMask = format_aspect(vkFormat),
			.baseMipLevel = 0,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkAssert(
		vkCreateImageView(
			sDevice->device,
			&imageViewFullCi,
			nullptr,
			&implTexture.imageViewFull
		)
	);

	implTexture.imageViewPerMip.resize(createInfo.mipLevels);
	for (u32 mip = 0; mip < createInfo.mipLevels; ++mip) {
		VkImageViewCreateInfo const imageViewMipCi = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = implTexture.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = vkFormat,
			.components = VkComponentMapping {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = format_aspect(vkFormat),
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkAssert(
			vkCreateImageView(
				sDevice->device,
				&imageViewMipCi,
				nullptr,
				&implTexture.imageViewPerMip[mip]
			)
		);
	}

	return sDevice->imagePool.allocate(implTexture);
}

void vkof::image_destroy(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	if (implTexture) {
		vkDestroyImageView(
			sDevice->device, implTexture->imageViewFull, nullptr
		);
		for (u32 mip = 0; mip < implTexture->imageViewPerMip.size(); ++mip) {
			vkDestroyImageView(
				sDevice->device, implTexture->imageViewPerMip[mip], nullptr
			);
		}
		vmaDestroyImage(
			sDevice->allocator, implTexture->image, implTexture->allocation
		);
		sDevice->imagePool.free(image);
	}
}

u32 vkof::image_width(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	return implTexture ? implTexture->width : 0;
}

u32 vkof::image_height(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	return implTexture ? implTexture->height : 0;
}

vkof::Sampler vkof::sampler_create(SamplerCreateInfo const & createInfo)
{
	auto const toFilter = [](SamplerFilter f) {
		return (
			f == SamplerFilter::nearest
				? VK_FILTER_NEAREST
				: VK_FILTER_LINEAR
		);
	};
	auto const toAddress = [](SamplerAddressMode m) {
		switch (m) {
			case SamplerAddressMode::repeat:
				return VK_SAMPLER_ADDRESS_MODE_REPEAT;
			case SamplerAddressMode::mirrored_repeat:
				return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
			default:
				return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		}
	};

	ImplSampler implSampler;
	VkSamplerCreateInfo const samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.magFilter = toFilter(createInfo.magFilter),
		.minFilter = toFilter(createInfo.minFilter),
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = toAddress(createInfo.addressModeU),
		.addressModeV = toAddress(createInfo.addressModeV),
		.addressModeW = toAddress(createInfo.addressModeW),
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	VkAssert(
		vkCreateSampler(
			sDevice->device,
			&samplerCreateInfo,
			nullptr,
			&implSampler.sampler
		)
	);
	return sDevice->samplerPool.allocate(implSampler);
}

void vkof::sampler_destroy(Sampler const & sampler)
{
	auto const implSampler = sDevice->samplerPool.get(sampler);
	if (implSampler) {
		vkDestroySampler(sDevice->device, implSampler->sampler, nullptr);
		sDevice->samplerPool.free(sampler);
	}
}

u64 vkof::image_sampler_handle(ImageSamplerHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	auto const implSampler = sDevice->samplerPool.get(info.sampler);

	if (!implTexture || !implSampler) {
		return 0;
	}

	VkImageViewHandleInfoNVX const handleInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
		.pNext = nullptr,
		.imageView = implTexture->imageViewFull,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.sampler = implSampler->sampler,
	};

	return u64(pfnGetImageViewHandleNVX(sDevice->device, &handleInfo));
}

static VkImageView image_storage_view_for_mip(
	ImplTexture const & implTexture, u32 mipLevel
)
{
	if (mipLevel >= implTexture.imageViewPerMip.size()) {
		return VK_NULL_HANDLE;
	}
	return implTexture.imageViewPerMip[mipLevel];
}

u64 vkof::image_storage_handle(ImageStorageHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	if (!implTexture) {
		return 0;
	}

	VkImageView const view = (
		info.mipLevel == 0
		? implTexture->imageViewFull
		: image_storage_view_for_mip(*implTexture, info.mipLevel)
	);

	VkImageViewHandleInfoNVX const handleInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
		.pNext = nullptr,
		.imageView = view,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.sampler = VK_NULL_HANDLE,
	};

	return u64(pfnGetImageViewHandleNVX(sDevice->device, &handleInfo));
}

// -----------------------------------------------------------------------------
// -- transient resources
// -----------------------------------------------------------------------------

vkof::TransientImage vkof::transient_image_create(
	TransientImageCreateInfo const & createInfo
) {
	u32 const w = (u32)(sDevice->swapchain.extent.width  * createInfo.scaleWidth);
	u32 const h = (u32)(sDevice->swapchain.extent.height * createInfo.scaleHeight);

	ImplTransientImage impl;
	impl.format         = createInfo.format;
	impl.scaleWidth     = createInfo.scaleWidth;
	impl.scaleHeight    = createInfo.scaleHeight;
	impl.mipLevels      = createInfo.mipLevels;
	impl.isDoubleBuffered = createInfo.isDoubleBuffered;
	impl.handles[0] = { .id = 0 };
	impl.handles[1] = { .id = 0 };

	auto const allocSlot = [&](u32 slot) {
		vkof::ImageCreateInfo const ci {
			.width    = w,
			.height   = h,
			.format   = createInfo.format,
			.mipLevels = createInfo.mipLevels,
			.optInitialData = {},
		};
		impl.handles[slot] = vkof::image_create(ci);
		sDevice->transientImageCleanupHandles.push_back(impl.handles[slot]);
	};

	allocSlot(0);
	if (createInfo.isDoubleBuffered) {
		allocSlot(1);
	}

	return sDevice->transientImagePool.allocate(impl);
}

vkof::TransientBuffer vkof::transient_buffer_create(
	TransientBufferCreateInfo const & createInfo
) {
	ImplTransientBuffer impl;
	impl.isDoubleBuffered = createInfo.isDoubleBuffered;
	impl.handles[0] = { .id = 0 };
	impl.handles[1] = { .id = 0 };

	auto const allocSlot = [&](u32 slot) {
		vkof::BufferCreateInfo const ci {
			.byteCount = createInfo.byteCount,
			.memory    = vkof::BufferMemory::DeviceOnly,
		};
		impl.handles[slot] = vkof::buffer_create(ci);
		sDevice->transientBufferCleanupHandles.push_back(impl.handles[slot]);
	};

	allocSlot(0);
	if (createInfo.isDoubleBuffered) {
		allocSlot(1);
	}

	return sDevice->transientBufferPool.allocate(impl);
}

vkof::Image vkof::transient_image_get_image(
	TransientImage const & transientImage
) {
	auto const impl = sDevice->transientImagePool.get(transientImage);
	if (!impl) {
		return vkof::Image { .id = 0 };
	}
	u32 const slot = (
		impl->isDoubleBuffered
			? (sDevice->frameIndex % 2)
			: 0u
	);
	return impl->handles[slot];
}

vkof::Buffer vkof::transient_buffer_get_buffer(
	TransientBuffer const & transientBuffer
) {
	auto const impl = sDevice->transientBufferPool.get(transientBuffer);
	if (!impl) {
		return vkof::Buffer { .id = 0 };
	}
	u32 const slot = (
		impl->isDoubleBuffered
			? (sDevice->frameIndex % 2)
			: 0u
	);
	return impl->handles[slot];
}

// -----------------------------------------------------------------------------
// -- render nodes
// -----------------------------------------------------------------------------

vkof::RenderNode vkof::render_node_create(
	RenderNodeCreateInfo const & createInfo
) {
	ImplRenderNode impl;
	impl.queue = createInfo.queue;
	return sDevice->renderNodePool.allocate(impl);
}

void vkof::render_node_destroy(RenderNode const & node)
{
	sDevice->renderNodePool.free(node);
}

void vkof::render_node_add_image(RenderNodeImageInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->imageBarriers.push_back(
		ImplRenderBarrier {
			.transientImage = info.image,
			.access = info.access,
		}
	);
}

void vkof::render_node_add_buffer(RenderNodeBufferInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->bufferBarriers.push_back(
		ImplBufferBarrier {
			.transientBuffer = info.buffer,
			.access = info.access,
		}
	);
}

void vkof::render_node_attachment_color(
	RenderNodeAttachmentColorInfo const & info
) {
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	ImplColorAttachment att;
	att.image     = info.image;
	att.loadOp    = info.loadOp;
	att.mipLevel  = info.mipLevel;
	att.colorIndex = info.colorIndex;
	att.clearColor[0] = (info.clearColor.size() > 0) ? info.clearColor.ptr()[0] : 0.0f;
	att.clearColor[1] = (info.clearColor.size() > 1) ? info.clearColor.ptr()[1] : 0.0f;
	att.clearColor[2] = (info.clearColor.size() > 2) ? info.clearColor.ptr()[2] : 0.0f;
	att.clearColor[3] = (info.clearColor.size() > 3) ? info.clearColor.ptr()[3] : 1.0f;
	node->colorAttachments.push_back(att);
}

void vkof::render_node_attachment_depth(
	RenderNodeAttachmentDepthInfo const & info
) {
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->depthAttachment = ImplDepthAttachment {
		.image      = info.image,
		.loadOp     = info.loadOp,
		.mipLevel   = info.mipLevel,
		.clearDepth = (info.clearDepth.size() > 0) ? info.clearDepth.ptr()[0] : 1.0f,
	};
}

void vkof::render_node_callback(RenderNodeCallbackInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->callback = info.callback;
}

// -----------------------------------------------------------------------------
// -- creation/destruction
// -----------------------------------------------------------------------------

void vkof::init()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	u32 const windowWidth = 1280;
	u32 const windowHeight = 720;
	GLFWwindow * window = (
		glfwCreateWindow(windowWidth, windowHeight, "vkof", nullptr, nullptr)
	);

	// -- instance
	auto instance = [&]() -> vkb::Instance {
		auto const instanceResults = (
			vkb::InstanceBuilder()
			.set_app_name("demo")
			.request_validation_layers(true)
			.use_default_debug_messenger()
			.require_api_version(1, 3, 0)
			.build()
		);
		if (!instanceResults) {
			printf(
				"failed to create instance: %s\n",
				instanceResults.error().message().c_str()
			);
			exit(1);
		}
		return instanceResults.value();
	}();

	// -- surface
	auto const surface = [&]() -> VkSurfaceKHR {
		VkSurfaceKHR surface;
		VkAssert(
			glfwCreateWindowSurface(
				instance.instance,
				window,
				/*allocator=*/ nullptr,
				&surface
			)
		);
		return surface;
	}();

	// -- device
	auto device = [&]() -> vkb::Device {
		VkPhysicalDeviceFeatures features {
			.robustBufferAccess = true,
			.fullDrawIndexUint32 = true,
			.imageCubeArray = true,
			.independentBlend = true,
			.geometryShader = true,
			.tessellationShader = true,
			.sampleRateShading = true,
			.dualSrcBlend = false,
			.logicOp = true,
			.multiDrawIndirect = true,
			.drawIndirectFirstInstance = true,
			.depthClamp = true,
			.depthBiasClamp = true,
			.fillModeNonSolid = true,
			.depthBounds = true,
			.wideLines = false,
			.largePoints = false,
			.alphaToOne = true,
			.multiViewport = true,
			.samplerAnisotropy = true,
			.textureCompressionETC2 = false,
			.textureCompressionASTC_LDR = false,
			.textureCompressionBC = false,
			.occlusionQueryPrecise = false,
			.pipelineStatisticsQuery = false,
			.vertexPipelineStoresAndAtomics = true,
			.fragmentStoresAndAtomics = true,
			.shaderTessellationAndGeometryPointSize = false,
			.shaderImageGatherExtended = false,
			.shaderStorageImageExtendedFormats = true,
			.shaderStorageImageMultisample = true,
			.shaderStorageImageReadWithoutFormat = true,
			.shaderStorageImageWriteWithoutFormat = true,
			.shaderUniformBufferArrayDynamicIndexing = true,
			.shaderSampledImageArrayDynamicIndexing = true,
			.shaderStorageBufferArrayDynamicIndexing = true,
			.shaderStorageImageArrayDynamicIndexing = true,
			.shaderClipDistance = false,
			.shaderCullDistance = false,
			.shaderFloat64 = true,
			.shaderInt64 = true,
			.shaderInt16 = true,
			.shaderResourceResidency = false,
			.shaderResourceMinLod = false,
			.sparseBinding = false,
			.sparseResidencyBuffer = false,
			.sparseResidencyImage2D = false,
			.sparseResidencyImage3D = false,
			.sparseResidency2Samples = false,
			.sparseResidency4Samples = false,
			.sparseResidency8Samples = false,
			.sparseResidency16Samples = false,
			.sparseResidencyAliased = false,
			.variableMultisampleRate = false,
			.inheritedQueries = false,
		};
		VkPhysicalDeviceMeshShaderFeaturesEXT const meshShaderFeatures = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
			),
			.pNext = nullptr,
			.taskShader = VK_TRUE,
			.meshShader = VK_TRUE,
			.multiviewMeshShader = VK_FALSE,
			.primitiveFragmentShadingRateMeshShader = VK_FALSE,
			.meshShaderQueries = VK_FALSE,
		};
		VkPhysicalDeviceMaintenance4Features const maintenance4Features = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES
			),
			.pNext = nullptr,
			.maintenance4 = VK_TRUE,
		};
		VkPhysicalDeviceDynamicRenderingFeatures const
			dynamicRenderingFeatures = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
				),
				.pNext = nullptr,
				.dynamicRendering = VK_TRUE,
			};
		VkPhysicalDeviceSynchronization2Features const
			synchronization2Features = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
				),
				.pNext = nullptr,
				.synchronization2 = VK_TRUE,
			};
		VkPhysicalDeviceVulkan12Features const vulkan12Features {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = nullptr,
			.samplerMirrorClampToEdge = VK_TRUE,
			.drawIndirectCount = VK_TRUE,
			.storageBuffer8BitAccess = VK_TRUE,
			.uniformAndStorageBuffer8BitAccess = VK_TRUE,
			.storagePushConstant8 = VK_TRUE,
			.shaderBufferInt64Atomics = VK_TRUE,
			.shaderSharedInt64Atomics = VK_TRUE,
			.shaderFloat16 = VK_TRUE,
			.shaderInt8 = VK_TRUE,
			.descriptorIndexing = VK_TRUE,
			.shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
			.shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
			.descriptorBindingPartiallyBound = VK_TRUE,
			.descriptorBindingVariableDescriptorCount = VK_TRUE,
			.runtimeDescriptorArray = VK_TRUE,
			.samplerFilterMinmax = VK_TRUE,
			.scalarBlockLayout = VK_TRUE,
			.imagelessFramebuffer = VK_TRUE,
			.uniformBufferStandardLayout = VK_TRUE,
			.shaderSubgroupExtendedTypes = VK_TRUE,
			.separateDepthStencilLayouts = VK_TRUE,
			.hostQueryReset = VK_TRUE,
			.timelineSemaphore = VK_TRUE,
			.bufferDeviceAddress = VK_TRUE,
			.bufferDeviceAddressCaptureReplay = VK_TRUE,
			.bufferDeviceAddressMultiDevice = VK_TRUE,
			.vulkanMemoryModel = VK_TRUE,
			.vulkanMemoryModelDeviceScope = VK_TRUE,
			.vulkanMemoryModelAvailabilityVisibilityChains = VK_TRUE,
			.shaderOutputViewportIndex = VK_TRUE,
			.shaderOutputLayer = VK_TRUE,
			.subgroupBroadcastDynamicId = VK_TRUE,
		};
		auto const physicalDeviceResults = (
			vkb::PhysicalDeviceSelector(instance)
			.set_surface(surface)
			.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
			.set_required_features(features)
			.add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
			.add_required_extension(VK_KHR_MAINTENANCE_6_EXTENSION_NAME)
			.add_required_extension(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME)
			.add_required_extension_features(meshShaderFeatures)
			.add_required_extension_features(maintenance4Features)
			.add_required_extension_features(dynamicRenderingFeatures)
			.add_required_extension_features(synchronization2Features)
			.add_required_extension_features(vulkan12Features)
			.select()
		);
		if (!physicalDeviceResults) {
			printf(
				"failed to select physical device: %s\n",
				physicalDeviceResults.error().message().c_str()
			);
			exit(1);
		}
		auto const deviceResults = (
			vkb::DeviceBuilder(physicalDeviceResults.value()).build()
		);
		if (!deviceResults) {
			printf(
				"failed to create logical device: %s\n",
				deviceResults.error().message().c_str()
			);
			exit(1);
		}
		return deviceResults.value();
	}();

	// -- queues
	auto const graphicsQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::graphics);
		if (!r) {
			printf(
				"failed to get graphics queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();
	auto const computeQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::compute);
		if (!r) {
			printf(
				"failed to get compute queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();

	// -- swapchain (match the window size)
	auto swapchain = [&]() -> vkb::Swapchain {
		auto const swapchainResults = (
			vkb::SwapchainBuilder(device)
			.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
			.set_desired_extent(windowWidth, windowHeight)
			.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
			.build()
		);
		if (!swapchainResults) {
			printf(
				"failed to create swapchain: %s\n",
				swapchainResults.error().message().c_str()
			);
			exit(1);
		}
		return swapchainResults.value();
	}();

	// -- vma allocator
	auto const allocator = [&]() -> VmaAllocator {
		VmaAllocatorCreateInfo const allocatorCreateInfo = {
			.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
			.physicalDevice = device.physical_device,
			.device = device.device,
			.preferredLargeHeapBlockSize = 0,
			.pAllocationCallbacks = nullptr,
			.pDeviceMemoryCallbacks = nullptr,
			.pHeapSizeLimit = nullptr,
			.pVulkanFunctions = nullptr,
			.instance = instance.instance,
			.vulkanApiVersion = VK_API_VERSION_1_3,
			.pTypeExternalMemoryHandleTypes = nullptr,
		};
		VmaAllocator allocator;
		VkAssert(vmaCreateAllocator(&allocatorCreateInfo, &allocator));
		return allocator;
	}();

	// -- depth swapchain images
	std::vector<VmaAllocation> depthSwapchainAllocs(swapchain.image_count);
	auto depthSwapchainImages = [&]() -> std::vector<VkImage> {
		std::vector<VkImage> images(swapchain.image_count);
		for (size_t i = 0; i < images.size(); ++i) {
			VkImageCreateInfo const imageCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.extent = VkExtent3D {
					.width = swapchain.extent.width,
					.height = swapchain.extent.height,
					.depth = 1,
				},
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = (
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
					| VK_IMAGE_USAGE_SAMPLED_BIT
				),
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.queueFamilyIndexCount = 0,
				.pQueueFamilyIndices = nullptr,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};
			VmaAllocationCreateInfo const allocationCreateInfo = {
				.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
				.requiredFlags = 0,
				.preferredFlags = 0,
				.memoryTypeBits = 0,
				.pool = nullptr,
				.pUserData = nullptr,
				.priority = 0.0f,
			};
			VkImage image;
			VkAssert(
				vmaCreateImage(
					allocator,
					&imageCreateInfo,
					&allocationCreateInfo,
					&image,
					&depthSwapchainAllocs[i],
					nullptr
				)
			);
			images[i] = image;
		}
		return images;
	}();
	auto depthSwapchainImageViews = [&]() -> std::vector<VkImageView> {
		std::vector<VkImageView> imageViews(depthSwapchainImages.size());
		for (size_t i = 0; i < imageViews.size(); ++i) {
			VkImageViewCreateInfo const imageViewCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = depthSwapchainImages[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.components = VkComponentMapping {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VkAssert(
				vkCreateImageView(
					device.device,
					&imageViewCreateInfo,
					nullptr,
					&imageViews[i]
				)
			);
		}
		return imageViews;
	}();

	// -- comand pool
	VkCommandPool commandPoolGraphics;
	{
		VkCommandPoolCreateInfo const commandPoolCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::graphics).value()
			),
		};
		VkAssert(
			vkCreateCommandPool(
				device.device,
				&commandPoolCreateInfo,
				nullptr,
				&commandPoolGraphics
			)
		);
	}

	VkCommandPool commandPoolCompute;
	{
		VkCommandPoolCreateInfo const commandPoolCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::compute).value()
			),
		};
		VkAssert(
			vkCreateCommandPool(
				device.device,
				&commandPoolCreateInfo,
				nullptr,
				&commandPoolCompute
			)
		);
	}

	VkPushConstantRange universalPushConstantRange {
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256, // [0,128) = root/frame constants; [128,256) = per-draw
	};
	VkPipelineLayoutCreateInfo const universalPipelineLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &universalPushConstantRange,
	};
	VkPipelineLayout pipelineLayoutUniversal;
	VkAssert(
		vkCreatePipelineLayout(
			device.device,
			/*createInfo=*/ &universalPipelineLayoutCreateInfo,
			/*allocator=*/ nullptr,
			&pipelineLayoutUniversal
		)
	);

	sDevice = new ImplDevice {
		.window = window,
		.instance = instance,
		.physicalDevice = device.physical_device,
		.device = device,
		.swapchain = swapchain,
		.swapchainImages = swapchain.get_images().value(),
		.swapchainImageViews = swapchain.get_image_views().value(),
		.swapchainDepthImages = depthSwapchainImages,
		.swapchainDepthImageViews = depthSwapchainImageViews,
		.swapchainDepthAllocs = depthSwapchainAllocs,
		.swapchainImageIndex = 0,
		.commandPoolGraphics = commandPoolGraphics,
		.commandPoolCompute = commandPoolCompute,
		.pipelineLayoutUniversal = pipelineLayoutUniversal,
		.surface = surface,
		.queueGraphics = graphicsQueue,
		.queueCompute = computeQueue,
		.allocator = allocator,
		.frameData = {},  // populated below after pool setup
		.frameIndex = 0,
		.semaphoresRender = {},
		.imagePool = (
			srat::HandlePool<vkof::Image, ImplTexture>::create(1024)
		),
		.bufferPool = (
			srat::HandlePool<vkof::Buffer, ImplBuffer>::create(1024)
		),
		.samplerPool = (
			srat::HandlePool<vkof::Sampler, ImplSampler>::create(16)
		),
		.transientImagePool = (
			srat::HandlePool<vkof::TransientImage, ImplTransientImage>
				::create(32)
		),
		.transientBufferPool = (
			srat::HandlePool<vkof::TransientBuffer, ImplTransientBuffer>
				::create(32)
		),
		.pipelineGraphicsPool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineGraphics>
				::create(128)
		),
		.pipelineComputePool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineCompute>
				::create(128)
		),
		.commandBufferPool = (
			srat::HandlePool<vkof::CommandBuffer, ImplCommandBuffer>
				::create(256)
		),
		.renderNodePool = (
			srat::HandlePool<vkof::RenderNode, ImplRenderNode>
				::create(64)
		),
		.profiler = {},
	};

	pfnGetImageViewHandleNVX = (
		(PFN_vkGetImageViewHandleNVX)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkGetImageViewHandleNVX"
		)
	);
	pfnVkCmdDrawMeshTasksEXT = (
		(PFN_vkCmdDrawMeshTasksEXT)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkCmdDrawMeshTasksEXT"
		)
	);

	// -- per-frame-in-flight: command buffers, semaphores, fences
	{
		VkCommandBufferAllocateInfo const cbai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = sDevice->commandPoolGraphics,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkSemaphoreCreateInfo const sci = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
		};
		// pre-signaled so the first frame doesn't wait forever
		VkFenceCreateInfo const fci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		for (auto & fd : sDevice->frameData) {
			VkAssert(
				vkAllocateCommandBuffers(
					sDevice->device, &cbai, &fd.cmdGraphics
				)
			);
			VkAssert(
				vkCreateSemaphore(
					sDevice->device, &sci, nullptr, &fd.semaphoreAcquire
				)
			);
			VkAssert(
				vkCreateFence(sDevice->device, &fci, nullptr, &fd.fence)
			);
		}
		sDevice->semaphoresRender.resize(sDevice->swapchain.image_count);
		for (VkSemaphore & sem : sDevice->semaphoresRender) {
			VkAssert(vkCreateSemaphore(sDevice->device, &sci, nullptr, &sem));
		}
	}
	sDevice->frameIndex = 0;

	// -- imgui
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(window, true);

	VkDescriptorPoolSize const imguiPoolSize = {
		.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
	};
	VkDescriptorPoolCreateInfo const imguiPoolCi = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &imguiPoolSize,
	};
	VkAssert(
		vkCreateDescriptorPool(
			sDevice->device, &imguiPoolCi, nullptr, &sDevice->imguiDescriptorPool
		)
	);

	VkFormat const swapchainFormat = sDevice->swapchain.image_format;
	VkPipelineRenderingCreateInfo const imguiRenderingCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.pNext = nullptr,
		.viewMask = 0,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &swapchainFormat,
		.depthAttachmentFormat = VK_FORMAT_UNDEFINED,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};
	ImGui_ImplVulkan_InitInfo imguiVkInfo = {};
	imguiVkInfo.Instance = sDevice->instance.instance;
	imguiVkInfo.PhysicalDevice = sDevice->device.physical_device;
	imguiVkInfo.Device = sDevice->device.device;
	imguiVkInfo.QueueFamily = (
		sDevice->device.get_queue_index(vkb::QueueType::graphics).value()
	);
	imguiVkInfo.Queue = sDevice->queueGraphics;
	imguiVkInfo.DescriptorPool = sDevice->imguiDescriptorPool;
	imguiVkInfo.RenderPass = VK_NULL_HANDLE;
	imguiVkInfo.MinImageCount = 2;
	imguiVkInfo.ImageCount = (u32)sDevice->swapchainImages.size();
	imguiVkInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	imguiVkInfo.UseDynamicRendering = true;
	imguiVkInfo.PipelineRenderingCreateInfo = imguiRenderingCi;
	ImGui_ImplVulkan_Init(&imguiVkInfo);
	ImGui_ImplVulkan_CreateFontsTexture();

	profiler_init();
}

void vkof::init_headless(u32 width, u32 height)
{
	// -- instance (headless: suppress all windowing extensions)
	auto instance = [&]() -> vkb::Instance {
		auto const r = vkb::InstanceBuilder()
			.set_app_name("vkof-test")
			.request_validation_layers(true)
			.use_default_debug_messenger()
			.require_api_version(1, 3, 0)
			.set_headless(true)
			.build();
		if (!r) {
			printf(
				"failed to create headless instance: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();

	// -- physical device + logical device (no surface)
	auto device = [&]() -> vkb::Device {
		VkPhysicalDeviceFeatures features {
			.robustBufferAccess = true,
			.fullDrawIndexUint32 = true,
			.imageCubeArray = true,
			.independentBlend = true,
			.geometryShader = true,
			.tessellationShader = true,
			.sampleRateShading = true,
			.dualSrcBlend = true,
			.logicOp = true,
			.multiDrawIndirect = true,
			.drawIndirectFirstInstance = true,
			.depthClamp = true,
			.depthBiasClamp = true,
			.fillModeNonSolid = true,
			.depthBounds = true,
			.wideLines = true,
			.largePoints = true,
			.alphaToOne = true,
			.multiViewport = true,
			.samplerAnisotropy = true,
			.textureCompressionETC2 = false,
			.textureCompressionASTC_LDR = false,
			.textureCompressionBC = true,
			.occlusionQueryPrecise = true,
			.pipelineStatisticsQuery = true,
			.vertexPipelineStoresAndAtomics = true,
			.fragmentStoresAndAtomics = true,
			.shaderTessellationAndGeometryPointSize = true,
			.shaderImageGatherExtended = true,
			.shaderStorageImageExtendedFormats = true,
			.shaderStorageImageMultisample = true,
			.shaderStorageImageReadWithoutFormat = true,
			.shaderStorageImageWriteWithoutFormat = true,
			.shaderUniformBufferArrayDynamicIndexing = true,
			.shaderSampledImageArrayDynamicIndexing = true,
			.shaderStorageBufferArrayDynamicIndexing = true,
			.shaderStorageImageArrayDynamicIndexing = true,
			.shaderClipDistance = true,
			.shaderCullDistance = true,
			.shaderFloat64 = true,
			.shaderInt64 = true,
			.shaderInt16 = true,
			.shaderResourceResidency = true,
			.shaderResourceMinLod = true,
			.sparseBinding = false,
			.sparseResidencyBuffer = false,
			.sparseResidencyImage2D = false,
			.sparseResidencyImage3D = false,
			.sparseResidency2Samples = false,
			.sparseResidency4Samples = false,
			.sparseResidency8Samples = false,
			.sparseResidency16Samples = false,
			.sparseResidencyAliased = false,
			.variableMultisampleRate = false,
			.inheritedQueries = false,
		};
		VkPhysicalDeviceMeshShaderFeaturesEXT const meshShaderFeatures = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
			),
			.pNext = nullptr,
			.taskShader = VK_TRUE,
			.meshShader = VK_TRUE,
			.multiviewMeshShader = VK_FALSE,
			.primitiveFragmentShadingRateMeshShader = VK_FALSE,
			.meshShaderQueries = VK_FALSE,
		};
		VkPhysicalDeviceMaintenance4Features const maintenance4Features = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES
			),
			.pNext = nullptr,
			.maintenance4 = VK_TRUE,
		};
		VkPhysicalDeviceDynamicRenderingFeatures const
			dynamicRenderingFeatures = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
				),
				.pNext = nullptr,
				.dynamicRendering = VK_TRUE,
			};
		VkPhysicalDeviceSynchronization2Features const
			synchronization2Features = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
				),
				.pNext = nullptr,
				.synchronization2 = VK_TRUE,
			};
		VkPhysicalDeviceVulkan12Features const vulkan12Features {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = nullptr,
			.samplerMirrorClampToEdge = VK_TRUE,
			.drawIndirectCount = VK_TRUE,
			.storageBuffer8BitAccess = VK_TRUE,
			.uniformAndStorageBuffer8BitAccess = VK_TRUE,
			.storagePushConstant8 = VK_TRUE,
			.shaderBufferInt64Atomics = VK_TRUE,
			.shaderSharedInt64Atomics = VK_TRUE,
			.shaderFloat16 = VK_TRUE,
			.shaderInt8 = VK_TRUE,
			.descriptorIndexing = VK_TRUE,
			.shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
			.shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
			.descriptorBindingPartiallyBound = VK_TRUE,
			.descriptorBindingVariableDescriptorCount = VK_TRUE,
			.runtimeDescriptorArray = VK_TRUE,
			.samplerFilterMinmax = VK_TRUE,
			.scalarBlockLayout = VK_TRUE,
			.imagelessFramebuffer = VK_TRUE,
			.uniformBufferStandardLayout = VK_TRUE,
			.shaderSubgroupExtendedTypes = VK_TRUE,
			.separateDepthStencilLayouts = VK_TRUE,
			.hostQueryReset = VK_TRUE,
			.timelineSemaphore = VK_TRUE,
			.bufferDeviceAddress = VK_TRUE,
			.bufferDeviceAddressCaptureReplay = VK_TRUE,
			.bufferDeviceAddressMultiDevice = VK_TRUE,
			.vulkanMemoryModel = VK_TRUE,
			.vulkanMemoryModelDeviceScope = VK_TRUE,
			.vulkanMemoryModelAvailabilityVisibilityChains = VK_TRUE,
			.shaderOutputViewportIndex = VK_TRUE,
			.shaderOutputLayer = VK_TRUE,
			.subgroupBroadcastDynamicId = VK_TRUE,
		};
		auto const physResult = (
			vkb::PhysicalDeviceSelector(instance)
			.defer_surface_initialization()
			.require_present(false)
			.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
			.set_required_features(features)
			.add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
			.add_required_extension(VK_KHR_MAINTENANCE_6_EXTENSION_NAME)
			.add_required_extension(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME)
			.add_required_extension_features(meshShaderFeatures)
			.add_required_extension_features(maintenance4Features)
			.add_required_extension_features(dynamicRenderingFeatures)
			.add_required_extension_features(synchronization2Features)
			.add_required_extension_features(vulkan12Features)
			.select()
		);
		if (!physResult) {
			printf(
				"failed to select headless physical device: %s\n",
				physResult.error().message().c_str()
			);
			exit(1);
		}
		auto const devResult = (
			vkb::DeviceBuilder(physResult.value()).build()
		);
		if (!devResult) {
			printf(
				"failed to create headless logical device: %s\n",
				devResult.error().message().c_str()
			);
			exit(1);
		}
		return devResult.value();
	}();

	// -- queues
	auto const graphicsQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::graphics);
		if (!r) {
			printf(
				"failed to get graphics queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();
	auto const computeQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::compute);
		if (!r) {
			printf(
				"failed to get compute queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();

	// -- VMA allocator
	auto const allocator = [&]() -> VmaAllocator {
		VmaAllocatorCreateInfo const allocatorCreateInfo = {
			.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
			.physicalDevice = device.physical_device,
			.device = device.device,
			.preferredLargeHeapBlockSize = 0,
			.pAllocationCallbacks = nullptr,
			.pDeviceMemoryCallbacks = nullptr,
			.pHeapSizeLimit = nullptr,
			.pVulkanFunctions = nullptr,
			.instance = instance.instance,
			.vulkanApiVersion = VK_API_VERSION_1_3,
			.pTypeExternalMemoryHandleTypes = nullptr,
		};
		VmaAllocator a;
		VkAssert(vmaCreateAllocator(&allocatorCreateInfo, &a));
		return a;
	}();

	// -- fake swapchain color image (slot 0, no real swapchain)
	VkImage headlessColorImage;
	VmaAllocation headlessColorAlloc;
	VkImageView headlessColorView;
	{
		VkImageCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_B8G8R8A8_UNORM,
			.extent = VkExtent3D { width, height, 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = (
				  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_SAMPLED_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
				| VK_IMAGE_USAGE_TRANSFER_DST_BIT
				| VK_IMAGE_USAGE_STORAGE_BIT
			),
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		VmaAllocationCreateInfo const aci = {
			.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
			.requiredFlags = 0,
			.preferredFlags = 0,
			.memoryTypeBits = 0,
			.pool = nullptr,
			.pUserData = nullptr,
			.priority = 0.0f,
		};
		VkAssert(vmaCreateImage(allocator, &ci, &aci, &headlessColorImage, &headlessColorAlloc, nullptr));
		VkImageViewCreateInfo const vci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = headlessColorImage,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_B8G8R8A8_UNORM,
			.components = VkComponentMapping {
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkAssert(vkCreateImageView(device.device, &vci, nullptr, &headlessColorView));
	}

	// -- fake depth image (slot 0)
	VkImage headlessDepthImage;
	VmaAllocation headlessDepthAlloc;
	VkImageView headlessDepthView;
	{
		VkImageCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_D32_SFLOAT,
			.extent = VkExtent3D { width, height, 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = (
				  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_SAMPLED_BIT
			),
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		VmaAllocationCreateInfo const aci = {
			.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
			.requiredFlags = 0,
			.preferredFlags = 0,
			.memoryTypeBits = 0,
			.pool = nullptr,
			.pUserData = nullptr,
			.priority = 0.0f,
		};
		VkAssert(vmaCreateImage(allocator, &ci, &aci, &headlessDepthImage, &headlessDepthAlloc, nullptr));
		VkImageViewCreateInfo const vci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = headlessDepthImage,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_D32_SFLOAT,
			.components = VkComponentMapping {
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkAssert(vkCreateImageView(device.device, &vci, nullptr, &headlessDepthView));
	}

	// -- command pools
	VkCommandPool commandPoolGraphics;
	{
		VkCommandPoolCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::graphics).value()
			),
		};
		VkAssert(vkCreateCommandPool(device.device, &ci, nullptr, &commandPoolGraphics));
	}
	VkCommandPool commandPoolCompute;
	{
		VkCommandPoolCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::compute).value()
			),
		};
		VkAssert(vkCreateCommandPool(device.device, &ci, nullptr, &commandPoolCompute));
	}

	// -- universal push-constant layout
	VkPushConstantRange universalPushConstantRange {
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
	};
	VkPipelineLayoutCreateInfo const universalLayoutCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &universalPushConstantRange,
	};
	VkPipelineLayout pipelineLayoutUniversal;
	VkAssert(
		vkCreatePipelineLayout(
			device.device, &universalLayoutCi, nullptr, &pipelineLayoutUniversal
		)
	);

	// -- minimal vkb::Swapchain stub so existing code reads extent/image_count
	vkb::Swapchain headlessSwapchain {};
	headlessSwapchain.device = device.device;
	headlessSwapchain.swapchain = VK_NULL_HANDLE;
	headlessSwapchain.image_count = 1;
	headlessSwapchain.image_format = VK_FORMAT_B8G8R8A8_UNORM;
	headlessSwapchain.extent = VkExtent2D { width, height };

	sDevice = new ImplDevice {
		.window = nullptr,
		.instance = instance,
		.physicalDevice = device.physical_device,
		.device = device,
		.swapchain = headlessSwapchain,
		.swapchainImages = { headlessColorImage },
		.swapchainImageViews = { headlessColorView },
		.swapchainDepthImages = { headlessDepthImage },
		.swapchainDepthImageViews = { headlessDepthView },
		.swapchainDepthAllocs = {},
		.headlessSwapchainAlloc = headlessColorAlloc,
		.headlessDepthAlloc = headlessDepthAlloc,
		.swapchainImageIndex = 0,
		.commandPoolGraphics = commandPoolGraphics,
		.commandPoolCompute = commandPoolCompute,
		.pipelineLayoutUniversal = pipelineLayoutUniversal,
		.surface = VK_NULL_HANDLE,
		.queueGraphics = graphicsQueue,
		.queueCompute = computeQueue,
		.allocator = allocator,
		.frameData = {},
		.frameIndex = 0,
		.semaphoresRender = {},
		.imagePool = (
			srat::HandlePool<vkof::Image, ImplTexture>::create(1024)
		),
		.bufferPool = (
			srat::HandlePool<vkof::Buffer, ImplBuffer>::create(1024)
		),
		.samplerPool = (
			srat::HandlePool<vkof::Sampler, ImplSampler>::create(16)
		),
		.transientImagePool = (
			srat::HandlePool<vkof::TransientImage, ImplTransientImage>
				::create(32)
		),
		.transientBufferPool = (
			srat::HandlePool<vkof::TransientBuffer, ImplTransientBuffer>
				::create(32)
		),
		.pipelineGraphicsPool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineGraphics>
				::create(128)
		),
		.pipelineComputePool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineCompute>
				::create(128)
		),
		.commandBufferPool = (
			srat::HandlePool<vkof::CommandBuffer, ImplCommandBuffer>
				::create(256)
		),
		.renderNodePool = (
			srat::HandlePool<vkof::RenderNode, ImplRenderNode>
				::create(64)
		),
		.profiler = {},
	};

	pfnGetImageViewHandleNVX = (
		(PFN_vkGetImageViewHandleNVX)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkGetImageViewHandleNVX"
		)
	);
	pfnVkCmdDrawMeshTasksEXT = (
		(PFN_vkCmdDrawMeshTasksEXT)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkCmdDrawMeshTasksEXT"
		)
	);

	// -- per-frame-in-flight sync (same as windowed)
	{
		VkCommandBufferAllocateInfo const cbai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = sDevice->commandPoolGraphics,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkSemaphoreCreateInfo const sci = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
		};
		VkFenceCreateInfo const fci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		for (auto & fd : sDevice->frameData) {
			VkAssert(
				vkAllocateCommandBuffers(
					sDevice->device, &cbai, &fd.cmdGraphics
				)
			);
			VkAssert(
				vkCreateSemaphore(
					sDevice->device, &sci, nullptr, &fd.semaphoreAcquire
				)
			);
			VkAssert(
				vkCreateFence(sDevice->device, &fci, nullptr, &fd.fence)
			);
		}
	}
	sDevice->frameIndex = 0;
	profiler_init();
}

void vkof::device_wait_idle()
{
	vkDeviceWaitIdle(sDevice->device);
}

void vkof::shutdown()
{
	vkDeviceWaitIdle(sDevice->device);

	if (sDevice->profiler.queryPool != VK_NULL_HANDLE) {
		vkDestroyQueryPool(sDevice->device, sDevice->profiler.queryPool, nullptr);
	}

	if (sDevice->window != nullptr) {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(
			sDevice->device, sDevice->imguiDescriptorPool, nullptr
		);
	}

	// -- destroy transient resources (owned by the transient pools)
	for (auto const & h : sDevice->transientImageCleanupHandles) {
		vkof::image_destroy(h);
	}
	for (auto const & h : sDevice->transientBufferCleanupHandles) {
		vkof::buffer_destroy(h);
	}

	// -- destroy per-frame and per-image sync primitives
	// (command buffers are freed when commandPoolGraphics is destroyed)
	for (auto & fd : sDevice->frameData) {
		vkDestroySemaphore(sDevice->device, fd.semaphoreAcquire, nullptr);
		vkDestroyFence(sDevice->device, fd.fence, nullptr);
	}
	for (VkSemaphore sem : sDevice->semaphoresRender) {
		vkDestroySemaphore(sDevice->device, sem, nullptr);
	}

	for (VkImageView view : sDevice->swapchainDepthImageViews) {
		vkDestroyImageView(sDevice->device, view, nullptr);
	}
	if (sDevice->window == nullptr) {
		// headless: depth and color images are VMA-allocated; use stored allocs
		vmaDestroyImage(
			sDevice->allocator,
			sDevice->swapchainDepthImages[0],
			sDevice->headlessDepthAlloc
		);
		for (VkImageView view : sDevice->swapchainImageViews) {
			vkDestroyImageView(sDevice->device, view, nullptr);
		}
		vmaDestroyImage(
			sDevice->allocator,
			sDevice->swapchainImages[0],
			sDevice->headlessSwapchainAlloc
		);
	} else {
		for (size_t i = 0; i < sDevice->swapchainDepthImages.size(); ++i) {
			vmaDestroyImage(
				sDevice->allocator,
				sDevice->swapchainDepthImages[i],
				sDevice->swapchainDepthAllocs[i]
			);
		}
	}

	for (const auto & id : sDevice->pipelineGraphicsHandles) {
		auto const & pipeline = sDevice->pipelineGraphicsPool.get(id);
		vkDestroyPipeline(sDevice->device, pipeline->pipeline, nullptr);
	}
	for (auto const & id : sDevice->pipelineComputeHandles) {
		auto const & pipeline = sDevice->pipelineComputePool.get(id);
		vkDestroyPipeline(sDevice->device, pipeline->pipeline, nullptr);
	}

	vmaDestroyAllocator(sDevice->allocator);
	vkDestroyPipelineLayout(
		sDevice->device, sDevice->pipelineLayoutUniversal, nullptr
	);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolGraphics, nullptr);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolCompute, nullptr);

	if (sDevice->window != nullptr) {
		// windowed: let vkb destroy swapchain image views, swapchain, surface
		sDevice->swapchain.destroy_image_views(sDevice->swapchainImageViews);
		vkb::destroy_swapchain(sDevice->swapchain);
		vkb::destroy_surface(sDevice->instance, sDevice->surface);
	}
	vkb::destroy_device(sDevice->device);
	vkb::destroy_instance(sDevice->instance);
	if (sDevice->window != nullptr) {
		glfwDestroyWindow(sDevice->window);
		glfwTerminate();
	}

	delete sDevice;
	sDevice = nullptr;
}

// -----------------------------------------------------------------------------
// -- pipeline creation
// -----------------------------------------------------------------------------

static VkPipeline build_compute_pipeline(
	ImplPipelineCompute const & impl
)
{
	VkShaderModule const computeShaderModule = (
		compile_and_load_shader_module(
			impl.pathCompute, ShaderStage::compute,
			impl.defines, impl.includePaths
		)
	);
	if (computeShaderModule == VK_NULL_HANDLE) {
		return VK_NULL_HANDLE;
	}
	VkPipelineShaderStageCreateInfo const shaderStageCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = computeShaderModule,
		.pName = "main",
		.pSpecializationInfo = nullptr,
	};
	VkComputePipelineCreateInfo const pipelineCi = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = shaderStageCi,
		.layout = sDevice->pipelineLayoutUniversal,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};
	VkPipeline pipeline;
	VkResult const result = vkCreateComputePipelines(
		sDevice->device,
		VK_NULL_HANDLE, // pipeline cache
		1, // createInfoCount
		&pipelineCi,
		nullptr, // allocator
		&pipeline
	);
	vkDestroyShaderModule(sDevice->device, computeShaderModule, nullptr);
	if (result != VK_SUCCESS) {
		printf("failed to create compute pipeline: %d\n", result);
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

static VkPipeline build_graphics_pipeline(
	ImplPipelineGraphics const & impl
)
{
	// -- compile mesh + fragment from GLSL source
	VkShaderModule const meshModule = (
		compile_and_load_shader_module(
			impl.pathMesh, ShaderStage::mesh,
			impl.defines, impl.includePaths
		)
	);
	VkShaderModule const fragmentModule = (
		compile_and_load_shader_module(
			impl.pathFragment, ShaderStage::fragment,
			impl.defines, impl.includePaths
		)
	);
	if (meshModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		// keep-old-on-failure: bail before creating anything.
		if (meshModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
		}
		if (fragmentModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, fragmentModule, nullptr);
		}
		return VK_NULL_HANDLE;
	}

	// -- dynamic rendering attachment formats
	std::vector<VkFormat> colorFormats;
	colorFormats.reserve(impl.colorAttachmentFormats.size());
	for (vkof::ImageFormat f : impl.colorAttachmentFormats) {
		colorFormats.push_back(to_vk_format(f));
	}
	VkFormat const depthFormat = to_vk_format(impl.depthAttachmentFormat);

	VkPipelineRenderingCreateInfo const renderingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.pNext = nullptr,
		.viewMask = 0,
		.colorAttachmentCount = (u32)colorFormats.size(),
		.pColorAttachmentFormats = colorFormats.data(),
		.depthAttachmentFormat = depthFormat,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};

	// -- mesh + fragment stages (vertex-pulling model: no vertex stage)
	VkPipelineShaderStageCreateInfo const stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_MESH_BIT_EXT,
			.module = meshModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
	};

	VkPipelineViewportStateCreateInfo const viewportState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = nullptr,
		.scissorCount = 1,
		.pScissors = nullptr,
	};

	VkPipelineRasterizationStateCreateInfo const rasterState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = to_vk_cull(impl.cullMode),
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo const multisampleState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE,
	};

	bool const alpha = (impl.blendMode == vkof::BlendMode::alpha_blend);
	VkPipelineColorBlendAttachmentState const blendAttachment = {
		.blendEnable = (alpha ? VK_TRUE : VK_FALSE),
		.srcColorBlendFactor = (
			alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE
		),
		.dstColorBlendFactor = (
			alpha
				? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
				: VK_BLEND_FACTOR_ZERO
		),
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = (
			  VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT
			| VK_COLOR_COMPONENT_A_BIT
		),
	};
	std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(
		colorFormats.size(), blendAttachment
	);
	VkPipelineColorBlendStateCreateInfo const blendState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = (u32)blendAttachments.size(),
		.pAttachments = blendAttachments.data(),
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
	};

	VkBool32 depthTest { VK_FALSE };
	VkBool32 depthWrite { VK_FALSE };
	depth_test_flags(impl.depthTest, depthTest, depthWrite);
	VkPipelineDepthStencilStateCreateInfo const depthState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthTestEnable = depthTest,
		.depthWriteEnable = depthWrite,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {},
		.back = {},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	};

	VkDynamicState const dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo const dynamicState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo const createInfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCreateInfo,
		.flags = 0,
		.stageCount = 2,
		.pStages = stages,
		// mesh pipeline: no vertex input / input assembly
		.pVertexInputState = nullptr,
		.pInputAssemblyState = nullptr,
		.pTessellationState = nullptr,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthState,
		.pColorBlendState = &blendState,
		.pDynamicState = &dynamicState,
		.layout = sDevice->pipelineLayoutUniversal,
		.renderPass = VK_NULL_HANDLE,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};

	VkPipeline pipeline;
	VkAssert(
		vkCreateGraphicsPipelines(
			sDevice->device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline
		)
	);

	// modules consumed by pipeline creation; safe to destroy now
	vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
	vkDestroyShaderModule(sDevice->device, fragmentModule, nullptr);
	return pipeline;
}

// -----------------------------------------------------------------------------
// -- shader compilation
// -----------------------------------------------------------------------------

// last compile error, surfaced so a failed hot reload can be shown
// rather than crashing the session.
static std::string sLastCompileError;

static char const * shader_stage_flag(ShaderStage const & stage) {
	switch (stage) {
		case ShaderStage::fragment: return "frag";
		case ShaderStage::compute:  return "comp";
		case ShaderStage::mesh:     return "mesh";
	}
	return "vert";
}

static CompileResult system_capture(char const * cmd) {
	std::string output;
	FILE * pipe = popen(cmd, "r");
	if (!pipe) { return { -1, "failed to launch compiler" }; }
	char buffer[256];
	while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
		output += buffer;
	}
	int const status = pclose(pipe);
	return { status, output };
}

// compiles glsl -> spv, returns the module. on failure returns
// VK_NULL_HANDLE and leaves sLastCompileError set (caller decides whether
// to keep the old pipeline or abort).
static VkShaderModule compile_and_load_shader_module(
	std::string const & glslPath,
	ShaderStage stage,
	std::vector<std::string> const & defines,
	std::vector<std::string> const & includePaths
) {
	std::string const shaderDir = (
		glslPath.substr(0, glslPath.rfind('/') + 1)
	);
	std::string const baseName = (
		glslPath.substr(glslPath.rfind('/') + 1)
	);
	std::string const binDir = shaderDir + "bin";
	std::string const spvPath = binDir + "/" + baseName + ".spv";
	std::filesystem::create_directories(binDir);

	std::string cmd = (
		std::string("glslangValidator -S ")
		+ shader_stage_flag(stage)
		+ " --target-env vulkan1.3"
	);
	for (std::string const & d : defines) {
		cmd += " -D" + d;
	}
	for (std::string const & p : includePaths) {
		cmd += " -I" + p;
	}
	cmd += " -o " + spvPath + " " + glslPath;
	CompileResult const result = system_capture(cmd.c_str());
	if (result.exitCode != 0) {
		printf(
			"failed to compile shader %s:\n%s\n",
			glslPath.c_str(), result.output.c_str()
		);
		sLastCompileError = result.output;
		return VK_NULL_HANDLE;
	}
	sLastCompileError.clear();

	FILE * f = fopen(spvPath.c_str(), "rb");
	if (!f) {
		printf("failed to open compiled spv: %s\n", spvPath.c_str());
		return VK_NULL_HANDLE;
	}
	fseek(f, 0, SEEK_END);
	size_t const size = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<u8> code(size);
	fread(code.data(), 1, size, f);
	fclose(f);

	VkShaderModuleCreateInfo const ci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = code.size(),
		.pCode = (u32 const *)code.data(),
	};
	VkShaderModule mod;
	VkAssert(vkCreateShaderModule(sDevice->device, &ci, nullptr, &mod));
	return mod;
}

// -----------------------------------------------------------------------------
// -- shader hot reload
// -----------------------------------------------------------------------------

static bool file_changed(
	std::string const & path,
	std::filesystem::file_time_type const & lastWriteTime
) {
	return file_write_time(path) != lastWriteTime;
}

static void pipeline_hot_reload()
{
	// -- check if compute or graphics pipeilne changed
	bool anyChanged = false;
	for (auto const & id : sDevice->pipelineGraphicsHandles) {
		ImplPipelineGraphics const & impl = (
			*sDevice->pipelineGraphicsPool.get(id)
		);
		if (file_changed(impl.pathFragment, impl.lastWriteTimeFragment)) {
			anyChanged = true;
			break;
		}
		if (file_changed(impl.pathMesh, impl.lastWriteTimeMesh)) {
			anyChanged = true;
			break;
		}
	}
	if (!anyChanged) {
		for (auto const & id : sDevice->pipelineComputeHandles) {
			ImplPipelineCompute const & impl = (
				*sDevice->pipelineComputePool.get(id)
			);
			if (file_changed(impl.pathCompute, impl.lastWriteTimeCompute)) {
				anyChanged = true;
				break;
			}
		}
	}

	if (!anyChanged) {
		return;
	}

	// -- reload everything
	vkDeviceWaitIdle(sDevice->device);

	// for (auto & [id, impl] : sDevice->pipelineGraphicsPool) {
	for (auto const & id : sDevice->pipelineGraphicsHandles) {
		ImplPipelineGraphics & impl = (
			*sDevice->pipelineGraphicsPool.get(id)
		);
		bool const vtxChanged = (
			file_changed(impl.pathMesh, impl.lastWriteTimeMesh)
		);
		bool const frgChanged = (
			file_changed(impl.pathFragment, impl.lastWriteTimeFragment)
		);
		if (!vtxChanged && !frgChanged) { continue; }

		VkPipeline const rebuilt = build_graphics_pipeline(impl);
		if (rebuilt == VK_NULL_HANDLE) {
			// compile failed: keep the old pipeline, do NOT advance the
			// tracked write times, so the next save retries automatically.
			printf(
				"hot reload kept old pipeline '%s/%s' (compile failed)\n",
				impl.pathMesh.c_str(), impl.pathFragment.c_str()
			);
			continue;
		}

		VkPipeline const old = impl.pipeline;
		impl.pipeline = rebuilt;
		impl.lastWriteTimeMesh = file_write_time(impl.pathMesh);
		impl.lastWriteTimeFragment = file_write_time(impl.pathFragment);
		vkDestroyPipeline(sDevice->device, old, nullptr);
		printf(
			"hot reloaded pipeline '%s/%s'\n",
			impl.pathMesh.c_str(), impl.pathFragment.c_str()
		);
	}

	// -- compute
	for (auto const & id : sDevice->pipelineComputeHandles) {
		ImplPipelineCompute & impl = (
			*sDevice->pipelineComputePool.get(id)
		);
		if (!file_changed(impl.pathCompute, impl.lastWriteTimeCompute)) {
			continue;
		}

		VkPipeline const rebuilt = build_compute_pipeline(impl);
		if (rebuilt == VK_NULL_HANDLE) {
			printf(
				"hot reload kept old compute pipeline '%s' (compile failed)\n",
				impl.pathCompute.c_str()
			);
			continue;
		}

		VkPipeline const old = impl.pipeline;
		impl.pipeline = rebuilt;
		impl.lastWriteTimeCompute = file_write_time(impl.pathCompute);
		vkDestroyPipeline(sDevice->device, old, nullptr);
		printf("hot reloaded compute pipeline '%s'\n", impl.pathCompute.c_str());
	}
}

// -----------------------------------------------------------------------------
// -- pipeline creation
// -----------------------------------------------------------------------------

vkof::Pipeline vkof::pipeline_graphics_create(
	PipelineGraphicsCreateInfo const & createInfo
) {
	std::vector<vkof::ImageFormat> const colorFormats(
		/*__first=*/ createInfo.attachmentColorFormats.ptr(),
		/*__last=*/ (
			  createInfo.attachmentColorFormats.ptr()
			+ createInfo.attachmentColorFormats.size()
		)
	);
	auto implPipeline = ImplPipelineGraphics {
		.pipeline = VK_NULL_HANDLE,
		.colorAttachmentFormats = std::move(colorFormats),
		.depthAttachmentFormat = createInfo.attachmentDepthStencilFormat,
		.depthTest = createInfo.depthTest,
		.cullMode = createInfo.cullMode,
		.blendMode = createInfo.blendMode,

		.pathFragment = createInfo.pathFragment,
		.pathMesh = createInfo.pathMesh,
		.defines = std::vector<std::string>(
			createInfo.defines.ptr(),
			createInfo.defines.ptr() + createInfo.defines.size()
		),
		.includePaths = std::vector<std::string>(
			createInfo.includePaths.ptr(),
			createInfo.includePaths.ptr() + createInfo.includePaths.size()
		),

		.lastWriteTimeFragment = file_write_time(createInfo.pathFragment),
		.lastWriteTimeMesh = file_write_time(createInfo.pathMesh),
	};
	implPipeline.pipeline = build_graphics_pipeline(implPipeline);
	if (implPipeline.pipeline == VK_NULL_HANDLE) {
		printf(
			"failed to create graphics pipeline '%s | %s'\n",
			createInfo.pathMesh, createInfo.pathFragment
		);
		return vkof::Pipeline { .id = 0 };
	}
	vkof::Pipeline const handle = (
		sDevice->pipelineGraphicsPool.allocate(implPipeline)
	);
	sDevice->pipelineGraphicsHandles.emplace_back(handle.id);
	return handle;
}

vkof::Pipeline vkof::pipeline_compute_create(
	vkof::PipelineComputeCreateInfo const & createInfo
) {
	ImplPipelineCompute implPipeline;
	implPipeline.pathCompute = createInfo.pathCompute;
	implPipeline.defines = std::vector<std::string>(
		createInfo.defines.ptr(),
		createInfo.defines.ptr() + createInfo.defines.size()
	);
	implPipeline.includePaths = std::vector<std::string>(
		createInfo.includePaths.ptr(),
		createInfo.includePaths.ptr() + createInfo.includePaths.size()
	);
	implPipeline.layout = sDevice->pipelineLayoutUniversal;
	implPipeline.pipeline = build_compute_pipeline(implPipeline);
	if (implPipeline.pipeline == VK_NULL_HANDLE) {
		printf(
			"failed to create compute pipeline '%s'\n",
			createInfo.pathCompute
		);
		return vkof::Pipeline { .id = 0 };
	}
	implPipeline.lastWriteTimeCompute = (
		file_write_time(implPipeline.pathCompute)
	);
	auto const handle = sDevice->pipelineComputePool.allocate(implPipeline);
	sDevice->pipelineComputeHandles.emplace_back(handle.id);
	return handle;
}

void vkof::pipeline_destroy(Pipeline const & pipeline)
{
	// try graphics pool first
	if (auto const impl = sDevice->pipelineGraphicsPool.get(pipeline)) {
		vkDestroyPipeline(sDevice->device, impl->pipeline, nullptr);
		sDevice->pipelineGraphicsPool.free(pipeline);
		auto & v = sDevice->pipelineGraphicsHandles;
		v.erase(
			std::remove_if(
				v.begin(), v.end(),
				[&](vkof::Pipeline const & h) {
					return h.id == pipeline.id;
				}
			),
			v.end()
		);
		return;
	}
	// then compute pool
	if (auto const impl = sDevice->pipelineComputePool.get(pipeline)) {
		vkDestroyPipeline(sDevice->device, impl->pipeline, nullptr);
		sDevice->pipelineComputePool.free(pipeline);
		auto & v = sDevice->pipelineComputeHandles;
		v.erase(
			std::remove_if(
				v.begin(), v.end(),
				[&](vkof::Pipeline const & h) {
					return h.id == pipeline.id;
				}
			),
			v.end()
		);
		return;
	}
}

// -----------------------------------------------------------------------------
// -- draw/dispatch/copy
// -----------------------------------------------------------------------------

void vkof::cmd_draw(CmdDraw const & draw)
{
	auto const implCmd = sDevice->commandBufferPool.get(draw.cmd);
	auto const implPipeline = sDevice->pipelineGraphicsPool.get(draw.pipeline);
	if (!implCmd || !implPipeline) {
		return;
	}

	// bind pipeline only if it changed on this command buffer
	if (
		   implCmd->boundPipeline.id != draw.pipeline.id
		|| implCmd->bindPoint != VK_PIPELINE_BIND_POINT_GRAPHICS
	) {
		vkCmdBindPipeline(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			implPipeline->pipeline
		);
		implCmd->boundPipeline = draw.pipeline;
		implCmd->bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	}

	// per-draw push goes to the [128,256) window
	if (draw.pushconstant.size() > 0) {
		vkCmdPushConstants(
			implCmd->cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 128,
			(u32)draw.pushconstant.size(),
			draw.pushconstant.ptr()
		);
	}

	pfnVkCmdDrawMeshTasksEXT(
		implCmd->cmd,
		/*groupCountX=*/ draw.vertexCount,
		/*groupCountY=*/ draw.instanceCount,
		/*groupCountZ=*/ 1
	);
}

void vkof::cmd_dispatch(CmdDispatch const & dispatch)
{
	auto const implCmd = sDevice->commandBufferPool.get(dispatch.cmd);
	auto const implPipeline = (
		sDevice->pipelineComputePool.get(dispatch.pipeline)
	);
	if (!implCmd || !implPipeline) {
		return;
	}

	if (
		   implCmd->boundPipeline.id != dispatch.pipeline.id
		|| implCmd->bindPoint != VK_PIPELINE_BIND_POINT_COMPUTE
	) {
		vkCmdBindPipeline(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			implPipeline->pipeline
		);
		implCmd->boundPipeline = dispatch.pipeline;
		implCmd->bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
	}

	if (dispatch.pushconstant.size() > 0) {
		vkCmdPushConstants(
			implCmd->cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 128,
			(u32)dispatch.pushconstant.size(),
			dispatch.pushconstant.ptr()
		);
	}

	vkCmdDispatch(
		implCmd->cmd,
		dispatch.groupCountX,
		dispatch.groupCountY,
		dispatch.groupCountZ
	);
}

static bool resolve_image(
	vkof::Image const & image,
	VkImage & outImage,
	u32 & outWidth,
	u32 & outHeight
) {
	if (image.id == (u64)-1) {
		outImage = sDevice->swapchainImages[sDevice->swapchainImageIndex];
		outWidth = sDevice->swapchain.extent.width;
		outHeight = sDevice->swapchain.extent.height;
		return true;
	}
	if (auto const impl = sDevice->imagePool.get(image)) {
		outImage = impl->image;
		outWidth = impl->width;
		outHeight = impl->height;
		return true;
	}
	return false;
}

void vkof::cmd_copy_image(CmdCopyImage const & copy)
{
	auto const implCmd = sDevice->commandBufferPool.get(copy.cmd);
	if (!implCmd) {
		return;
	}

	VkImage srcImage;
	VkImage dstImage;
	u32 srcWidth, srcHeight, dstWidth, dstHeight;
	if (
		   !resolve_image(copy.srcImage, srcImage, srcWidth, srcHeight)
		|| !resolve_image(copy.dstImage, dstImage, dstWidth, dstHeight)
	) {
		return;
	}

	VkImageCopy const region = {
		.srcSubresource = VkImageSubresourceLayers {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = copy.srcBaseMipLevel,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.srcOffset = VkOffset3D {
			.x = (i32)copy.srcOffsetX,
			.y = (i32)copy.srcOffsetY,
			.z = 0,
		},
		.dstSubresource = VkImageSubresourceLayers {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = copy.dstBaseMipLevel,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.dstOffset = VkOffset3D {
			.x = (i32)copy.dstOffsetX,
			.y = (i32)copy.dstOffsetY,
			.z = 0,
		},
		.extent = VkExtent3D {
			.width = copy.width,
			.height = copy.height,
			.depth = 1,
		},
	};

	// the graph leaves images in GENERAL at pass boundaries, so both ends
	// are GENERAL here; no transition needed inside the copy.
	vkCmdCopyImage(
		implCmd->cmd,
		srcImage,
		VK_IMAGE_LAYOUT_GENERAL,
		dstImage,
		VK_IMAGE_LAYOUT_GENERAL,
		1,
		&region
	);
}

// -----------------------------------------------------------------------------
// -- render graph
// -----------------------------------------------------------------------------

GLFWwindow * vkof::window()
{
	return sDevice->window;
}

void vkof::imgui_begin()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	sDevice->imguiFrameActive = true;
}

static void profiler_init()
{
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(sDevice->physicalDevice, &props);

	u32 qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		sDevice->physicalDevice, &qfCount, nullptr
	);
	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(
		sDevice->physicalDevice, &qfCount, qfProps.data()
	);

	auto const gfxFamilyResult = sDevice->device.get_queue_index(vkb::QueueType::graphics);
	if (!gfxFamilyResult.has_value()) { return; }
	u32 const gfxFamily = gfxFamilyResult.value();
	if (gfxFamily >= qfCount || qfProps[gfxFamily].timestampValidBits == 0) { return; }

	VkQueryPoolCreateInfo const ci = {
		.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queryType = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = kMaxProfiledNodes * 2 * kFramesInFlight,
		.pipelineStatistics = 0,
	};
	if (vkCreateQueryPool(sDevice->device, &ci, nullptr, &sDevice->profiler.queryPool) != VK_SUCCESS) {
		return;
	}
	sDevice->profiler.timestampPeriodMs = (double)props.limits.timestampPeriod / 1e6;
	sDevice->profiler.gpuSupported = true;
}

void vkof::render_graph_execute(RenderGraphExecuteInfo const & exec)
{
	u32 const frameSlot = sDevice->frameIndex % kFramesInFlight;
	auto & fd = sDevice->frameData[frameSlot];

	// -- wait for the previous use of this frame slot to finish
	VkAssert(
		vkWaitForFences(
			sDevice->device, 1, &fd.fence, VK_TRUE, UINT64_MAX
		)
	);
	VkAssert(vkResetFences(sDevice->device, 1, &fd.fence));

	// -- read back GPU timestamps from the previous use of this frame slot
	//    (fence above guarantees GPU is done, so results are available)
	Profiler & prof = sDevice->profiler;
	{
		u32 const prevCount = prof.slotNodeCounts[frameSlot];
		if (prof.gpuSupported && prevCount > 0) {
			std::array<u64, kMaxProfiledNodes * 2> ts {};
			VkResult const qr = vkGetQueryPoolResults(
				sDevice->device, prof.queryPool,
				frameSlot * kMaxProfiledNodes * 2, prevCount * 2,
				prevCount * 2 * sizeof(u64), ts.data(),
				sizeof(u64), VK_QUERY_RESULT_64_BIT
			);
			if (qr == VK_SUCCESS) {
				for (u32 i = 0; i < prevCount; ++i) {
					u64 const t0 = ts[i * 2 + 0];
					u64 const t1 = ts[i * 2 + 1];
					prof.timings[i].gpuMs = (double)(t1 - t0) * prof.timestampPeriodMs;
				}
			}
		}
		prof.displayCount = prevCount;
		prof.frameNodeCount = 0;
	}

	// -- acquire the next swapchain image (windowed only)
	if (sDevice->window != nullptr) {
		VkAssert(
			vkAcquireNextImageKHR(
				sDevice->device,
				sDevice->swapchain.swapchain,
				UINT64_MAX,
				fd.semaphoreAcquire,
				VK_NULL_HANDLE,
				&sDevice->swapchainImageIndex
			)
		);
	}

	VkCommandBuffer const cmd = fd.cmdGraphics;
	VkAssert(vkResetCommandBuffer(cmd, 0));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	// -- reset timestamp queries for this frame slot
	if (prof.gpuSupported) {
		vkCmdResetQueryPool(
			cmd, prof.queryPool,
			frameSlot * kMaxProfiledNodes * 2, kMaxProfiledNodes * 2
		);
	}

	// -- root push constants go in the [0, 128) window
	if (exec.rootPushconstant.size() > 0) {
		vkCmdPushConstants(
			cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 0,
			(u32)exec.rootPushconstant.size(),
			exec.rootPushconstant.ptr()
		);
	}

	// -- collect every unique transient image referenced by any node so we
	//    can batch all UNDEFINED->GENERAL transitions into one barrier call.
	//    the swapchain image always gets the same treatment.
	std::vector<VkImageMemoryBarrier2> initBarriers;
	{
		std::vector<u64> seen; // transientImage ids already added

		auto const addTransient = [&](vkof::TransientImage const & ti) {
			if (!ti.id) { return; }
			for (u64 id : seen) {
				if (id == ti.id) { return; }
			}
			seen.push_back(ti.id);

			vkof::Image const img = vkof::transient_image_get_image(ti);
			auto const implTex = sDevice->imagePool.get(img);
			if (!implTex) { return; }

			VkFormat const fmt = to_vk_format(
				sDevice->transientImagePool.get(ti)->format
			);
			VkImageAspectFlags const aspect = format_aspect(fmt);

			initBarriers.push_back(VkImageMemoryBarrier2 {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.pNext = nullptr,
				.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = implTex->image,
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask     = aspect,
					.baseMipLevel   = 0,
					.levelCount     = VK_REMAINING_MIP_LEVELS,
					.baseArrayLayer = 0,
					.layerCount     = 1,
				},
			});
		};

		for (auto const & nodeHandle : exec.nodes) {
			auto const node = sDevice->renderNodePool.get(nodeHandle);
			if (!node) { continue; }
			for (auto const & b : node->imageBarriers) {
				addTransient(b.transientImage);
			}
			for (auto const & att : node->colorAttachments) {
				addTransient(att.image);
			}
			if (node->depthAttachment) {
				addTransient(node->depthAttachment->image);
			}
		}

		// always transition the swapchain image UNDEFINED->GENERAL
		initBarriers.push_back(VkImageMemoryBarrier2 {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sDevice->swapchainImages[sDevice->swapchainImageIndex],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		});
	}
	if (!initBarriers.empty()) {
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = (u32)initBarriers.size(),
			.pImageMemoryBarriers   = initBarriers.data(),
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// -- allocate one command-buffer handle to hand to all node callbacks
	ImplCommandBuffer implCmdBuf = {
		.cmd = cmd,
		.boundPipeline = { .id = 0 },
		.bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	};
	vkof::CommandBuffer const cmdHandle = (
		sDevice->commandBufferPool.allocate(implCmdBuf)
	);

	// -- global memory barrier emitted between nodes
	VkMemoryBarrier2 const nodeBoundaryBarrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = (
			VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
		),
		.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = (
			VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
		),
	};
	VkDependencyInfo const nodeBoundaryDep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &nodeBoundaryBarrier,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};

	// -- execute each node in declared order
	bool firstNode = true;
	for (auto const & nodeHandle : exec.nodes) {
		auto const node = sDevice->renderNodePool.get(nodeHandle);
		if (!node) { continue; }

		u32 const nodeIdx = prof.frameNodeCount++;
		bool const canProfile = prof.gpuSupported && nodeIdx < kMaxProfiledNodes;
		auto const cpuT0 = std::chrono::steady_clock::now();

		if (canProfile) {
			vkCmdWriteTimestamp2(
				cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				prof.queryPool, frameSlot * kMaxProfiledNodes * 2 + nodeIdx * 2
			);
		}

		// barrier between nodes (skip before the very first)
		if (!firstNode) {
			vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);
		}
		firstNode = false;

		// -- dynamic rendering setup if this node has attachments
		bool const hasColor = !node->colorAttachments.empty();
		bool const hasDepth = node->depthAttachment.has_value();
		if (hasColor || hasDepth) {
			// determine render area from the first attachment's image
			u32 renderW = sDevice->swapchain.extent.width;
			u32 renderH = sDevice->swapchain.extent.height;
			if (hasColor) {
				auto const & firstAtt = node->colorAttachments[0];
				auto const tiImpl = (
					sDevice->transientImagePool.get(firstAtt.image)
				);
				if (tiImpl) {
					renderW = (u32)(
						sDevice->swapchain.extent.width * tiImpl->scaleWidth
					);
					renderH = (u32)(
						sDevice->swapchain.extent.height * tiImpl->scaleHeight
					);
				}
			} else if (hasDepth) {
				auto const tiImpl = (
					sDevice->transientImagePool.get(
						node->depthAttachment->image
					)
				);
				if (tiImpl) {
					renderW = (u32)(
						sDevice->swapchain.extent.width * tiImpl->scaleWidth
					);
					renderH = (u32)(
						sDevice->swapchain.extent.height * tiImpl->scaleHeight
					);
				}
			}

			// sort color attachments by colorIndex so pColorAttachments
			// is in the right order.
			u32 maxColorIdx = 0;
			for (auto const & att : node->colorAttachments) {
				if (att.colorIndex > maxColorIdx) {
					maxColorIdx = att.colorIndex;
				}
			}
			std::vector<VkRenderingAttachmentInfo> colorInfos(
				maxColorIdx + 1,
				VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = VK_NULL_HANDLE,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = {},
				}
			);
			for (auto const & att : node->colorAttachments) {
				vkof::Image const img = (
					vkof::transient_image_get_image(att.image)
				);
				auto const tex = sDevice->imagePool.get(img);
				if (!tex) { continue; }
				VkImageView const view = (
					att.mipLevel == 0
						? tex->imageViewFull
						: image_storage_view_for_mip(*tex, att.mipLevel)
				);
				auto const toLoadOp = [](vkof::RenderNodeLoadOp op) {
					switch (op) {
						case vkof::RenderNodeLoadOp::clear:
							return VK_ATTACHMENT_LOAD_OP_CLEAR;
						case vkof::RenderNodeLoadOp::discard:
							return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
						default:
							return VK_ATTACHMENT_LOAD_OP_LOAD;
					}
				};
				colorInfos[att.colorIndex] = VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = view,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = toLoadOp(att.loadOp),
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = VkClearValue {
						.color = {
							.float32 = {
								att.clearColor[0],
								att.clearColor[1],
								att.clearColor[2],
								att.clearColor[3],
							},
						},
					},
				};
			}

			VkRenderingAttachmentInfo depthInfo {};
			if (hasDepth) {
				auto const & datt = *node->depthAttachment;
				vkof::Image const img = (
					vkof::transient_image_get_image(datt.image)
				);
				auto const tex = sDevice->imagePool.get(img);
				VkImageView const view = (
					tex
						? (datt.mipLevel == 0
							? tex->imageViewFull
							: image_storage_view_for_mip(*tex, datt.mipLevel))
						: VK_NULL_HANDLE
				);
				auto const toLoadOp = [](vkof::RenderNodeLoadOp op) {
					switch (op) {
						case vkof::RenderNodeLoadOp::clear:
							return VK_ATTACHMENT_LOAD_OP_CLEAR;
						case vkof::RenderNodeLoadOp::discard:
							return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
						default:
							return VK_ATTACHMENT_LOAD_OP_LOAD;
					}
				};
				depthInfo = VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = view,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = toLoadOp(datt.loadOp),
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = VkClearValue {
						.depthStencil = { datt.clearDepth, 0 },
					},
				};
			}

			VkRenderingInfo const renderingInfo = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.pNext = nullptr,
				.flags = 0,
				.renderArea = VkRect2D {
					.offset = { 0, 0 },
					.extent = { renderW, renderH },
				},
				.layerCount = 1,
				.viewMask = 0,
				.colorAttachmentCount = (u32)colorInfos.size(),
				.pColorAttachments = colorInfos.data(),
				.pDepthAttachment = hasDepth ? &depthInfo : nullptr,
				.pStencilAttachment = nullptr,
			};
			vkCmdBeginRendering(cmd, &renderingInfo);

			// dynamic viewport and scissor to match the render area
			VkViewport const viewport = {
				.x = 0.0f, .y = 0.0f,
				.width = (f32)renderW, .height = (f32)renderH,
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};
			VkRect2D const scissor = {
				.offset = { 0, 0 },
				.extent = { renderW, renderH },
			};
			vkCmdSetViewport(cmd, 0, 1, &viewport);
			vkCmdSetScissor(cmd, 0, 1, &scissor);
		}

		// -- invoke the node's recording callback
		if (node->callback) {
			node->callback(cmdHandle);
		}

		if (hasColor || hasDepth) {
			vkCmdEndRendering(cmd);
		}

		if (canProfile) {
			vkCmdWriteTimestamp2(
				cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
				prof.queryPool, frameSlot * kMaxProfiledNodes * 2 + nodeIdx * 2 + 1
			);
		}
		if (nodeIdx < kMaxProfiledNodes) {
			prof.timings[nodeIdx].cpuMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - cpuT0
			).count();
		}
	}

	prof.slotNodeCounts[frameSlot] = prof.frameNodeCount;

	sDevice->commandBufferPool.free(cmdHandle);

	if (sDevice->window != nullptr && exec.finalImage.id != 0) {
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		vkof::Image const srcImg = vkof::transient_image_get_image(exec.finalImage);
		VkImage srcVkImage;
		u32 srcW, srcH;
		if (resolve_image(srcImg, srcVkImage, srcW, srcH)) {
			VkImage const dstVkImage = (
				sDevice->swapchainImages[sDevice->swapchainImageIndex]
			);
			u32 const copyW = sDevice->swapchain.extent.width;
			u32 const copyH = sDevice->swapchain.extent.height;
			VkImageCopy const region = {
				.srcSubresource = VkImageSubresourceLayers {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffset = { 0, 0, 0 },
				.dstSubresource = VkImageSubresourceLayers {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffset = { 0, 0, 0 },
				.extent = { copyW, copyH, 1 },
			};
			vkCmdCopyImage(
				cmd,
				srcVkImage, VK_IMAGE_LAYOUT_GENERAL,
				dstVkImage, VK_IMAGE_LAYOUT_GENERAL,
				1, &region
			);
		}
	}

	if (sDevice->window != nullptr && sDevice->imguiFrameActive) {
		sDevice->imguiFrameActive = false;
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		// -- profiler window
		if (prof.displayCount > 0) {
			ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
			ImGui::Begin("profiler", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("%-6s  %8s  %8s", "node", "cpu ms", "gpu ms");
			ImGui::Separator();
			for (u32 i = 0; i < prof.displayCount; ++i) {
				NodeTiming const & t = prof.timings[i];
				ImGui::Text("node %-2u  %8.3f  %8.3f", i, t.cpuMs, t.gpuMs);
			}
			ImGui::End();
		}

		// -- vma memory dashboard
		{
			VkPhysicalDeviceMemoryProperties memProps;
			vkGetPhysicalDeviceMemoryProperties(
				sDevice->physicalDevice, &memProps
			);
			VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(sDevice->allocator, budgets);

			ImGui::SetNextWindowPos(ImVec2(10.0f, 150.0f), ImGuiCond_FirstUseEver);
			ImGui::Begin("memory", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			for (u32 i = 0; i < memProps.memoryHeapCount; ++i) {
				VmaBudget const & b = budgets[i];
				bool const isDevice = (
					(
						memProps.memoryHeaps[i].flags
						& VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
					) != 0
				);
				f32 const usedMb = (f32)b.usage / (1024.0f * 1024.0f);
				f32 const budgetMb = (f32)b.budget / (1024.0f * 1024.0f);
				f32 const fraction = budgetMb > 0.0f ? usedMb / budgetMb : 0.0f;
				ImGui::Text("heap %u  %s", i, isDevice ? "device" : "host");
				ImGui::ProgressBar(fraction, ImVec2(180.0f, 0.0f));
				ImGui::SameLine();
				ImGui::Text("%.0f / %.0f MB", usedMb, budgetMb);
				ImGui::Text(
					"  allocs %u  blocks %u",
					b.statistics.allocationCount, b.statistics.blockCount
				);
			}
			ImGui::End();
		}

		// -- shader error overlay
		if (!sLastCompileError.empty()) {
			ImGui::SetNextWindowPos(ImVec2(10.0f, 400.0f), ImGuiCond_FirstUseEver);
			ImGui::SetNextWindowSize(ImVec2(620.0f, 260.0f), ImGuiCond_FirstUseEver);
			ImGui::PushStyleColor(
				ImGuiCol_TitleBg, ImVec4(0.5f, 0.05f, 0.05f, 1.0f)
			);
			ImGui::PushStyleColor(
				ImGuiCol_TitleBgActive, ImVec4(0.7f, 0.05f, 0.05f, 1.0f)
			);
			ImGui::Begin("shader error", nullptr, 0);
			ImGui::PopStyleColor(2);
			ImGui::TextUnformatted(sLastCompileError.c_str());
			ImGui::End();
		}

		ImGui::Render();
		ImDrawData * const imguiDrawData = ImGui::GetDrawData();

		VkRenderingAttachmentInfo const imguiColorAtt = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.pNext = nullptr,
			.imageView = sDevice->swapchainImageViews[sDevice->swapchainImageIndex],
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {},
		};
		VkRenderingInfo const imguiRenderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.pNext = nullptr,
			.flags = 0,
			.renderArea = VkRect2D {
				.offset = { 0, 0 },
				.extent = {
					sDevice->swapchain.extent.width,
					sDevice->swapchain.extent.height,
				},
			},
			.layerCount = 1,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = &imguiColorAtt,
			.pDepthAttachment = nullptr,
			.pStencilAttachment = nullptr,
		};
		vkCmdBeginRendering(cmd, &imguiRenderingInfo);
		if (imguiDrawData) {
			ImGui_ImplVulkan_RenderDrawData(imguiDrawData, cmd);
		}
		vkCmdEndRendering(cmd);
	}

	// -- transition swapchain GENERAL -> PRESENT_SRC (windowed only)
	if (sDevice->window != nullptr) {
		VkImageMemoryBarrier2 const presentBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
			.dstAccessMask = VK_ACCESS_2_NONE,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sDevice->swapchainImages[sDevice->swapchainImageIndex],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		};
		VkDependencyInfo const presentDep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &presentBarrier,
		};
		vkCmdPipelineBarrier2(cmd, &presentDep);
	}

	VkAssert(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo const cmdSubmit = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.pNext = nullptr,
		.commandBuffer = cmd,
		.deviceMask = 0,
	};
	if (sDevice->window != nullptr) {
		// windowed: wait on acquire semaphore, signal render semaphore + fence
		VkSemaphoreSubmitInfo const waitInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.pNext = nullptr,
			.semaphore = fd.semaphoreAcquire,
			.value = 0,
			.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.deviceIndex = 0,
		};
		VkSemaphore const renderSem = (
			sDevice->semaphoresRender[sDevice->swapchainImageIndex]
		);
		VkSemaphoreSubmitInfo const signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.pNext = nullptr,
			.semaphore = renderSem,
			.value = 0,
			.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.deviceIndex = 0,
		};
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount   = 1,
			.pWaitSemaphoreInfos      = &waitInfo,
			.commandBufferInfoCount   = 1,
			.pCommandBufferInfos      = &cmdSubmit,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos    = &signalInfo,
		};
		VkAssert(
			vkQueueSubmit2(
				sDevice->queueGraphics, 1, &submitInfo, fd.fence
			)
		);
		VkPresentInfoKHR const presentInfo = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &renderSem,
			.swapchainCount = 1,
			.pSwapchains = &sDevice->swapchain.swapchain,
			.pImageIndices = &sDevice->swapchainImageIndex,
			.pResults = nullptr,
		};
		vkQueuePresentKHR(sDevice->queueGraphics, &presentInfo);
	} else {
		// headless: no acquire/present semaphores, submit with fence only
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount   = 0,
			.pWaitSemaphoreInfos      = nullptr,
			.commandBufferInfoCount   = 1,
			.pCommandBufferInfos      = &cmdSubmit,
			.signalSemaphoreInfoCount = 0,
			.pSignalSemaphoreInfos    = nullptr,
		};
		VkAssert(
			vkQueueSubmit2(
				sDevice->queueGraphics, 1, &submitInfo, fd.fence
			)
		);
	}

	++sDevice->frameIndex;
	pipeline_hot_reload();
}

void vkof::screenshot(
	vkof::TransientImage const & image,
	char const * const path
) {
	vkof::device_wait_idle();

	vkof::Image const img = vkof::transient_image_get_image(image);
	VkImage vkImg;
	u32 w, h;
	if (!resolve_image(img, vkImg, w, h)) { return; }

	u64 const byteCount = (u64)w * h * 4u;

	VkBuffer stagingBuf;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo allocInfo {};
	{
		VkBufferCreateInfo const bci = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = byteCount,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
		};
		VmaAllocationCreateInfo const aci = {
			.flags = (
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
				| VMA_ALLOCATION_CREATE_MAPPED_BIT
			),
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = 0,
			.preferredFlags = 0,
			.memoryTypeBits = 0,
			.pool = nullptr,
			.pUserData = nullptr,
			.priority = 0.0f,
		};
		VkAssert(vmaCreateBuffer(
			sDevice->allocator, &bci, &aci, &stagingBuf, &stagingAlloc, &allocInfo
		));
	}

	VkCommandPool const pool = sDevice->commandPoolGraphics;
	VkCommandBuffer cmd;
	{
		VkCommandBufferAllocateInfo const cmdAlloc = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1u,
		};
		VkAssert(vkAllocateCommandBuffers(sDevice->device, &cmdAlloc, &cmd));
	}

	{
		VkCommandBufferBeginInfo const beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			.pInheritanceInfo = nullptr,
		};
		VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));
	}

	VkImageSubresourceRange const fullImage = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0u,
		.levelCount = 1u,
		.baseArrayLayer = 0u,
		.layerCount = 1u,
	};

	{
		VkImageMemoryBarrier2 const barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkImg,
			.subresourceRange = fullImage,
		};
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1u,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	{
		VkBufferImageCopy const region = {
			.bufferOffset = 0u,
			.bufferRowLength = 0u,
			.bufferImageHeight = 0u,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0u,
				.baseArrayLayer = 0u,
				.layerCount = 1u,
			},
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { w, h, 1u },
		};
		vkCmdCopyImageToBuffer(
			cmd, vkImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			stagingBuf, 1u, &region
		);
	}

	{
		VkImageMemoryBarrier2 const barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkImg,
			.subresourceRange = fullImage,
		};
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1u,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	VkAssert(vkEndCommandBuffer(cmd));

	VkFence fence;
	{
		VkFenceCreateInfo const fci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
		};
		VkAssert(vkCreateFence(sDevice->device, &fci, nullptr, &fence));
	}

	{
		VkCommandBufferSubmitInfo const cmdSubmit = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.pNext = nullptr,
			.commandBuffer = cmd,
			.deviceMask = 0,
		};
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount = 0,
			.pWaitSemaphoreInfos = nullptr,
			.commandBufferInfoCount = 1u,
			.pCommandBufferInfos = &cmdSubmit,
			.signalSemaphoreInfoCount = 0,
			.pSignalSemaphoreInfos = nullptr,
		};
		VkAssert(vkQueueSubmit2(sDevice->queueGraphics, 1u, &submitInfo, fence));
	}

	VkAssert(vkWaitForFences(sDevice->device, 1u, &fence, VK_TRUE, UINT64_MAX));

	vkDestroyFence(sDevice->device, fence, nullptr);
	vkFreeCommandBuffers(sDevice->device, pool, 1u, &cmd);

	stbi_write_png(
		path, (int)w, (int)h, 4, allocInfo.pMappedData, (int)(w * 4u)
	);

	vmaDestroyBuffer(sDevice->allocator, stagingBuf, stagingAlloc);
}
