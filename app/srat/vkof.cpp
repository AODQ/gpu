#include <srat/vkof.hpp>
#include <srat/core-handle.hpp>

#include <imgui.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

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
	std::string const & glslPath, ShaderStage stage
);
static bool file_changed(
	std::string const & path,
	std::filesystem::file_time_type & lastWriteTime
);
static void pipeline_hot_reload();

static VkFormat to_vk_format(vkof::ImageFormat format) {
	switch (format) {
		case vkof::ImageFormat::r8g8b8a8_unorm:
			return VK_FORMAT_R8G8B8A8_UNORM;
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
		std::array<ImplTexture, 2> textures;
	};

	struct ImplTransientBuffer {
		VkBuffer buffer;
		VmaAllocation allocation;
		bool isDoubleBuffered;
	};

	struct ImplRenderBarrier
	{
		vkof::TransientImage transientImage;
		VkImageLayout layout;
	};

	struct ImplRenderNode
	{
		vkof::CommandBuffer cmd;
		std::vector<ImplRenderBarrier> barriers;
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

		std::filesystem::file_time_type lastWriteTimeFragment;
		std::filesystem::file_time_type lastWriteTimeMesh;
	};

	struct ImplPipelineCompute
	{
		VkPipeline pipeline;
		VkPipelineLayout layout;
		std::string pathCompute;
		std::filesystem::file_time_type lastWriteTimeCompute;
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

		u32 swapchainImageIndex;

		VkCommandPool commandPoolGraphics;
		VkCommandPool commandPoolCompute;

		VkPipelineLayout pipelineLayoutUniversal;

		VkSurfaceKHR surface;
		VkQueue queueGraphics;
		VkQueue queueCompute;
		VmaAllocator allocator;

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

		// need a linear list of pipeline handles for hot reload iteration
		std::vector<vkof::Pipeline> pipelineGraphicsHandles {};
		std::vector<vkof::Pipeline> pipelineComputeHandles {};

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
					? (VmaAllocationCreateFlags)VMA_ALLOCATION_CREATE_MAPPED_BIT
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
		// note: byte length not tracked here yet; returns base ptr
		return srat::slice<u8>((u8 *)implBuffer->mapped, 0);
	}
	return srat::slice<u8>(nullptr, 0);
}

// -----------------------------------------------------------------------------
// -- image
// -----------------------------------------------------------------------------

vkof::Image vkof::image_create(vkof::ImageCreateInfo const & createInfo)
{
	ImplTexture implTexture;
	VkFormat const vkFormat = (VkFormat)createInfo.format;
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

vkof::Image vkof::image_swapchain()
{
	return vkof::Image { .id = u64(-1), };
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
			.textureCompressionETC2 = true,
			.textureCompressionASTC_LDR = true,
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
			VmaAllocation allocation;
			VkImage image;
			VkAssert(
				vmaCreateImage(
					allocator,
					&imageCreateInfo,
					&allocationCreateInfo,
					&image,
					&allocation,
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
		.stageFlags = (
			  VK_SHADER_STAGE_VERTEX_BIT
			| VK_SHADER_STAGE_MESH_BIT_EXT
			| VK_SHADER_STAGE_FRAGMENT_BIT
		),
		.offset = 0,
		.size = 128, // application-defined, max is device limit
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
		.swapchainImageIndex = 0,
		.commandPoolGraphics = commandPoolGraphics,
		.commandPoolCompute = commandPoolCompute,
		.pipelineLayoutUniversal = pipelineLayoutUniversal,
		.surface = surface,
		.queueGraphics = graphicsQueue,
		.queueCompute = computeQueue,
		.allocator = allocator,
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

	// TODO not yet created in init (each needs a design decision):
	//   - frames-in-flight: command pools, acquire/render semaphores,
	//     fences, frame index
	//   - universal pipeline layout (256B push + bindless set)
	//   - bindless backend (NV resident handles or descriptor array)
	//   - pipeline pool + pipeline create impls
	//   - per-frame node scratch + deferred-destroy trash lists
}

void vkof::shutdown()
{
	// wait for the GPU to finish before destroying anything
	vkDeviceWaitIdle(sDevice->device);

	for (VkImageView view : sDevice->swapchainDepthImageViews) {
		vkDestroyImageView(sDevice->device, view, nullptr);
	}
	for (VkImage image : sDevice->swapchainDepthImages) {
		vmaDestroyImage(sDevice->allocator, image, VK_NULL_HANDLE);
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
	sDevice->swapchain.destroy_image_views(sDevice->swapchainImageViews);
	vkDestroyPipelineLayout(
		sDevice->device, sDevice->pipelineLayoutUniversal, nullptr
	);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolGraphics, nullptr);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolCompute, nullptr);

	vkb::destroy_swapchain(sDevice->swapchain);
	vkb::destroy_surface(sDevice->instance, sDevice->surface);
	vkb::destroy_device(sDevice->device);
	vkb::destroy_instance(sDevice->instance);
	glfwDestroyWindow(sDevice->window);
	glfwTerminate();

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
		compile_and_load_shader_module(impl.pathCompute, ShaderStage::compute)
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
		compile_and_load_shader_module(impl.pathMesh, ShaderStage::mesh)
	);
	VkShaderModule const fragmentModule = (
		compile_and_load_shader_module(impl.pathFragment, ShaderStage::fragment)
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
	std::string const & glslPath, ShaderStage stage
) {
	std::string const spvPath = glslPath + ".spv";

	std::string const cmd = (
		std::string("glslangValidator -S ")
		+ shader_stage_flag(stage)
		+ " --target-env vulkan1.3 -Ishaders/ -o "
		+ spvPath + " " + glslPath
	);
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

		.lastWriteTimeFragment = file_write_time(createInfo.pathFragment),
		.lastWriteTimeMesh = file_write_time(createInfo.pathMesh),
	};
	implPipeline.pipeline = build_graphics_pipeline(implPipeline);
	if (implPipeline.pipeline == VK_NULL_HANDLE) {
		printf(
			"failed to create graphics pipeline '%s/%s'\n",
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

	// vertex-pulling model: mesh shader, not vkCmdDraw. one workgroup per
	// vertexCount-derived dispatch is the caller's job inside the shader;
	// here vertexCount maps to the task/mesh group count.
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
