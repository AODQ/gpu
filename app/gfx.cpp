#include "gfx.hpp"

#include <imgui.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

// -----------------------------------------------------------------------------

gfx::Device gfx::device_init() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwSwapInterval(1);
	GLFWwindow * const window = (
		glfwCreateWindow(1280, 720, "demo", nullptr, nullptr)
	);

	// -- instance
	auto const instance = [&]() -> vkb::Instance {
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
	auto const device = [&]() -> vkb::Device {
		VkPhysicalDeviceFeatures features {
			.robustBufferAccess = true,
			.fullDrawIndexUint32 = true,
			.imageCubeArray = false,
			.independentBlend = false,
			.geometryShader = false,
			.tessellationShader = false,
			.sampleRateShading = false,
			.dualSrcBlend = false,
			.logicOp = true,
			.multiDrawIndirect = true,
			.drawIndirectFirstInstance = true,
			.depthClamp = false,
			.depthBiasClamp = false,
			.fillModeNonSolid = false,
			.depthBounds = false,
			.wideLines = false,
			.largePoints = false,
			.alphaToOne = false,
			.multiViewport = false,
			.samplerAnisotropy = false,
			.textureCompressionETC2 = false,
			.textureCompressionASTC_LDR = false,
			.textureCompressionBC = false,
			.occlusionQueryPrecise = false,
			.pipelineStatisticsQuery = false,
			.vertexPipelineStoresAndAtomics = false,
			.fragmentStoresAndAtomics = false,
			.shaderTessellationAndGeometryPointSize = false,
			.shaderImageGatherExtended = false,
			.shaderStorageImageExtendedFormats = false,
			.shaderStorageImageMultisample = false,
			.shaderStorageImageReadWithoutFormat = false,
			.shaderStorageImageWriteWithoutFormat = false,
			.shaderUniformBufferArrayDynamicIndexing = false,
			.shaderSampledImageArrayDynamicIndexing = false,
			.shaderStorageBufferArrayDynamicIndexing = false,
			.shaderStorageImageArrayDynamicIndexing = false,
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
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
			.pNext = nullptr,
			.taskShader = VK_TRUE,
			.meshShader = VK_TRUE,
			.multiviewMeshShader = VK_FALSE,
			.primitiveFragmentShadingRateMeshShader = VK_FALSE,
			.meshShaderQueries = VK_FALSE,
		};

		VkPhysicalDeviceMaintenance4Features const maintenance4Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
			.pNext = nullptr,
			.maintenance4 = VK_TRUE,
		};

		VkPhysicalDeviceDynamicRenderingFeatures const dynamicRenderingFeatures ={
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
			.pNext = nullptr,
			.dynamicRendering = VK_TRUE,
		};

		VkPhysicalDeviceSynchronization2Features const synchronization2Features ={
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
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
			.add_required_extension(VK_KHR_MAINTENANCE_6_EXTENSION_NAME )
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
			vkb::DeviceBuilder(physicalDeviceResults.value())
			.build()
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
		auto const graphicsQueueResults = (
			device.get_queue(vkb::QueueType::graphics)
		);
		if (!graphicsQueueResults) {
			printf(
				"failed to get graphics queue: %s\n",
				graphicsQueueResults.error().message().c_str()
			);
			exit(1);
		}
		return graphicsQueueResults.value();
	}();

	// -- swapchain
	auto const swapchain = [&]() -> vkb::Swapchain {
		auto const swapchainResults = (
			vkb::SwapchainBuilder(device)
			.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
			.set_desired_extent(800, 600)
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
			.flags = (
				VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
			),
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
	auto const depthSwapchainImages = [&]() -> std::vector<VkImage> {
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

	auto const depthSwapchainImageViews = [&]() -> std::vector<VkImageView> {
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

	VkSampler samplerNearest;
	{
		VkSamplerCreateInfo const samplerCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
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
				device.device,
				&samplerCreateInfo,
				nullptr,
				&samplerNearest
			)
		);
	}

	return Device {
		.window = window,
		.instance = instance,
		.physicalDevice = device.physical_device,
		.swapchain = swapchain,
		.swapchainDepthImages = depthSwapchainImages,
		.swapchainDepthImageViews = depthSwapchainImageViews,
		.surface = surface,
		.device = device,
		.allocator = allocator,
		.samplerNearest = samplerNearest,
		.graphicsQueue = graphicsQueue,
		.graphicsQueueFamily = (
			device.get_queue_index(vkb::QueueType::graphics).value()
		),
	};
}

// -----------------------------------------------------------------------------

VkShaderModule gfx::shader_load(
	gfx::Device const & device,
	std::filesystem::path const & path
) {
	auto code = [&]() -> std::vector<u8> {
		FILE * f = fopen(path.c_str(), "rb");
		if (!f) {
			printf("failed to open shader file: %s\n", path.c_str());
			exit(1);
		 }
		fseek(f, 0, SEEK_END);
		size_t size = ftell(f);
		fseek(f, 0, SEEK_SET);
		std::vector<u8> code(size);
		fread(code.data(), 1, size, f);
		fclose(f);
		return code;
	}();

	VkShaderModuleCreateInfo ci {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = code.size(),
		.pCode = (const uint32_t*)code.data(),
	};
	VkShaderModule mod;
	VkAssert(vkCreateShaderModule(device.device, &ci, nullptr, &mod));
	return mod;
}

// -----------------------------------------------------------------------------

struct CompileResult {
	i32 exitCode;
	std::string output;
};

CompileResult system_capture(std::string const & cmd) {
	std::string const fullCmd = cmd + " 2>&1";

	FILE * const pipe = popen(fullCmd.c_str(), "r");
	if (!pipe) {
		printf("failed to run command: %s\n", cmd.c_str());
		exit(1);
	}

	std::string output;
	static std::array<char, 4096> buffer;
	while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
		output += buffer.data();
	}
	printf("command output: %s\n", output.c_str());

	i32 status = pclose(pipe);
	i32 exitCode = WEXITSTATUS(status);
	printf("status: %d exit code: %d\n", status, exitCode);
	if (exitCode != 0) {
		exit(0);
	}
	return { exitCode, output };
}

// -----------------------------------------------------------------------------

static std::string lastCompileOutputError = "";

std::string const & gfx::last_pipline_compile_error() {
	return lastCompileOutputError;
}

// -----------------------------------------------------------------------------

gfx::Pipeline gfx::pipeline_create(
	gfx::Device const & device,
	VkPipelineLayout const pipelineLayout,
	std::filesystem::path const & pathMesh,
	std::filesystem::path const & pathFrag,
	bool const alphaBlending,
	bool const isMeshPipeline,
	Pipeline const * hotReloadPipeline
) {
	auto meshSpvPath = pathMesh.string() + ".spv";
	auto fragSpvPath = pathFrag.string() + ".spv";

	// check if shader files exist, if not then can't compile etc
	if (!std::filesystem::exists(pathMesh)) {
		printf("mesh shader file does not exist: %s\n", pathMesh.c_str());
		exit(1);
	}
	if (!std::filesystem::exists(pathFrag)) {
		printf("fragment shader file does not exist: %s\n", pathFrag.c_str());
		exit(1);
	}

	// Compile if needed
	CompileResult const compileMeshResult = system_capture(
		(
			(
				isMeshPipeline
				? ("glslangValidator -S mesh --target-env vulkan1.3 -o ")
				: ("glslangValidator -S vert --target-env vulkan1.3 -o ")
			)
			+ meshSpvPath + " " + pathMesh.string()
		).c_str()
	);
	CompileResult const compileFragResult = system_capture(
		("glslangValidator -S frag --target-env vulkan1.3 -o "
		+ fragSpvPath + " " + pathFrag.string()).c_str()
	);
	if (compileMeshResult.exitCode != 0) {
		printf("failed to compile mesh shader: %s\n", pathMesh.c_str());
		lastCompileOutputError = compileMeshResult.output;
	}
	else if (compileFragResult.exitCode != 0) {
		printf("failed to compile fragment shader: %s\n", pathFrag.c_str());
		lastCompileOutputError = compileFragResult.output;
	}
	else {
		lastCompileOutputError.clear();
	}

	// Load shader modules
	auto loadShader = [&](std::string const & spvPath) -> VkShaderModule {
		FILE * f = fopen(spvPath.c_str(), "rb");
		if (!f) {
			printf("failed to open shader: %s\n", spvPath.c_str());
			exit(1);
		}
		fseek(f, 0, SEEK_END);
		size_t size = ftell(f);
		fseek(f, 0, SEEK_SET);
		std::vector<u8> code(size);
		fread(code.data(), 1, size, f);
		fclose(f);

		VkShaderModuleCreateInfo ci {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.codeSize = code.size(),
			.pCode = (const uint32_t*)code.data(),
		};
		VkShaderModule mod;
		VkAssert(vkCreateShaderModule(device.device, &ci, nullptr, &mod));
		return mod;
	};

	VkShaderModule shaderMesh = loadShader(meshSpvPath);
	VkShaderModule shaderFrag = loadShader(fragSpvPath);

	// Destroy old shaders if hot reloading
	if (hotReloadPipeline) {
		vkDestroyShaderModule(device.device, hotReloadPipeline->shaderMesh, nullptr);
		vkDestroyShaderModule(device.device, hotReloadPipeline->shaderFrag, nullptr);
		vkDestroyPipeline(device.device, hotReloadPipeline->pipeline, nullptr);
	}

	VkFormat const colorFormat = device.swapchain.image_format;

	VkPipelineRenderingCreateInfo const pipelineRenderingCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.pNext = nullptr,
		.viewMask = 0,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &colorFormat,
		.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};

	VkPipelineShaderStageCreateInfo const shaderStages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = (
				isMeshPipeline
				? VK_SHADER_STAGE_MESH_BIT_EXT
				: VK_SHADER_STAGE_VERTEX_BIT
			),
			.module = shaderMesh,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shaderFrag,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
	};

	VkPipelineViewportStateCreateInfo const viewportStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = nullptr,
		.scissorCount = 1,
		.pScissors = nullptr,
	};

	VkPipelineRasterizationStateCreateInfo const rasterizationStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo const multisampleStateCreateInfo {
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

	VkPipelineColorBlendAttachmentState const colorBlendAttachmentStateNoAlpha {
		.blendEnable = VK_FALSE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
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
	VkPipelineColorBlendAttachmentState const colorBlendAttachmentStateAlpha {
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
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

	VkPipelineColorBlendStateCreateInfo const colorBlendStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &(alphaBlending
			? colorBlendAttachmentStateAlpha
			: colorBlendAttachmentStateNoAlpha
		),
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
	};

	VkPipelineDepthStencilStateCreateInfo const depthStencilStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {
			.failOp = VK_STENCIL_OP_KEEP,
			.passOp = VK_STENCIL_OP_KEEP,
			.depthFailOp = VK_STENCIL_OP_KEEP,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.compareMask = 0,
			.writeMask = 0,
			.reference = 0,
		},
		.back = {
			.failOp = VK_STENCIL_OP_KEEP,
			.passOp = VK_STENCIL_OP_KEEP,
			.depthFailOp = VK_STENCIL_OP_KEEP,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.compareMask = 0,
			.writeMask = 0,
			.reference = 0,
		},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	};

	VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo const dynamicStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamicStates,
	};

	VkPipelineVertexInputStateCreateInfo const vertexInputStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = nullptr,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = nullptr,
	};

	VkPipelineInputAssemblyStateCreateInfo const inputAssemblyStateCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	VkGraphicsPipelineCreateInfo const pipelineCreateInfo {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &pipelineRenderingCreateInfo,
		.flags = 0,
		.stageCount = 2,
		.pStages = shaderStages,
		.pVertexInputState = (isMeshPipeline ? nullptr : &vertexInputStateCreateInfo),
		.pInputAssemblyState = (isMeshPipeline ? nullptr : &inputAssemblyStateCreateInfo),
		.pTessellationState = nullptr,
		.pViewportState = &viewportStateCreateInfo,
		.pRasterizationState = &rasterizationStateCreateInfo,
		.pMultisampleState = &multisampleStateCreateInfo,
		.pDepthStencilState = &depthStencilStateCreateInfo,
		.pColorBlendState = &colorBlendStateCreateInfo,
		.pDynamicState = &dynamicStateCreateInfo,
		.layout = pipelineLayout,
		.renderPass = VK_NULL_HANDLE,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};

	VkPipeline pipeline;
	VkAssert(
		vkCreateGraphicsPipelines(
			device.device,
			VK_NULL_HANDLE,
			1,
			&pipelineCreateInfo,
			nullptr,
			&pipeline
		)
	);

	return Pipeline {
		.pipeline = pipeline,
		.layout = pipelineLayout,
		.shaderMesh = shaderMesh,
		.shaderFrag = shaderFrag,
		.lastWriteTimeMesh = std::filesystem::last_write_time(pathMesh),
		.lastWriteTimeFrag = std::filesystem::last_write_time(pathFrag),
	};
}

// -----------------------------------------------------------------------------

gfx::PipelineCompute gfx::pipeline_compute_create(
	gfx::Device const & device,
	VkPipelineLayout const pipelineLayout,
	std::filesystem::path const & pathComp,
	PipelineCompute const * hotReloadPipeline
) {
	auto const compSpvPath = pathComp.string() + ".spv";

	CompileResult const compileResult = system_capture(
		("glslangValidator -S comp --target-env vulkan1.3 -Ishaders/ -o "
		+ compSpvPath + " " + pathComp.string()).c_str()
	);
	if (compileResult.exitCode != 0) {
		printf("failed to compile compute shader: %s\n", pathComp.c_str());
		lastCompileOutputError = compileResult.output;
	}
	else {
		lastCompileOutputError.clear();
	}

	auto loadShader = [&](std::string const & spvPath) -> VkShaderModule {
		FILE * f = fopen(spvPath.c_str(), "rb");
		if (!f) { printf("failed to open shader: %s\n", spvPath.c_str()); exit(1); }
		fseek(f, 0, SEEK_END);
		size_t size = ftell(f);
		fseek(f, 0, SEEK_SET);
		std::vector<u8> code(size);
		fread(code.data(), 1, size, f);
		fclose(f);

		VkShaderModuleCreateInfo ci {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.codeSize = code.size(),
			.pCode = (const uint32_t *)code.data(),
		};
		VkShaderModule mod;
		VkAssert(vkCreateShaderModule(device.device, &ci, nullptr, &mod));
		return mod;
	};

	VkShaderModule const shaderComp = loadShader(compSpvPath);

	if (hotReloadPipeline) {
		vkDestroyShaderModule(device.device, hotReloadPipeline->shaderComp, nullptr);
		vkDestroyPipeline(device.device, hotReloadPipeline->pipeline, nullptr);
	}

	VkComputePipelineCreateInfo const pipelineCreateInfo {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shaderComp,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
		.layout = pipelineLayout,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};

	VkPipeline pipeline;
	VkAssert(
		vkCreateComputePipelines(
			device.device,
			VK_NULL_HANDLE,
			1,
			&pipelineCreateInfo,
			nullptr,
			&pipeline
		)
	);

	return PipelineCompute {
		.pipeline = pipeline,
		.layout = pipelineLayout,
		.shaderComp = shaderComp,   // reusing the field as "the module"
		.lastWriteTimeComp = std::filesystem::last_write_time(pathComp),
	};
}

// -----------------------------------------------------------------------------

gfx::CommandPool gfx::command_pool_create(
	gfx::Device const & device
) {
	VkCommandPoolCreateInfo commandPoolCreateInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = device.graphicsQueueFamily,
	};
	VkCommandPool commandPool;
	VkAssert(
		vkCreateCommandPool(
			device.device,
			&commandPoolCreateInfo,
			nullptr,
			&commandPool
		)
	);
	return CommandPool { .pool = commandPool };
}

// -----------------------------------------------------------------------------

gfx::Frame gfx::frame_create(
	gfx::Device const & device,
	CommandPool const & commandPool
) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = commandPool.pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer commandBuffer;
	VkAssert(
		vkAllocateCommandBuffers(
			device.device,
			&commandBufferAllocateInfo,
			&commandBuffer
		)
	);

	VkSemaphoreCreateInfo semaphoreCreateInfo {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
	};

	VkSemaphore semaphoreSwapchainImageAvailable;
	VkAssert(
		vkCreateSemaphore(
			device.device,
			&semaphoreCreateInfo,
			nullptr,
			&semaphoreSwapchainImageAvailable
		)
	);

	VkSemaphore semaphoreRenderFinished;
	VkAssert(
		vkCreateSemaphore(
			device.device,
			&semaphoreCreateInfo,
			nullptr,
			&semaphoreRenderFinished
		)
	);

	VkFenceCreateInfo fenceCreateInfo {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT,
	};
	VkFence fenceCommandBufferInFlight;
	VkAssert(
		vkCreateFence(
			device.device,
			&fenceCreateInfo,
			nullptr,
			&fenceCommandBufferInFlight
		)
	);

	return Frame {
		.commandBuffer = commandBuffer,
		.semaphoreSwapchainImageAvailable = semaphoreSwapchainImageAvailable,
		.semaphoreRenderFinished = semaphoreRenderFinished,
		.fenceCommandBufferInFlight = fenceCommandBufferInFlight,
	};
}

// -----------------------------------------------------------------------------
// -- imgui
// -----------------------------------------------------------------------------

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_glfw.h>

static VkDescriptorPool imguiDescriptorPool;

void gfx::imgui_init(gfx::Device const & device) {
	VkDescriptorPoolSize poolSizes[] = {
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
	};

	VkDescriptorPoolCreateInfo poolInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = 1000,
		.poolSizeCount = (u32)std::size(poolSizes),
		.pPoolSizes = poolSizes,
	};
	VkAssert(vkCreateDescriptorPool(device.device, &poolInfo, nullptr, &imguiDescriptorPool));

	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForVulkan(device.window, true);

	ImGui_ImplVulkan_InitInfo initInfo {
		.Instance = device.instance.instance,
		.PhysicalDevice = device.physicalDevice.physical_device,
		.Device = device.device,
		.QueueFamily = device.graphicsQueueFamily,
		.Queue = device.graphicsQueue,
		.DescriptorPool = imguiDescriptorPool,
		.RenderPass = VK_NULL_HANDLE, // dynamic rendering
		.MinImageCount = 2,
		.ImageCount = device.swapchain.image_count,
		.MSAASamples = VK_SAMPLE_COUNT_1_BIT,
		.PipelineCache = VK_NULL_HANDLE,
		.Subpass = 0,
		.UseDynamicRendering = true,
		.PipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.pNext = nullptr,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &device.swapchain.image_format,
			.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
			.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
		},
		.Allocator = nullptr,
		.CheckVkResultFn = nullptr,
		.MinAllocationSize = 1024 * 1024,
	};
	ImGui_ImplVulkan_Init(&initInfo);
}

void gfx::imgui_shutdown() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	vkDestroyDescriptorPool(nullptr, imguiDescriptorPool, nullptr); // Get device from somewhere
}

void gfx::imgui_new_frame() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void gfx::imgui_render(VkCommandBuffer commandBuffer) {
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}
