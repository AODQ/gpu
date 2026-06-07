#include "cull.hpp"

#include <vector>

#include <vk_mem_alloc.h>

// -----------------------------------------------------------------------------

#include <filesystem>


std::filesystem::path binary_dir() {
	return std::filesystem::canonical("/proc/self/exe").parent_path();
}

std::filesystem::path pwd_dir() {
	return std::filesystem::current_path();
}

f32 floatUniform() {
	return (f32)rand() / (f32)RAND_MAX;
}

#define fract(x) (f32)((x) - (f32)floorf(x))

// -----------------------------------------------------------------------------

namespace {

gfx::Pipeline sPipeline;
gfx::PipelineCompute sPipelineCompute;
gfx::PipelineCompute sPipelineComputeMip;

VkBuffer sSharedBufferVertex;
VkBuffer sSharedBufferIndex;
VkDeviceAddress sSharedBufferVertexVa;

u32 sDimensionWidth;
u32 sDimensionHeight;
u32 sDimensionMipLevels;

VkBuffer sSharedBufferModel;
VkDeviceAddress sSharedBufferModelVa;

struct GpuModel {
	f32v4 boundingSphere;
	u32 indexOffset;
	u32 indexCount;
	i32 vertexOffset;
};

struct GpuPushConstant {
	f32m44 viewProj;
	uint64_t instanceCount;
	VkDeviceAddress vertexBufferVa;
	VkDeviceAddress instanceBufferVa;
	VkDeviceAddress modelBufferVa;
	VkDeviceAddress indirectDrawBufferVa;
	VkDeviceAddress indirectDrawCounterBufferVa;
};

struct GpuInstance {
	f32m44 transform;
	f32v4 color;
	u32 modelIndex;
};

VkBuffer sSharedBufferInstance;
VkDeviceAddress sSharedBufferInstanceVa;

struct Instance {
	u32 modelIndex;
};

std::vector<Instance> sInstances;

struct CommandDrawIndexed {
	u32 indexCount;
	u32 instanceCount;
	u32 firstIndex;
	i32 vertexOffset;
	u32 firstInstance;
};

struct VertexAttribute {
	f32v3 position;
	f32v3 normal;
};

struct Model {
	u32 modelIndex;
};

std::vector<Model> sModels;

VkBuffer sIndirectDrawBuffer;
VkDeviceAddress sIndirectDrawBufferVa;
VkBuffer sIndirectDrawCounterBuffer;
VkDeviceAddress sIndirectDrawCounterBufferVa;

VkBuffer sIndirectDrawCounterBufferReadback;
VmaAllocation sIndirectDrawCounterBufferReadbackAllocation;
void * sIndirectDrawCounterBufferReadbackMapped;

VkImage sDepthPyramidImage;
VkImageView sDepthPyramidImageView;
VmaAllocation sDepthPyramidImageAllocation;

PFN_vkCmdPushDescriptorSet2 fnVkCmdPushDescriptorSet2KHR;
}

static void modelGenerateSphere(
	u32 const stacks,
	u32 const slices,
	std::vector<VertexAttribute> & outVertices,
	std::vector<u32> & outIndices
) {
	for (u32 i = 0; i <= stacks; ++i) {
		float const phi = M_PI * float(i) / float(stacks);
		float const cp = cosf(phi), sp = sinf(phi);
		for (u32 j = 0; j <= slices; ++j) {
			float const theta = 2.0f * M_PI * float(j) / float(slices);
			auto const p = f32v3{ sp * cosf(theta), cp, sp * sinf(theta) };
			outVertices.emplace_back(p, p);
		}
	}

	u32 const row = slices + 1;
	for (u32 i = 0; i < stacks; ++i) {
		for (u32 j = 0; j < slices; ++j) {
			u32 const a = i * row + j;
			u32 const b = a + row;
			outIndices.insert(outIndices.end(), { a, b, a + 1,  a + 1, b, b + 1 });
		}
	}
}

static void modelGenerateBox(
	std::vector<VertexAttribute> & outVertices,
	std::vector<u32> & outIndices
) {
	struct Face { f32v3 normal; f32v3 u; f32v3 v; };
	Face const faces[6] = {
		{{  0,  0,  1}, {  1, 0,  0}, { 0, 1,  0}},
		{{  0,  0, -1}, { -1, 0,  0}, { 0, 1,  0}},
		{{  1,  0,  0}, {  0, 0, -1}, { 0, 1,  0}},
		{{ -1,  0,  0}, {  0, 0,  1}, { 0, 1,  0}},
		{{  0,  1,  0}, {  1, 0,  0}, { 0, 0, -1}},
		{{  0, -1,  0}, {  1, 0,  0}, { 0, 0,  1}},
	};
	for (auto const & f : faces) {
		u32 const base = (u32)outVertices.size();
		for (i32 sy = -1; sy <= 1; sy += 2)
		for (i32 sx = -1; sx <= 1; sx += 2) {
			f32v3 const p = f.normal + f.u * (f32)sx + f.v * (f32)sy;
			outVertices.emplace_back(p, f.normal);
		};
		outIndices.insert(
			outIndices.end(),
			{ base, base + 1, base + 2, base + 2, base + 1, base + 3 }
		);
	}
}

static void loadPipelineStandard(
	gfx::Device const & device
) {
	VkPushConstantRange const pushConstantRange {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(GpuPushConstant),
	};
	VkPipelineLayoutCreateInfo const pipelineLayoutCi {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};

	VkPipelineLayout pipelineLayout;
	VkAssert(
		vkCreatePipelineLayout(
			device.vkDevice(),
			&pipelineLayoutCi,
			nullptr,
			&pipelineLayout
		)
	);

	sPipeline = (
		gfx::pipeline_create(
			device,
			pipelineLayout,
			pwd_dir() / "shaders/standard.vert",
			pwd_dir() / "shaders/standard.frag",
			/*alphaBlending=*/ false,
			/*isMeshPipeline=*/ false,
			/*hotReloadPipeline=*/ nullptr
		)
	);
}

static void loadPipelineCompute(
	gfx::Device const & device
) {
	VkPushConstantRange const pushConstantRange {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(GpuPushConstant),
	};
	VkPipelineLayoutCreateInfo const pipelineLayoutCi {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};

	VkPipelineLayout pipelineLayout;
	VkAssert(
		vkCreatePipelineLayout(
			device.vkDevice(),
			&pipelineLayoutCi,
			nullptr,
			&pipelineLayout
		)
	);

	sPipelineCompute = (
		gfx::pipeline_compute_create(
			device,
			pipelineLayout,
			pwd_dir() / "shaders/visibility.comp",
			/*hotReloadPipeline=*/ nullptr
		)
	);

	{
		VkDescriptorSetLayoutBinding const binding[2] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = nullptr,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = nullptr,
			}
		};
		VkDescriptorSetLayoutCreateInfo const descriptorSetLayoutCi {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
			.bindingCount = 2,
			.pBindings = binding,
		};
		VkDescriptorSetLayout descriptorSetLayout;
		VkAssert(
			vkCreateDescriptorSetLayout(
				device.vkDevice(),
				&descriptorSetLayoutCi,
				nullptr,
				&descriptorSetLayout
			)
		);
		VkPipelineLayoutCreateInfo const pipelineLayoutCi {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptorSetLayout,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr
		};
		VkPipelineLayout mipPipelineLayout;
		VkAssert(
			vkCreatePipelineLayout(
				device.vkDevice(),
				&pipelineLayoutCi,
				nullptr,
				&mipPipelineLayout
			)
		);
		sPipelineComputeMip = (
			gfx::pipeline_compute_create(
				device,
				mipPipelineLayout,
				pwd_dir() / "shaders/hiz_mip.comp",
				/*hotReloadPipeline=*/ nullptr
			)
		);
	}
}

VkBuffer uploadBuffer(
	gfx::Device const & device,
	VkCommandPool const & commandPool,
	void const * const data,
	VkDeviceSize const dataByteCount,
	VkBufferUsageFlags const usage,
	VmaAllocation & outAllocation
) {
	// -- staging buffer (host visible)
	VkBuffer staging;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo stagingInfo;
	if (data) {
		VkBufferCreateInfo const stagingCi {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = dataByteCount,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
		};
		VmaAllocationCreateInfo const stagingAllocCi {
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
		VkAssert(
			vmaCreateBuffer(
				/*allocator=*/ device.allocator,
				/*pBufferCreateInfo=*/ &stagingCi,
				/*pAllocationCreateInfo=*/ &stagingAllocCi,
				/*pBuffer=*/ &staging,
				/*pAllocation=*/ &stagingAlloc,
				/*pAllocationInfo=*/ &stagingInfo
			)
		);
		memcpy(stagingInfo.pMappedData, data, dataByteCount);
	}

	// -- device-local
	VkBufferCreateInfo dstCi {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = dataByteCount,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo dstAllocCi {
		.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer dst;
	VkAssert(
		vmaCreateBuffer(
			/*allocator=*/ device.allocator,
			/*pBufferCreateInfo=*/ &dstCi,
			/*pAllocationCreateInfo=*/ &dstAllocCi,
			/*pBuffer=*/ &dst,
			/*pAllocation=*/ &outAllocation,
			/*pAllocationInfo=*/ nullptr
		)
	);

	// one-shot copy
	if (!data) {
		return dst;
	}

	VkCommandBufferAllocateInfo const cmdAllocCi {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(device.vkDevice(), &cmdAllocCi, &cmd));
	VkCommandBufferBeginInfo const cmdBeginCi {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};

	vkBeginCommandBuffer(cmd, &cmdBeginCi);
	VkBufferCopy const copyRegion {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = dataByteCount,
	};
	vkCmdCopyBuffer(cmd, staging, dst, 1, &copyRegion);
	vkEndCommandBuffer(cmd);

	VkSubmitInfo const submitCi {
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
	vkQueueSubmit(device.graphicsQueue, 1, &submitCi, VK_NULL_HANDLE);
	vkQueueWaitIdle(device.graphicsQueue);
	vkFreeCommandBuffers(device.vkDevice(), commandPool, 1, &cmd);
	vmaDestroyBuffer(device.allocator, staging, stagingAlloc);
	return dst;
}

// -----------------------------------------------------------------------------

void Cull::init(
	gfx::Device const & device,
	VkCommandPool const & commandPool
) {
	loadPipelineStandard(device);
	loadPipelineCompute(device);

	sDimensionWidth = device.swapchain.extent.width;
	sDimensionHeight = device.swapchain.extent.height;
	sDimensionMipLevels = (
		  1
		+ (u32)std::floor(std::log2(std::max(sDimensionWidth, sDimensionHeight)))
	);

	std::vector<VertexAttribute> allVertices;
	std::vector<u32> allIndices;
	std::vector<GpuModel> allModels;
	std::vector<GpuInstance> allInstances;

	auto const loadModel = [&](
		std::vector<VertexAttribute> const & vertices,
		std::vector<u32> const & indices
	) -> void {
		u32 const firstIndex = (u32)allIndices.size();
		u32 const vertexOffset = (u32)allVertices.size();

		// compute bounding sphere for the model
		f32v4 const boundingSphere = [vertices]() -> f32v4 {
			f32v3 low { vertices[0].position };
			f32v3 high { vertices[0].position };
			for (auto const & v : vertices) {
				low.x = std::min(low.x, v.position.x);
				low.y = std::min(low.y, v.position.y);
				low.z = std::min(low.z, v.position.z);
				high.x = std::max(high.x, v.position.x);
				high.y = std::max(high.y, v.position.y);
				high.z = std::max(high.z, v.position.z);
			}
			f32v3 const center = (low + high) * 0.5f;
			f32 const radius = length(high - center);
			return { center.x, center.y, center.z, radius };
		}();

		// record the model
		allVertices.insert(allVertices.end(), vertices.begin(), vertices.end());
		allIndices.insert(allIndices.end(), indices.begin(), indices.end());
		allModels.push_back(GpuModel {
			.boundingSphere = boundingSphere,
			.indexOffset = firstIndex,
			.indexCount = (u32)indices.size(),
			.vertexOffset = (i32)vertexOffset,
		});

		sModels.emplace_back(Model {
			.modelIndex = (u32)allModels.size() - 1,
		});
	};

	// load up sphere
	{
		std::vector<VertexAttribute> vertices;
		std::vector<u32> indices;
		modelGenerateSphere(16, 16, vertices, indices);
		loadModel(vertices, indices);
	}

	// load up box
	{
		std::vector<VertexAttribute> vertices;
		std::vector<u32> indices;
		modelGenerateBox(vertices, indices);
		loadModel(vertices, indices);
	}

	// -- allocate model buffer
	{
		VmaAllocation alloc;
		sSharedBufferModel = uploadBuffer(
			device,
			commandPool,
			&allModels[0],
			sizeof(GpuModel) * allModels.size(),
			(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			),
			alloc
		);
		VkBufferDeviceAddressInfo const modelBufferAddressInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.pNext = nullptr,
			.buffer = sSharedBufferModel,
		};
		sSharedBufferModelVa = (
			vkGetBufferDeviceAddress(
				device.vkDevice(),
				&modelBufferAddressInfo
			)
		);
	}

	// -- allocate index/vertex buffers
	{
		VmaAllocation vertexAlloc;
		VmaAllocation indexAlloc;
		sSharedBufferVertex = uploadBuffer(
			device,
			commandPool,
			&allVertices[0],
			sizeof(VertexAttribute) * allVertices.size(),
			(
				  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			),
			vertexAlloc
		);
		sSharedBufferIndex = uploadBuffer(
			device,
			commandPool,
			&allIndices[0],
			sizeof(u32) * allIndices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			indexAlloc
		);

		VkBufferDeviceAddressInfo const vertexBufferAddressInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.pNext = nullptr,
			.buffer = sSharedBufferVertex,
		};
		sSharedBufferVertexVa = (
			vkGetBufferDeviceAddress(
				device.vkDevice(),
				&vertexBufferAddressInfo
			)
		);
	}

	// create instance data for all the models
	for (size_t i = 0; i < sModels.size()*4; ++i) {
		auto const transform = f32m44 {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			(f32)(i%4) * 4.0f - 2.0f,
			(f32)(i/4) * 1.5f,
			0.0f, 1.0f
		};
		GpuInstance instanceMem;
		instanceMem.transform = transform;
		instanceMem.modelIndex = (u32)i/4;
		allInstances.emplace_back(GpuInstance {
			.transform = transform,
			.color = f32v4 {
				floatUniform(),
				floatUniform(),
				floatUniform(),
				1.0f
			},
			.modelIndex = (u32)i/4,
		});
		sInstances.emplace_back(Instance {
			.modelIndex = (u32)i/4,
		});
	}

	// create instance buffer
	{
		VmaAllocation alloc;
		sSharedBufferInstance = uploadBuffer(
			device,
			commandPool,
			&allInstances[0],
			sizeof(GpuInstance) * allInstances.size(),
			(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			),
			alloc
		);
		VkBufferDeviceAddressInfo const instanceBufferAddressInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.pNext = nullptr,
			.buffer = sSharedBufferInstance,
		};
		sSharedBufferInstanceVa = (
			vkGetBufferDeviceAddress(
				device.vkDevice(),
				&instanceBufferAddressInfo
			)
		);
	}

	// load up indirect draw buffer
	{
		VmaAllocation alloc;
		sIndirectDrawBuffer = ::uploadBuffer(
			device,
			commandPool,
			nullptr,
			sizeof(CommandDrawIndexed) * sInstances.size(),
			(
				  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
				| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			),
			alloc
		);
		VkBufferDeviceAddressInfo const indirectDrawBufferAddressInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.pNext = nullptr,
			.buffer = sIndirectDrawBuffer,
		};
		sIndirectDrawBufferVa = (
			vkGetBufferDeviceAddress(
				device.vkDevice(),
				&indirectDrawBufferAddressInfo
			)
		);
	}

	// load up indirect draw counter buffer
	{
		u32 counterInitialValue = (u32)sInstances.size();
		VmaAllocation alloc;
		sIndirectDrawCounterBuffer = ::uploadBuffer(
			device,
			commandPool,
			&counterInitialValue,
			sizeof(u32),
			(
				  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
				| VK_BUFFER_USAGE_TRANSFER_DST_BIT
				| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			),
			alloc
		);
		VkBufferDeviceAddressInfo const indirectDrawCounterBufferAddressInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.pNext = nullptr,
			.buffer = sIndirectDrawCounterBuffer,
		};
		sIndirectDrawCounterBufferVa = (
			vkGetBufferDeviceAddress(
				device.vkDevice(),
				&indirectDrawCounterBufferAddressInfo
			)
		);
	}

	// allocate readback buffer
	{
		VkBufferCreateInfo const readbackBufferCi {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = sizeof(u32),
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
		};
		VmaAllocationCreateInfo const readbackBufferAllocCi {
			.flags = (
				  VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
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
		VmaAllocationInfo info;
		VkAssert(
			vmaCreateBuffer(
				/*allocator=*/ device.allocator,
				/*pBufferCreateInfo=*/ &readbackBufferCi,
				/*pAllocationCreateInfo=*/ &readbackBufferAllocCi,
				/*pBuffer=*/ &sIndirectDrawCounterBufferReadback,
				/*pAllocation=*/ &sIndirectDrawCounterBufferReadbackAllocation,
				/*pAllocationInfo=*/ &info
			)
		);
		sIndirectDrawCounterBufferReadbackMapped = info.pMappedData;
	}

	// -- load up depth pyramid image
	{
		VkImageCreateInfo const imageCi {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_R32_SFLOAT,
			.extent = VkExtent3D {
				.width = device.swapchain.extent.width,
				.height = device.swapchain.extent.height,
				.depth = 1,
			},
			.mipLevels = sDimensionMipLevels,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = (
				  VK_IMAGE_USAGE_SAMPLED_BIT
				| VK_IMAGE_USAGE_STORAGE_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
				| VK_IMAGE_USAGE_TRANSFER_DST_BIT
			),
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		VmaAllocationCreateInfo const imageAllocCi {
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
				/*allocator=*/ device.allocator,
				/*pCreateInfo=*/ &imageCi,
				/*pAllocationCreateInfo=*/ &imageAllocCi,
				/*pImage=*/ &sDepthPyramidImage,
				/*pAllocation=*/ &sDepthPyramidImageAllocation,
				/*pAllocationInfo=*/ nullptr
			)
		);

		sDepthPyramidImageView = [&]() -> VkImageView {
			VkImageViewCreateInfo const viewCi {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = sDepthPyramidImage,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_R32_SFLOAT,
				.components = {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = imageCi.mipLevels,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VkImageView view;
			VkAssert(
				vkCreateImageView(
					device.vkDevice(),
					&viewCi,
					nullptr,
					&view
				)
			);
			return view;
		}();
	}

	fnVkCmdPushDescriptorSet2KHR = (
		(PFN_vkCmdPushDescriptorSet2)
		vkGetDeviceProcAddr(
			device.device,
			"vkCmdPushDescriptorSet2KHR"
		)
	);
}

void Cull::shutdown(gfx::Device const & device) {
	vkDestroyPipeline(device.vkDevice(), sPipeline.pipeline, nullptr);
	vkDestroyPipelineLayout(device.vkDevice(), sPipeline.layout, nullptr);
	vkDestroyPipeline(device.vkDevice(), sPipelineCompute.pipeline, nullptr);
	vkDestroyPipelineLayout(device.vkDevice(), sPipelineCompute.layout, nullptr);
}

static void bindPushConstants(
	VkCommandBuffer commandBuffer,
	VkPipelineLayout layout,
	VkShaderStageFlagBits stages,
	f32m44 const & viewProj
) {
	// { mat4 transform, uint modelIndex, vec4 boundingSphere, }
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ 0, // f32m44 alignment
		/*size=*/ sizeof(viewProj),
		&viewProj
	);
	uint64_t const instanceCount = (uint64_t)sInstances.size();
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, instanceCount),
		/*size=*/ sizeof(uint64_t),
		&instanceCount
	);
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, vertexBufferVa),
		/*size=*/ sizeof(VkDeviceAddress),
		&sSharedBufferVertexVa
	);
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, instanceBufferVa),
		/*size=*/ sizeof(VkDeviceAddress),
		&sSharedBufferInstanceVa
	);
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, modelBufferVa),
		/*size=*/ sizeof(VkDeviceAddress),
		&sSharedBufferModelVa
	);
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, indirectDrawBufferVa),
		/*size=*/ sizeof(VkDeviceAddress),
		&sIndirectDrawBufferVa
	);
	vkCmdPushConstants(
		commandBuffer,
		layout,
		stages,
		/*offset=*/ offsetof(GpuPushConstant, indirectDrawCounterBufferVa),
		/*size=*/ sizeof(VkDeviceAddress),
		&sIndirectDrawCounterBufferVa
	);
}

void Cull::update(
	gfx::Device const & device,
	VkCommandBuffer commandBuffer,
	f32m44 const & viewProj
) {
	vkCmdFillBuffer(
		commandBuffer,
		sIndirectDrawCounterBuffer,
		/*dstOffset=*/ 0,
		/*size=*/ sizeof(u32),
		/*data=*/ 0
	);
	{ // barrier transfer -> compute r/w
		VkMemoryBarrier2 const barrier {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = (
				VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
			),
		};
		VkDependencyInfo const depInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &barrier,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 0,
			.pImageMemoryBarriers = nullptr,
		};
		vkCmdPipelineBarrier2(commandBuffer, &depInfo);
	}
	vkCmdBindPipeline(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		sPipelineCompute.pipeline
	);

	bindPushConstants(
		commandBuffer,
		sPipelineCompute.layout,
		VK_SHADER_STAGE_COMPUTE_BIT, viewProj
	);
	vkCmdDispatch(
		commandBuffer,
		(sInstances.size()+127)/128,
		1,
		1
	);

	VkMemoryBarrier2 const barrier {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
		.dstStageMask = (
			  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
			| VK_PIPELINE_STAGE_2_TRANSFER_BIT
		),
		.dstAccessMask = (
			  VK_ACCESS_2_SHADER_READ_BIT
			| VK_ACCESS_2_TRANSFER_READ_BIT
		),
	};
	VkDependencyInfo const depInfo {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};
	vkCmdPipelineBarrier2(commandBuffer, &depInfo);

	// copy to readback
	{
		
		VkBufferCopy const copyRegion {
			.srcOffset = 0,
			.dstOffset = 0,
			.size = sizeof(u32),
		};
		vkCmdCopyBuffer(
			commandBuffer,
			/*srcBuffer=*/ sIndirectDrawCounterBuffer,
			/*dstBuffer=*/ sIndirectDrawCounterBufferReadback,
			1, &copyRegion
		);
	}
}


void Cull::draw(
	gfx::Device const & device,
	VkCommandBuffer commandBuffer,
	f32m44 const & viewProj
) {
	vkCmdBindPipeline(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		sPipeline.pipeline
	);

	vkCmdBindIndexBuffer(
		commandBuffer,
		sSharedBufferIndex,
		0,
		VK_INDEX_TYPE_UINT32
	);

	bindPushConstants(
		commandBuffer,
		sPipeline.layout,
		VK_SHADER_STAGE_VERTEX_BIT,
		viewProj
	);
	vkCmdDrawIndexedIndirectCount(
		commandBuffer,
		/*buffer=*/ sIndirectDrawBuffer,
		/*offset=*/ 0,
		/*countBuffer=*/ sIndirectDrawCounterBuffer,
		/*countBufferOffset=*/ 0,
		/*maxDrawCount=*/ (u32)sInstances.size(),
		/*stride=*/ sizeof(CommandDrawIndexed)
	);
}

void Cull::resolveDepth(
	gfx::Device const & device,
	VkCommandBuffer commandBuffer,
	VkImage const depthImage,
	VkImageView const depthImageView
) {
	// transition depth image to shader read only
	{
		VkImageMemoryBarrier2 const barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = depthImage,
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkDependencyInfo const depInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(commandBuffer, &depInfo);
	}

	{ // transition hiZ pyramid
		VkImageMemoryBarrier2 const barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = 0,
			.srcAccessMask = 0,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sDepthPyramidImage,
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = VK_REMAINING_MIP_LEVELS,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkDependencyInfo const depInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(commandBuffer, &depInfo);
	}

	// bind hiZ pipeline
	vkCmdBindPipeline(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		sPipelineComputeMip.pipeline
	);

	// bind descriptor push set for sampler+texture

	auto const descriptorImageInfo = VkDescriptorImageInfo{
		.sampler = device.samplerNearest,
		.imageView = depthImageView,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	auto const descriptorWrite = VkWriteDescriptorSet {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = VK_NULL_HANDLE,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &descriptorImageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};

	auto const descriptorImageInfoMip = VkDescriptorImageInfo{
		.sampler = VK_NULL_HANDLE,
		.imageView = sDepthPyramidImageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	auto const descriptorWriteMip = VkWriteDescriptorSet {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = VK_NULL_HANDLE,
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &descriptorImageInfoMip,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};

	std::array<VkWriteDescriptorSet, 2> writes = {{
		descriptorWrite, descriptorWriteMip
	}};

	VkPushDescriptorSetInfo const pushDescriptorSetInfo {
		.sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO_KHR,
		.pNext = nullptr,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.layout = sPipelineComputeMip.layout,
		.set = 0,
		.descriptorWriteCount = (uint32_t)writes.size(),
		.pDescriptorWrites = writes.data(),
	};

	fnVkCmdPushDescriptorSet2KHR(commandBuffer, &pushDescriptorSetInfo);

	// emit dispatches
	vkCmdDispatch(
		commandBuffer,
		(sDimensionWidth + 15) / 16,
		(sDimensionHeight + 15) / 16,
		1
	);

	// transition depth attachment back to depth attachment optimal
	{
		VkImageMemoryBarrier2 const barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
			.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = depthImage,
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkDependencyInfo const depInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(commandBuffer, &depInfo);
	}
}

u32 Cull::lastVisibleCount() {
	return *(u32*)sIndirectDrawCounterBufferReadbackMapped;
}

u32 Cull::totalInstanceCount() {
	return (u32)sInstances.size();
}

VkImageView Cull::imageHiz() {
	return sDepthPyramidImageView;
}

void Cull::imageHizTransition(VkCommandBuffer const & commandBuffer) {
	VkImageMemoryBarrier2 const barrier {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
		.dstStageMask = 0,
		.dstAccessMask = 0,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = sDepthPyramidImage,
		.subresourceRange = VkImageSubresourceRange {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkDependencyInfo const depInfo {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 0,
		.pMemoryBarriers = nullptr,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &depInfo);
}
