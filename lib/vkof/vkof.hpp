#pragma once
#include <cstdint>

#include <srat/core-array.hpp>
#include <srat/core-types.hpp>

#include <functional>

struct GLFWwindow;

// -----------------------------------------------------------------------------
// -- pipeline
// -----------------------------------------------------------------------------

namespace vkof
{

	enum struct ImageFormat {
		none,
		r8g8b8a8_unorm,
		r8g8b8a8_srgb,
		r16g16b16a16_sfloat,
		r32_float,
		d24_unorm_s8_uint,
	};

	struct Pipeline { u64 id; };

	enum struct DepthTest {
		write_on_test_off,
		write_on_test_on,
		write_off_test_on,
		write_off_test_off,
	};

	enum struct CullMode {
		none,
		front,
		back,
	};

	enum struct BlendMode {
		none,
		alpha_blend,
	};

	struct PipelineGraphicsCreateInfo {
		char const * pathMesh;
		char const * pathFragment;
		srat::slice<ImageFormat const> attachmentColorFormats;
		ImageFormat attachmentDepthStencilFormat;
		DepthTest depthTest;
		CullMode cullMode;
		BlendMode blendMode;
		srat::slice<char const * const> defines {};
		srat::slice<char const * const> includePaths {};
	};

	struct PipelineComputeCreateInfo {
		char const * pathCompute;
		srat::slice<char const * const> defines {};
		srat::slice<char const * const> includePaths {};
	};

	Pipeline pipeline_graphics_create(
		PipelineGraphicsCreateInfo const & createInfo
	);

	Pipeline pipeline_compute_create(
		PipelineComputeCreateInfo const & createInfo
	);

	void pipeline_destroy(Pipeline const & pipeline);

}

// -----------------------------------------------------------------------------
// -- command buffer
// -----------------------------------------------------------------------------

namespace vkof
{
	struct CommandBuffer { u64 id; };
}

// -----------------------------------------------------------------------------
// -- buffer
// -----------------------------------------------------------------------------

namespace vkof
{
	struct Buffer { u64 id; };

	enum struct BufferMemory : u32 {
		DeviceOnly,
		HostWritable,
	};

	struct BufferCreateInfo {
		u64 byteCount;
		BufferMemory memory;
	};

	Buffer buffer_create(BufferCreateInfo const & createInfo);
	void buffer_destroy(Buffer const & buffer);

	u64 buffer_virtual_address(Buffer const & buffer);
	srat::slice<u8> buffer_host_address(Buffer const & buffer);

	struct BufferUploadInfo {
		Buffer buffer;
		u64 byteOffset;
		srat::slice<u8 const> data;
	};
	void buffer_upload(BufferUploadInfo const & uploadInfo);
}

// -----------------------------------------------------------------------------
// -- image
// -----------------------------------------------------------------------------

namespace vkof
{
	struct Image { u64 id; };

	struct ImageCreateInfo {
		u32 width;
		u32 height;
		ImageFormat format;
		u32 mipLevels;
		srat::slice<u8 const> optInitialData;
	};

	Image image_create(ImageCreateInfo const & createInfo);
	void image_destroy(Image const & image);

	u32 image_width(Image const & image);
	u32 image_height(Image const & image);

	struct Sampler { u64 id; };

	enum struct SamplerFilter {
		nearest,
		linear,
	};

	enum struct SamplerAddressMode {
		repeat,
		mirrored_repeat,
		clamp_to_edge,
	};

	struct SamplerCreateInfo {
		SamplerFilter magFilter;
		SamplerFilter minFilter;
		SamplerAddressMode addressModeU;
		SamplerAddressMode addressModeV;
		SamplerAddressMode addressModeW;
	};
	Sampler sampler_create(SamplerCreateInfo const & createInfo);
	void sampler_destroy(Sampler const & sampler);

	// for use with sampler2d
	struct ImageSamplerHandleInfo {
		Image image;
		Sampler sampler;
	};
	u64 image_sampler_handle(ImageSamplerHandleInfo const & info);

	// for use with image2d
	struct ImageStorageHandleInfo {
		Image image;
		u32 mipLevel;
	};
	u64 image_storage_handle(ImageStorageHandleInfo const & info);


}

// -----------------------------------------------------------------------------
// -- draw/dispatch
// -----------------------------------------------------------------------------

namespace vkof
{

	struct CmdDraw {
		CommandBuffer cmd;
		Pipeline pipeline;
		srat::slice<u8 const> pushconstant;
		u32 vertexCount;
		u32 instanceCount;
	};
	void cmd_draw(CmdDraw const & draw);

	struct CmdDispatch {
		CommandBuffer cmd;
		Pipeline pipeline;
		srat::slice<u8 const> pushconstant;
		u32 groupCountX;
		u32 groupCountY;
		u32 groupCountZ;
	};
	void cmd_dispatch(CmdDispatch const & dispatch);

	struct CmdCopyImage {
		CommandBuffer cmd;
		Image srcImage;
		u32 srcBaseMipLevel { 0 };
		u32 srcOffsetX { 0 };
		u32 srcOffsetY { 0 };
		Image dstImage;
		u32 dstBaseMipLevel { 0 };
		u32 dstOffsetX { 0 };
		u32 dstOffsetY { 0 };
		u32 width;
		u32 height;
	};
	void cmd_copy_image(CmdCopyImage const & copy);
}

// -----------------------------------------------------------------------------
// -- renderpass
// -----------------------------------------------------------------------------

namespace vkof
{
	struct RenderNode { u64 id; };
	struct TransientImage { u64 id; };
	struct TransientBuffer { u64 id; };

	enum struct RenderNodeAccess { read, write, readWrite, };
	enum struct RenderNodeLoadOp { load, clear, discard, };

	enum struct CommandQueue {
		graphics,
		compute,
	};

	struct TransientImageCreateInfo {
		ImageFormat format;
		f32 scaleWidth;
		f32 scaleHeight;
		u32 mipLevels;
		bool isDoubleBuffered;
	};

	TransientImage transient_image_create(
		TransientImageCreateInfo const & createInfo
	);

	struct TransientBufferCreateInfo {
		u64 byteCount;
		bool isDoubleBuffered;
	};
	TransientBuffer transient_buffer_create(
		TransientBufferCreateInfo const & createInfo
	);

	Image transient_image_get_image(TransientImage const & transientImage);
	Buffer transient_buffer_get_buffer(TransientBuffer const & transientBuffer);

	struct RenderNodeCreateInfo {
		CommandQueue queue;
	};
	RenderNode render_node_create(RenderNodeCreateInfo const & createInfo);
	void render_node_destroy(RenderNode const & node);

	struct RenderNodeImageInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeAccess access;
	};
	void render_node_add_image(RenderNodeImageInfo const & info);

	struct RenderNodeBufferInfo {
		RenderNode node;
		TransientBuffer buffer;
		RenderNodeAccess access;
	};
	void render_node_add_buffer(RenderNodeBufferInfo const & info);

	struct RenderNodeAttachmentColorInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeLoadOp loadOp;
		u32 mipLevel;
		u32 colorIndex;
		srat::slice<f32 const> clearColor;
	};
	void render_node_attachment_color(
		RenderNodeAttachmentColorInfo const & info
	);

	struct RenderNodeAttachmentDepthInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeLoadOp loadOp;
		u32 mipLevel;
		srat::slice<f32 const> clearDepth;
	};
	void render_node_attachment_depth(
		RenderNodeAttachmentDepthInfo const & info
	);

	struct RenderNodeCallbackInfo {
		RenderNode node;
		std::function<void(vkof::CommandBuffer const & cmd)> callback;
	};
	void render_node_callback(RenderNodeCallbackInfo const & info);

	struct RenderGraphExecuteInfo {
		srat::slice<RenderNode const> nodes;
		srat::slice<u8 const> rootPushconstant;
		// if set, vkof copies this image to the swapchain before ImGui
		TransientImage finalImage { 0 };
	};
	void render_graph_execute(RenderGraphExecuteInfo const & executeInfo);
}

// -----------------------------------------------------------------------------
// -- lifetime
// -----------------------------------------------------------------------------

namespace vkof
{
	void init();
	// Headless init: no window, no surface, no swapchain.
	// Uses a fixed offscreen extent (default 512x512).
	// render_graph_execute skips acquire/present in this mode.
	void init_headless(u32 width = 512, u32 height = 512);
	void shutdown();
	void device_wait_idle();
	GLFWwindow * window();

	// Call once per frame before building any ImGui widgets.
	// render_graph_execute renders and presents the UI automatically.
	void imgui_begin();

	// Save a transient image to a PNG file. Blocks until the GPU is idle.
	// Intended for headless rendering — call after render_graph_execute.
	void screenshot(TransientImage const & image, char const * const path);
}

// -----------------------------------------------------------------------------
// -- buffer (extended)
// -----------------------------------------------------------------------------

namespace vkof
{
	// GPU -> CPU blocking copy. Symmetric counterpart to buffer_upload.
	struct BufferDownloadInfo {
		Buffer buffer;
		u64 byteOffset;
		srat::slice<u8> dst;  // caller-owned destination slice
	};
	void buffer_download(BufferDownloadInfo const & info);
}
