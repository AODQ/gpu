# vkof

Thin Vulkan abstraction. All public types are opaque `u64` handles — zero means invalid. No raw Vulkan types in the public API.

Push constants are split into two windows:
- `[0, 128)` — root, set once per `render_graph_execute` via `rootPushconstant`
- `[128, 256)` — per-draw/dispatch, set per `cmd_draw` / `cmd_dispatch`

Shaders are compiled from GLSL at runtime (no pre-compiled `.spv`). Hot reload is automatic — pipelines recompile when the source file changes on disk.

---

## Lifetime

```cpp
vkof::init();          // windowed (GLFW)
vkof::init_headless(); // no window, for tests
// ...
vkof::device_wait_idle();
vkof::shutdown();
```

---

## Pipelines

```cpp
vkof::Pipeline pipe = vkof::pipeline_graphics_create({
	.pathMesh = "/abs/path/scene.mesh",
	.pathFragment = "/abs/path/scene.frag",
	.attachmentColorFormats = { &fmt, 1 },
	.attachmentDepthStencilFormat = vkof::ImageFormat::d24_unorm_s8_uint,
	.depthTest = vkof::DepthTest::write_on_test_on,
	.cullMode = vkof::CullMode::back,
	.blendMode = vkof::BlendMode::none,
	.defines = { defines, defineCount },     // optional, passed as -DFOO=1
	.includePaths = { paths, pathCount },    // optional, passed as -I/path
});

vkof::Pipeline comp = vkof::pipeline_compute_create({
	.pathCompute = "/abs/path/pass.comp",
	.defines = { defines, defineCount },
	.includePaths = { paths, pathCount },
});

vkof::pipeline_destroy(pipe);
```

---

## Buffers

```cpp
vkof::Buffer buf = vkof::buffer_create({
	.byteCount = sizeof(data),
	.memory = vkof::BufferMemory::DeviceOnly,
});
vkof::buffer_upload({
	.buffer = buf,
	.byteOffset = 0,
	.data = { ptr, bytes },
});

u64 bda = vkof::buffer_virtual_address(buf);

// host-mapped write (HostWritable only)
srat::slice<u8> mapped = vkof::buffer_host_address(buf);

vkof::buffer_destroy(buf);
```

In GLSL, access via buffer references:

```glsl
layout(buffer_reference, scalar) buffer MyBuf { MyStruct data[]; };
MyBuf buf = MyBuf(pc.draw.myBufAddr);
MyStruct s = buf.data[gl_WorkGroupID.x];
```

---

## Images

```cpp
vkof::Image img = vkof::image_create({
	.width = 1024,
	.height = 1024,
	.format = vkof::ImageFormat::r8g8b8a8_unorm,
	.mipLevels = 1,
	.optInitialData = { pixels, byteCount },
});

vkof::Sampler samp = vkof::sampler_create({
	.magFilter = vkof::SamplerFilter::linear,
	.minFilter = vkof::SamplerFilter::linear,
	.addressModeU = vkof::SamplerAddressMode::repeat,
	.addressModeV = vkof::SamplerAddressMode::repeat,
	.addressModeW = vkof::SamplerAddressMode::repeat,
});

// bindless handles — pass as u64 to shaders
u64 samplerHandle = vkof::image_sampler_handle({ .image = img, .sampler = samp });
u64 storageHandle = vkof::image_storage_handle({ .image = img, .mipLevel = 0 });
```

---

## Render graph

Transient images are frame-lifetime resources managed by vkof. Double-buffered ones get a fresh copy each frame.

```cpp
vkof::TransientImage color = vkof::transient_image_create({
	.format = vkof::ImageFormat::r8g8b8a8_unorm,
	.scaleWidth = 1.0f,
	.scaleHeight = 1.0f,
	.mipLevels = 1,
	.isDoubleBuffered = true,
});
vkof::TransientImage depth = vkof::transient_image_create({
	.format = vkof::ImageFormat::d24_unorm_s8_uint,
	.scaleWidth = 1.0f,
	.scaleHeight = 1.0f,
	.mipLevels = 1,
	.isDoubleBuffered = true,
});

vkof::RenderNode node = vkof::render_node_create({
	.queue = vkof::CommandQueue::graphics,
});

static f32 kClear[] = { 0.0f, 0.0f, 0.0f, 1.0f };
static f32 kDepth = 1.0f;
vkof::render_node_attachment_color({
	.node = node,
	.image = color,
	.loadOp = vkof::RenderNodeLoadOp::clear,
	.mipLevel = 0,
	.colorIndex = 0,
	.clearColor = { kClear, 4 },
});
vkof::render_node_attachment_depth({
	.node = node,
	.image = depth,
	.loadOp = vkof::RenderNodeLoadOp::clear,
	.mipLevel = 0,
	.clearDepth = { &kDepth, 1 },
});
vkof::render_node_callback({
	.node = node,
	.callback = [&](vkof::CommandBuffer const & cmd) {
		vkof::cmd_draw({
			.cmd = cmd,
			.pipeline = pipe,
			.pushconstant = { (u8 const *)&drawPC, sizeof(drawPC) },
			.vertexCount = meshletCount,
			.instanceCount = 1,
		});
	},
});

vkof::RenderNode nodes[] = { node };
vkof::render_graph_execute({
	.nodes = { nodes, 1 },
	.rootPushconstant = { (u8 const *)&globalPC, sizeof(globalPC) },
	.finalImage = color,
});
vkof::render_node_destroy(node);
```

`cmd_draw` dispatches `vkCmdDrawMeshTasksEXT(cmd, vertexCount, instanceCount, 1)` — `vertexCount` is the meshlet group count.

---

## ImGui

```cpp
vkof::imgui_begin();
ImGui::Begin("my window");
// ...
ImGui::End();
// render_graph_execute presents ImGui automatically
```

Three overlays appear automatically when `imgui_begin()` is called:
- **profiler** — CPU and GPU time per render node
- **memory** — VMA heap usage and allocation counts
- **shader error** — last compile failure with red title bar, clears on successful recompile

---

## Headless / tests

```cpp
vkof::init_headless();
// dispatch compute, readback, assert — no display needed
vkof::shutdown();
```

See `unit-test/helpers.hpp` for `test::dispatch<Push>()` and `test::readback<T>()`.
