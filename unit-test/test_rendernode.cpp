// unit-test/test_rendernode.cpp — render-node and render-graph tests

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

// Push struct for fill_u32 shader (per-dispatch, starts at byte 128)
struct FillPush {
	u64 dstVa;
	u32 value;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) {
	return (count + kLocalSize - 1) / kLocalSize;
}

// ---------------------------------------------------------------------------
// Headless render-node tests
// ---------------------------------------------------------------------------

TEST_SUITE("[headless]") {

// 1. render_node_create and render_node_destroy
TEST_CASE("rendernode: create/destroy") {
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	CHECK(node.id != 0);
	vkof::render_node_destroy(node);
}

// 2. Graphics queue render node is also valid
TEST_CASE("rendernode: graphics queue create/destroy") {
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	CHECK(node.id != 0);
	vkof::render_node_destroy(node);
}

// 3. render_graph_execute with an empty node list does not crash
TEST_CASE("rendernode: execute empty node list") {
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(nullptr, 0),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	test::gpu_wait();
}

// 4. render_graph_execute with a compute callback fires; result verifiable via readback
TEST_CASE("rendernode: compute callback executes on GPU") {
	constexpr u32 kCount = 256;
	constexpr u32 kValue = 0xABCD1234u;

	auto pl  = vkof::pipeline_compute_create({ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" });
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	srat::slice<u8 const> const pushBytes = test::as_bytes(push);

	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node	 = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd		  = cmd,
				.pipeline	 = pl,
				.pushconstant = pushBytes,
				.groupCountX  = groups_for(kCount),
				.groupCountY  = 1,
				.groupCountZ  = 1,
			});
		},
	});

	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	vkof::render_node_destroy(node);

	auto result = test::readback<u32>(buf, 0, kCount);
	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

// 5. render_node_add_image (with transient image) does not crash
TEST_CASE("rendernode: add_image barrier") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 0.5f,
		.scaleHeight	  = 0.5f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});

	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_image({
		.node   = node,
		.image  = ti,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

// 6. render_node_attachment_color: clear-only graphics node executes without crash
TEST_CASE("rendernode: attachment_color clear") {
	auto ti = vkof::transient_image_create({
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth = 0.25f,
		.scaleHeight = 0.25f,
		.mipLevels = 1,
		.isDoubleBuffered = false,
	});
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	f32 const clearColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	vkof::render_node_attachment_color({
		.node = node,
		.image = ti,
		.loadOp = vkof::RenderNodeLoadOp::clear,
		.mipLevel = 0u,
		.colorIndex = 0u,
		.clearColor = srat::slice<f32 const>(clearColor, 4u),
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

// 7. render_node_attachment_depth: clear-only depth node executes without crash
TEST_CASE("rendernode: attachment_depth clear") {
	auto depth = vkof::transient_image_create({
		.format = vkof::ImageFormat::d24_unorm_s8_uint,
		.scaleWidth = 0.25f,
		.scaleHeight = 0.25f,
		.mipLevels = 1,
		.isDoubleBuffered = false,
	});
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	f32 const clearDepth[] = { 1.0f };
	vkof::render_node_attachment_depth({
		.node = node,
		.image = depth,
		.loadOp = vkof::RenderNodeLoadOp::clear,
		.mipLevel = 0u,
		.clearDepth = srat::slice<f32 const>(clearDepth, 1u),
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

// 8. render_node_add_persistent_image: declared persistent image executes without crash
TEST_CASE("rendernode: persistent image declared") {
	auto img = vkof::image_create({
		.width = 64u,
		.height = 64u,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = srat::slice<u8 const>(nullptr, 0u),
	});
	REQUIRE(img.id != 0);

	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = img,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
	vkof::image_destroy(img);
}

// 9. Two sequential render_graph_execute calls with a compute node both succeed
TEST_CASE("rendernode: two sequential executes") {
	constexpr u32 kCount = 64;

	auto pl  = vkof::pipeline_compute_create({ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" });
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	for (u32 pass = 0; pass < 2; ++pass) {
		FillPush const push {
			.dstVa = vkof::buffer_virtual_address(buf),
			.value  = pass + 100u,
			.count  = kCount,
		};
		test::dispatch(pl, push, groups_for(kCount));
	}

	auto result = test::readback<u32>(buf, 0, kCount);
	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == 101u); // last dispatch wins
	}

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

} // TEST_SUITE("[headless]")

// ---------------------------------------------------------------------------
// Windowed render-node tests (only run with --windowed)
// ---------------------------------------------------------------------------

TEST_SUITE("[windowed]") {

// W1. render_node_create/destroy works in windowed mode
TEST_CASE("rendernode windowed: create/destroy") {
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	CHECK(node.id != 0);
	vkof::render_node_destroy(node);
}

// W2. render_graph_execute with empty node list runs in windowed mode
TEST_CASE("rendernode windowed: execute empty node list") {
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(nullptr, 0),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	test::gpu_wait();
}

} // TEST_SUITE("[windowed]")
