// unit-test/test_transient.cpp — 5 test cases for the vkof transient resource API

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

TEST_SUITE("[headless]") {

// 1. transient_image_create returns a non-zero handle
TEST_CASE("transient: image create returns valid handle") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 1.0f,
		.scaleHeight	  = 1.0f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});
	CHECK(ti.id != 0);
}

// 2. transient_image_get_image returns a valid (non-zero) image handle
TEST_CASE("transient: get_image returns valid handle") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 0.5f,
		.scaleHeight	  = 0.5f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});
	auto img = vkof::transient_image_get_image(ti);
	CHECK(img.id != 0);
}

// 3. transient_buffer_create returns a non-zero handle
TEST_CASE("transient: buffer create returns valid handle") {
	auto tb = vkof::transient_buffer_create({
		.byteCount		= 1024,
		.isDoubleBuffered = false,
	});
	CHECK(tb.id != 0);
}

// 4. transient_buffer_get_buffer returns a valid (non-zero) buffer handle
TEST_CASE("transient: get_buffer returns valid handle") {
	auto tb = vkof::transient_buffer_create({
		.byteCount		= 1024,
		.isDoubleBuffered = false,
	});
	auto buf = vkof::transient_buffer_get_buffer(tb);
	CHECK(buf.id != 0);
}

// 5. Double-buffered transient image provides non-zero handles on both frames
TEST_CASE("transient: double-buffered image has valid handles") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 1.0f,
		.scaleHeight	  = 1.0f,
		.mipLevels		= 1,
		.isDoubleBuffered = true,
	});
	// Get image on even frame (slot 0)
	auto img0 = vkof::transient_image_get_image(ti);
	CHECK(img0.id != 0);
	// Simulate a frame advance by calling render_graph_execute once (empty pass)
	{
		vkof::RenderNode emptyNode = vkof::render_node_create(
			{ .queue = vkof::CommandQueue::compute }
		);
		vkof::render_graph_execute({
			.nodes			= srat::slice<vkof::RenderNode const>(&emptyNode, 1),
			.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
		});
		vkof::render_node_destroy(emptyNode);
		test::gpu_wait();
	}
	// Get image on odd frame (slot 1)
	auto img1 = vkof::transient_image_get_image(ti);
	CHECK(img1.id != 0);
}

} // TEST_SUITE("[headless]")
