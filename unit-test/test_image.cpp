// unit-test/test_image.cpp — 11 test cases for the vkof image/sampler API

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

TEST_SUITE("[headless]") {

// 1. Create/destroy r8g8b8a8_unorm image
TEST_CASE("image: create/destroy r8g8b8a8_unorm") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 2. Width and height accessors match creation parameters
TEST_CASE("image: width/height accessors") {
	auto img = vkof::image_create({
		.width		  = 128,
		.height		 = 256,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(vkof::image_width(img)  == 128u);
	CHECK(vkof::image_height(img) == 256u);
	vkof::image_destroy(img);
}

// 3. r16g16b16a16_sfloat format
TEST_CASE("image: r16g16b16a16_sfloat format") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 4. r32_float format
TEST_CASE("image: r32_float format") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::r32_float,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 5. Depth (d24_unorm_s8_uint) format
TEST_CASE("image: d24_unorm_s8_uint format") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::d24_unorm_s8_uint,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 6. mipLevels > 1
TEST_CASE("image: mipLevels=4") {
	auto img = vkof::image_create({
		.width		  = 256,
		.height		 = 256,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 4,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 7. image_sampler_handle returns a non-zero handle (uses NVX extension)
TEST_CASE("image: image_sampler_handle non-zero") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::nearest,
		.minFilter	= vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});
	CHECK(vkof::image_sampler_handle({ .image = img, .sampler = smp }) != 0u);
	vkof::sampler_destroy(smp);
	vkof::image_destroy(img);
}

// 8. image_storage_handle returns a non-zero handle
TEST_CASE("image: image_storage_handle non-zero") {
	auto img = vkof::image_create({
		.width		  = 64,
		.height		 = 64,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(
		vkof::image_storage_handle({ .image = img, .mipLevel = 0 }) != 0u
	);
	vkof::image_destroy(img);
}

// 9. Small (1×1) image
TEST_CASE("image: 1x1 image") {
	auto img = vkof::image_create({
		.width		  = 1,
		.height		 = 1,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	CHECK(vkof::image_width(img)  == 1u);
	CHECK(vkof::image_height(img) == 1u);
	vkof::image_destroy(img);
}

// 11. Large (1024×1024) image
TEST_CASE("image: 1024x1024 image") {
	auto img = vkof::image_create({
		.width		  = 1024,
		.height		 = 1024,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

// 12. image_generate_mipmaps does not crash
TEST_CASE("image: generate_mipmaps no crash") {
	auto img = vkof::image_create({
		.width = 256,
		.height = 256,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 4,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	REQUIRE(img.id != 0);
	vkof::image_generate_mipmaps(img);
	test::gpu_wait();
	vkof::image_destroy(img);
}

// 13. image created with optInitialData round-trips via storage image readback
TEST_CASE("image: initial data round-trip") {
	constexpr u32 kW = 4u;
	constexpr u32 kH = 4u;
	constexpr u32 kCount = kW * kH;

	std::vector<f32> pixels(kCount);
	for (u32 i = 0u; i < kCount; ++i) {
		pixels[i] = (f32)i / (f32)kCount;
	}

	auto img = vkof::image_create({
		.width = kW,
		.height = kH,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(pixels.data()),
			kCount * sizeof(f32)
		),
	});
	REQUIRE(img.id != 0);

	u32 const imgHandle = (
		vkof::image_storage_handle({ .image = img, .mipLevel = 0u })
	);
	REQUIRE(imgHandle != 0u);

	auto outBuf = vkof::buffer_create({
		.byteCount = kCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "image_read_r32.comp",
	});
	REQUIRE(pl.id != 0);

	struct Push {
		u64 dstVa;
		u32 imgHandle;
		u32 width;
		u32 height;
	};
	Push const push {
		.dstVa = vkof::buffer_virtual_address(outBuf),
		.imgHandle = imgHandle,
		.width = kW,
		.height = kH,
	};

	vkof::RenderNode const node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = img,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = pl,
				.pushconstant = test::as_bytes(push),
				.groupCountX = (kCount + 63u) / 64u,
				.groupCountY = 1u,
				.groupCountZ = 1u,
			});
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);

	auto const result = test::readback<f32>(outBuf, 0u, kCount);
	for (u32 i = 0u; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(result[i] == doctest::Approx(pixels[i]).epsilon(0.001f));
	}

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::image_destroy(img);
}

TEST_CASE("image: rapid create/destroy cycle") {
	for (u32 i = 0u; i < 128u; i++) {
		vkof::Image const img = vkof::image_create({
			.width = 4u,
			.height = 4u,
			.format = vkof::ImageFormat::r8g8b8a8_unorm,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		CHECK(img.id != 0u);
		vkof::image_destroy(img);
	}
}

} // TEST_SUITE("[headless]")
