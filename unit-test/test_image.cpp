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

// 9. image_swapchain() returns the sentinel handle (id == u64(-1))
TEST_CASE("image: image_swapchain sentinel id") {
	auto sc = vkof::image_swapchain();
	CHECK(sc.id == static_cast<u64>(-1));
}

// 10. Small (1×1) image
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

} // TEST_SUITE("[headless]")
