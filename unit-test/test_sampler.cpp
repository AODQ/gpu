// unit-test/test_sampler.cpp — 8 test cases for the vkof sampler API

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

TEST_SUITE("[headless]") {

// 1. Create/destroy nearest-filter sampler
TEST_CASE("sampler: create/destroy nearest") {
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::nearest,
		.minFilter	= vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});
	CHECK(smp.id != 0);
	vkof::sampler_destroy(smp);
}

// 2. Create/destroy linear-filter sampler
TEST_CASE("sampler: create/destroy linear") {
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::linear,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	CHECK(smp.id != 0);
	vkof::sampler_destroy(smp);
}

// 3. Mirrored-repeat address mode
TEST_CASE("sampler: mirrored_repeat address mode") {
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::linear,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::mirrored_repeat,
		.addressModeV = vkof::SamplerAddressMode::mirrored_repeat,
		.addressModeW = vkof::SamplerAddressMode::mirrored_repeat,
	});
	CHECK(smp.id != 0);
	vkof::sampler_destroy(smp);
}

// 4. Mixed address modes (U=repeat, V=clamp, W=mirror)
TEST_CASE("sampler: mixed address modes") {
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::nearest,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::mirrored_repeat,
	});
	CHECK(smp.id != 0);
	vkof::sampler_destroy(smp);
}

// 5. image_sampler_handle with nearest sampler returns non-zero
TEST_CASE("sampler: image_sampler_handle nearest non-zero") {
	auto img = vkof::image_create({
		.width		  = 16,
		.height		 = 16,
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

// 6. image_sampler_handle with linear sampler returns non-zero
TEST_CASE("sampler: image_sampler_handle linear non-zero") {
	auto img = vkof::image_create({
		.width		  = 16,
		.height		 = 16,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	auto smp = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::linear,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	CHECK(vkof::image_sampler_handle({ .image = img, .sampler = smp }) != 0u);
	vkof::sampler_destroy(smp);
	vkof::image_destroy(img);
}

// 7. Multiple samplers created simultaneously have distinct ids
TEST_CASE("sampler: multiple samplers simultaneously") {
	auto s0 = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::nearest,
		.minFilter	= vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});
	auto s1 = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::linear,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	CHECK(s0.id != 0);
	CHECK(s1.id != 0);
	CHECK(s0.id != s1.id);
	vkof::sampler_destroy(s0);
	vkof::sampler_destroy(s1);
}

// 8. Two different samplers return different image_sampler_handles
TEST_CASE("sampler: different samplers produce different handles") {
	auto img = vkof::image_create({
		.width		  = 16,
		.height		 = 16,
		.format		 = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels	  = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	auto s0 = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::nearest,
		.minFilter	= vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});
	auto s1 = vkof::sampler_create({
		.magFilter	= vkof::SamplerFilter::linear,
		.minFilter	= vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	u32 const h0 = vkof::image_sampler_handle({ .image = img, .sampler = s0 });
	u32 const h1 = vkof::image_sampler_handle({ .image = img, .sampler = s1 });
	CHECK(h0 != h1);
	vkof::sampler_destroy(s0);
	vkof::sampler_destroy(s1);
	vkof::image_destroy(img);
}

} // TEST_SUITE("[headless]")
