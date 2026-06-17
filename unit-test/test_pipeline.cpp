// unit-test/test_pipeline.cpp — 6 test cases for the vkof pipeline API

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

TEST_SUITE("[headless]") {

// 1. pipeline_compute_create with a valid shader succeeds
TEST_CASE("pipeline: compute create valid shader") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

// 2. pipeline_compute_create with a valid passthrough shader also succeeds
TEST_CASE("pipeline: compute create passthrough shader") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "passthrough.comp",
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

// 3. pipeline_destroy does not crash
TEST_CASE("pipeline: destroy does not crash") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	REQUIRE(pl.id != 0);
	vkof::pipeline_destroy(pl);
	// double-destroy: handle is gone from pool, should be silently ignored
	vkof::pipeline_destroy(pl);
}

// 4. pipeline_compute_create with a non-existent path returns id=0
TEST_CASE("pipeline: invalid path returns null pipeline") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "does_not_exist.comp",
	});
	CHECK(pl.id == 0);
	// Destroying a null pipeline should be a no-op
	vkof::pipeline_destroy(pl);
}

// 5. Creating two pipelines simultaneously yields distinct ids
TEST_CASE("pipeline: two compute pipelines have distinct ids") {
	auto p0 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	auto p1 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "passthrough.comp",
	});
	REQUIRE(p0.id != 0);
	REQUIRE(p1.id != 0);
	CHECK(p0.id != p1.id);
	vkof::pipeline_destroy(p0);
	vkof::pipeline_destroy(p1);
}

// 6. Re-creating after destroy succeeds and produces a new (non-zero) id
TEST_CASE("pipeline: re-create after destroy") {
	auto p0 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	REQUIRE(p0.id != 0);
	vkof::pipeline_destroy(p0);

	auto p1 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	CHECK(p1.id != 0);
	vkof::pipeline_destroy(p1);
}

// 7. pipeline_graphics_create with minimal mesh + frag shaders
TEST_CASE("pipeline: graphics create") {
	static constexpr vkof::ImageFormat kColorFmts[] = {
		vkof::ImageFormat::r8g8b8a8_unorm,
	};
	auto pl = vkof::pipeline_graphics_create({
		.pathMesh = TEST_SHADER_DIR "test_minimal.mesh",
		.pathFragment = TEST_SHADER_DIR "test_minimal.frag",
		.attachmentColorFormats = srat::slice<vkof::ImageFormat const>(kColorFmts, 1u),
		.attachmentDepthStencilFormat = vkof::ImageFormat::none,
		.depthTest = vkof::DepthTest::write_off_test_off,
		.cullMode = vkof::CullMode::none,
		.blendMode = vkof::BlendMode::none,
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

// 8. pipeline_graphics_create with invalid shader path returns id=0
TEST_CASE("pipeline: graphics invalid path returns null") {
	static constexpr vkof::ImageFormat kColorFmts[] = {
		vkof::ImageFormat::r8g8b8a8_unorm,
	};
	auto pl = vkof::pipeline_graphics_create({
		.pathMesh = TEST_SHADER_DIR "does_not_exist.mesh",
		.pathFragment = TEST_SHADER_DIR "test_minimal.frag",
		.attachmentColorFormats = srat::slice<vkof::ImageFormat const>(kColorFmts, 1u),
		.attachmentDepthStencilFormat = vkof::ImageFormat::none,
		.depthTest = vkof::DepthTest::write_off_test_off,
		.cullMode = vkof::CullMode::none,
		.blendMode = vkof::BlendMode::none,
	});
	CHECK(pl.id == 0);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("pipeline: rapid compute create/destroy cycle") {
	for (u32 i = 0u; i < 32u; i++) {
		vkof::Pipeline const pl = vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "passthrough.comp",
		});
		CHECK(pl.id != 0u);
		vkof::pipeline_destroy(pl);
	}
}

} // TEST_SUITE("[headless]")
