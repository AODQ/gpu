// unit-test/test_hot_reload.cpp
// Verifies that modifying a #include'd file (not the primary shader) triggers
// a hot reload on the next render_graph_execute call.

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"

#include <filesystem>
#include <fstream>

struct HotReloadPush {
	u64 dstVa;
	u32 count;
	u32 _pad = 0u;
};

static constexpr u32 skLocalSize = 64u;

// Primary shader: writes RESULT_VALUE (defined in hot_reload_params.glsl) to
// every element of the output buffer.
static char const * skShaderSrc = R"glsl(
#version 460
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : require

layout(buffer_reference, buffer_reference_align = 4) buffer UintBuffer {
	uint data[];
};

layout(push_constant) uniform PushConstants {
	layout(offset = 128) uint64_t dstVa;
	uint count;
} pc;

layout(local_size_x = 64) in;

#include "hot_reload_params.glsl"

void main() {
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= pc.count) { return; }
	UintBuffer(pc.dstVa).data[idx] = RESULT_VALUE;
}
)glsl";

TEST_SUITE("[headless]") {

TEST_CASE("pipeline: hot reload triggers on include file change") {
	std::string const tmpDir = "/tmp/vkof-test-hotreload";
	std::string const shaderPath = tmpDir + "/hot_reload.comp";
	std::string const paramsPath = tmpDir + "/hot_reload_params.glsl";

	std::filesystem::remove_all(tmpDir);
	std::filesystem::create_directories(tmpDir);

	{ std::ofstream f(shaderPath); f << skShaderSrc; }
	{ std::ofstream f(paramsPath); f << "#define RESULT_VALUE 42u\n"; }

	vkof::Pipeline const pl = vkof::pipeline_compute_create({
		.pathCompute = shaderPath.c_str(),
	});
	REQUIRE(pl.id != 0u);

	vkof::Buffer const buf = test::make_buffer_u32(skLocalSize);
	HotReloadPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.count = skLocalSize,
	};

	// dispatch 1: baseline — include defines 42
	test::dispatch(pl, push, 1u);
	{
		auto const vals = test::readback<u32>(buf, 0u, skLocalSize);
		for (u32 i = 0u; i < skLocalSize; ++i) {
			REQUIRE(vals[i] == 42u);
		}
	}

	// change only the included file, not the primary shader
	{ std::ofstream f(paramsPath); f << "#define RESULT_VALUE 99u\n"; }

	// dispatch 2: render_graph_execute detects the changed include mtime and
	// recompiles at the end of this frame; the GPU work here still ran with
	// the old pipeline so the result is still 42
	test::dispatch(pl, push, 1u);
	test::readback<u32>(buf, 0u, skLocalSize); // drain; value unused

	// dispatch 3: uses the reloaded pipeline — expect 99
	test::dispatch(pl, push, 1u);
	{
		auto const vals = test::readback<u32>(buf, 0u, skLocalSize);
		for (u32 i = 0u; i < skLocalSize; ++i) {
			CHECK(vals[i] == 99u);
		}
	}

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	std::filesystem::remove_all(tmpDir);
}

} // TEST_SUITE("[headless]")
