#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_debug_printf : require

layout(set = 0, binding = 0) uniform sampler2D vkofTextures[];
layout(set = 0, binding = 4) uniform sampler3D vkofTextures3D[];

#define DDGI_NO_PC
#include "shared/ddgi-shared.h"

layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
} pc;

layout(location = 0) in vec3 inNormal;
layout(location = 1) perprimitiveEXT flat in uint inSphereIdx;

layout(location = 0) out vec4 outColor;

void main() {
	if (pc.global.ddgiGrid == 0u) {
		outColor = vec4(inNormal * 0.5 + vec3(0.5), 1.0);
		return;
	}

	const GpuDdgiCascades cascades = (
		GpuDdgiCascadesBuffer(pc.global.ddgiGrid).data
	);

	// probes are ordered: all cascade-0 probes, then cascade-1, etc.
	const uint probesPerCascade = (
		cascades.grids[0].probeCounts.x
		* cascades.grids[0].probeCounts.y
		* cascades.grids[0].probeCounts.z
	);
	if (inSphereIdx >= probesPerCascade * cascades.count) {
		outColor = vec4(inNormal * 0.5 + vec3(0.5), 1.0);
		return;
	}

	const uint ci = inSphereIdx / probesPerCascade;
	const uint localIdx = inSphereIdx % probesPerCascade;
	const GpuDdgiGrid grid = cascades.grids[ci];

	// decode (px, py, pz) from flat index within this cascade
	const uint px = localIdx % grid.probeCounts.x;
	const uint py = (localIdx / grid.probeCounts.x) % grid.probeCounts.y;
	const uint pz = localIdx / (grid.probeCounts.x * grid.probeCounts.y);

	// map normal direction into irradiance atlas texel
	const uint apx = (px + grid.scrollOffset.x) % grid.probeCounts.x;
	const uint apy = (py + grid.scrollOffset.y) % grid.probeCounts.y;
	const uint apz = (pz + grid.scrollOffset.z) % grid.probeCounts.z;
	const vec2 octUv = octahedronEncode(normalize(inNormal));
	const vec2 localUv = clamp((octUv * 0.5 + 0.5) * 8.0, 0.5, 7.5);
	const vec3 atlasUv = (
		vec3(
			(float(apx) * 8.0 + localUv.x) / (float(grid.probeCounts.x) * 8.0),
			(float(apy) * 8.0 + localUv.y) / (float(grid.probeCounts.y) * 8.0),
			(float(apz) + 0.5) / float(grid.probeCounts.z)
		)
	);

	const vec3 irradiance = (
		texture(
			vkofTextures3D[nonuniformEXT(grid.irradianceSamplerHandle)],
			atlasUv
		).rgb
	);
	outColor = vec4(irradiance, 1.0);
}
