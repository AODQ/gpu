
#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#include <srat/core-math.hpp>
#endif

#include "global_pc.h"

struct GpuDdgiGrid {
	f32v3 origin;
	u32v3 probeCounts;
	f32v3 probeSpacing;
	u32 raysPerProbe;
	u32 irradianceStorageHandle;
	u32 depthStorageHandle;
	u32 irradianceSamplerHandle;
	u32 depthSamplerHandle;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuDdgiGridBuffer {
	GpuDdgiGrid data;
};

#ifndef DDGI_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuDdgiGrid ddgi;
} pc;

vec3 rayOrigin() {
	const uint px = gl_WorkGroupID.x;
	const uint py = gl_WorkGroupID.y % pc.ddgi.probeCounts.y;
	const uint pz = gl_WorkGroupID.y / pc.ddgi.probeCounts.y;
	return (
		pc.ddgi.origin
		+ vec3(
			(float(px) + 0.5) * pc.ddgi.probeSpacing.x,
			(float(py) + 0.5) * pc.ddgi.probeSpacing.y,
			(float(pz) + 0.5) * pc.ddgi.probeSpacing.z
		)
	);
}

vec3 rayDirection() {
	const float fibonacci = 1.61803398875;
	const float pi = 3.14159265359;
	const float it = float(gl_LocalInvocationIndex);
	const float n = float(gl_WorkGroupSize.x * gl_WorkGroupSize.y);
	const float theta = 2.39996322973 * it;
	const float phi = acos(1.0 - 2.0 * (it + 0.5) / n);
	return vec3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
}

#endif // DDGI_NO_PC

vec3 octahedronDecode(const vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	if (v.z < 0.0) {
		v.xy = (1.0 - abs(v.yx)) * sign(v.xy);
	}
	return normalize(v);
}

vec2 octahedronEncode(const vec3 n) {
	vec3 v = n / (abs(n.x) + abs(n.y) + abs(n.z));
	if (v.z < 0.0) {
		v.xy = (1.0 - abs(v.yx)) * sign(v.xy);
	}
	return v.xy;
}
#endif // __cplusplus
