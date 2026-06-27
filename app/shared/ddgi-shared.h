
#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#include <srat/core-math.hpp>
#endif

#include "global_pc.h"

#ifdef __cplusplus
static constexpr u32 skMaxDdgiCascades = 8u;
#else
const uint skMaxDdgiCascades = 8u;
#endif

struct GpuDdgiGrid {
	f32v3 origin;
	u32v3 probeCounts;
	f32v3 probeSpacing;
	u32 irradianceStorageHandle;
	u32 depthStorageHandle;
	u32 irradianceSamplerHandle;
	u32 depthSamplerHandle;
	// offset into the probes to scroll
	u32v3 scrollOffset;
	// a slice of invalid probes in the grid based off delta of previous frame position
	u32v3 invalidStart;
	u32v3 invalidCount;
};

struct GpuDdgiCascades {
	GpuDdgiGrid grids[skMaxDdgiCascades];
	u32 count;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuDdgiCascadesBuffer {
	GpuDdgiCascades data;
};

#ifndef DDGI_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuDdgiGrid ddgi;
} pc;

// atlas slot = (worldIndex + scrollOffset) % probeCounts
uvec3 ddgiAtlasSlot(const uvec3 probe) {
	return uvec3(
		(probe.x + pc.ddgi.scrollOffset.x) % pc.ddgi.probeCounts.x,
		(probe.y + pc.ddgi.scrollOffset.y) % pc.ddgi.probeCounts.y,
		(probe.z + pc.ddgi.scrollOffset.z) % pc.ddgi.probeCounts.z
	);
}

// true if this atlas slot entered the grid this frame and has no valid history
bool ddgiIsNewProbe(const uvec3 slot) {
	bool isNew = false;
	const uint cx = pc.ddgi.probeCounts.x;
	if (pc.ddgi.invalidCount.x > 0u) {
		isNew = isNew || ((slot.x - pc.ddgi.invalidStart.x + cx) % cx < pc.ddgi.invalidCount.x);
	}
	const uint cy = pc.ddgi.probeCounts.y;
	if (pc.ddgi.invalidCount.y > 0u) {
		isNew = isNew || ((slot.y - pc.ddgi.invalidStart.y + cy) % cy < pc.ddgi.invalidCount.y);
	}
	const uint cz = pc.ddgi.probeCounts.z;
	if (pc.ddgi.invalidCount.z > 0u) {
		isNew = isNew || ((slot.z - pc.ddgi.invalidStart.z + cz) % cz < pc.ddgi.invalidCount.z);
	}
	return isNew;
}

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
