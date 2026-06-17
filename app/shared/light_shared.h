#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#include <srat/core-math.hpp>
#endif

struct GpuLight {
	f32v3 position;
	f32 radius;
	f32v3 color;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuLightBuffer {
	GpuLight data[];
};
#endif
