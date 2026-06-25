
#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#include <srat/core-math.hpp>
#endif

struct GpuDdgiGrid {
	f32v3 origin;
	u32v3 probeCounts;
	f32v3 probeSpacing;
	u32 raysPerProbe;
	u32 irradianceResolution;
	u32 depthResolution;
};
