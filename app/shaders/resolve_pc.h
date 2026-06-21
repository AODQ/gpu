#ifdef __cplusplus
#pragma once
#include <mor/mor_shared.h>
#endif

#include "shared/global_pc.h"

struct GpuResolveModelIndirect {
	BDA(GpuMorMeshletBuffer) meshlets;
	BDA(GpuMorMaterialBuffer) materials;
};

#ifdef __cplusplus
using GpuResolveModelIndirectBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuResolveModelIndirectBuffer {
	GpuResolveModelIndirect data[];
};
#endif

struct GpuResolvePC {
	u32 visibilityImageHandle;
	u32 outputImageHandle;
	BDA(GpuResolveModelIndirectBuffer) models;
};

#ifndef __cplusplus
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuResolvePC resolve;
} pc;
#endif
