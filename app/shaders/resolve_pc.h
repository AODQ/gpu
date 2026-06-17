#ifdef __cplusplus
#pragma once
#include <mor/mor_shared.h>
#endif

#include "shared/global_pc.h"
#include "shared/light_shared.h"

struct GpuResolveModelIndirect {
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorMaterialBuffer) materials;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorVertexAttributeBuffer) attributes;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	f32m44 modelMatrix;
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
	VA(GpuResolveModelIndirectBuffer) models;
};

#ifndef __cplusplus
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuResolvePC resolve;
} pc;
#endif
