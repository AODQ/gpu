#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

#ifdef __cplusplus
#include <mor/mor_shared.h>
#else
#include "mor/mor_shared.h"
#endif

struct GpuSceneDrawPC {
	u32 modelId;
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	f32m44 modelMatrix;
};

struct GpuSceneDeferredDrawInfo {
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorVertexAttributeBuffer) attributes;
	VA(GpuMorMaterialBuffer) materials;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
static_assert(sizeof(GpuSceneDrawPC) <= 128, "GpuSceneDrawPC must be <= 128 bytes");
#else
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuSceneDrawPC draw;
} pc;
#endif
