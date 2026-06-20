#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

#ifdef __cplusplus
#include <mor/mor_shared.h>
#else
#include "mor/mor_shared.h"
#endif

struct SceneDrawPC {
	u64 meshlets;
	u64 materials;
	u64 textures;
	u64 instances;
	u64 positions;
	u64 attributes;
	u64 meshletVerts;
	u64 meshletTris;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
static_assert(sizeof(SceneDrawPC) <= 128, "SceneDrawPC must be <= 128 bytes");
#else
layout(push_constant, scalar) uniform PC {
	GlobalPC    global;
	SceneDrawPC draw;
} pc;
#endif
