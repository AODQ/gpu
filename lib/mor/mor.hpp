#pragma once

#include <mor/mor_shared.h>
#include <srat/core-types.hpp>

namespace mor {

struct Scene;
struct GpuScene { u64 id; };

struct Buffers {
	u64 meshlets;
	u64 instances;
	u64 positions;
	u64 attributes;
	u64 meshletVerts;  // u32[] — global vertex indices per meshlet
	u64 meshletTris;   // u8[]  — local triangle indices (3 per tri)
};

Scene *  scene_create();
void     scene_destroy(Scene *);
void     scene_load_gltf(Scene *, char const * path);

u32 scene_instance_count(Scene const *);
u32 scene_meshlet_count(Scene const *);
u32 scene_vertex_count(Scene const *);

GpuScene scene_gpu_upload(Scene const *);
void     scene_gpu_destroy(GpuScene);

Buffers  scene_gpu_buffers(GpuScene);
u32      scene_gpu_meshlet_count(GpuScene);

} // namespace mor
