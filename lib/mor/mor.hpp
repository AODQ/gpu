#pragma once

#include <mor/mor_shared.h>
#include <srat/core-types.hpp>

namespace mor {

struct Scene { u64 id; };
struct GpuScene { u64 id; };

struct Buffers {
	u64 meshlets;
	u64 materials;
	u64 textures;
	u64 instances;
	u64 positions;
	u64 attributes;
	u64 meshletVerts;
	u64 meshletTris;
};

Scene scene_create();
void scene_destroy(Scene const & scene);
void scene_load_gltf(Scene const & scene, char const * const path);

u32 scene_instance_count(Scene const & scene);
u32 scene_meshlet_count(Scene const & scene);
u32 scene_vertex_count(Scene const & scene);

void scene_imgui_debug(Scene const & scene);

GpuScene scene_gpu_upload(Scene const & scene);
void scene_gpu_destroy(GpuScene const & scene);

Buffers scene_gpu_buffers(GpuScene const & scene);
u32 scene_gpu_meshlet_count(GpuScene const & scene);

void sampler_cache_destroy();

} // namespace mor
