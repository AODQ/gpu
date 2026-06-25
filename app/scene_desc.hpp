#pragma once

#include "shared/light_shared.h"
#include <srat/core-math.hpp>

#include <filesystem>
#include <string>
#include <vector>

struct SceneInstance
{
	std::string filename;
	f32v3 position;
	f32v3 rotation;
	f32 scale;
};

struct SceneDescDdgiVolume
{
	f32v3 origin;
	f32v3 probeSpacing;
	u32v3 probeCounts;
	u32 raysPerProbe;
	u32 irradianceResolution;
	u32 depthResolution;

	inline f32v3 max() const {
		return {
			origin.x + probeSpacing.x * (f32)probeCounts.x,
			origin.y + probeSpacing.y * (f32)probeCounts.y,
			origin.z + probeSpacing.z * (f32)probeCounts.z,
		};
	}
	inline f32v3 min() const { return origin; }
};

struct SceneDesc
{
	std::vector<SceneInstance> instances;
	std::vector<GpuLight> lights;
	std::vector<SceneDescDdgiVolume> ddgiVolumes;
};

[[nodiscard]] SceneDesc scene_desc_load(std::filesystem::path const & path);

void scene_desc_save(SceneDesc const & desc, std::filesystem::path const & path);

void scene_desc_add_instance(
	SceneDesc & desc,
	std::string filename,
	f32v3 const position = {}
);

// returns index of selected instance, -1 if none selected
[[nodiscard]] i32 scene_desc_imgui(
	SceneDesc & desc,
	std::filesystem::path & savePath
);
