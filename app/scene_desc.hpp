#pragma once

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

struct SceneDesc
{
	std::vector<SceneInstance> instances;
};

[[nodiscard]] SceneDesc scene_desc_load(std::filesystem::path const & path);

void scene_desc_save(SceneDesc const & desc, std::filesystem::path const & path);

void scene_desc_add_instance(SceneDesc & desc, std::string filename);

[[nodiscard]] i32 scene_desc_imgui(SceneDesc & desc, std::filesystem::path & savePath);
