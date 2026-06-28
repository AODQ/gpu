#pragma once

#include <srat/core-types.hpp>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

struct AssetEntry
{
	std::string name;
	std::filesystem::path path;
};

struct AssetLibrary
{
	std::vector<AssetEntry> entries;
};

[[nodiscard]] AssetLibrary asset_library_create(std::filesystem::path const & assetsDir);

[[nodiscard]] char const * asset_library_imgui(
	AssetLibrary const & lib,
	std::unordered_map<std::string, u32> const & ratings
);
