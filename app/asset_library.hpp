#pragma once

#include <filesystem>
#include <string>
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

[[nodiscard]] char const * asset_library_imgui(AssetLibrary const & lib);
