#include "asset_library.hpp"

#include <imgui.h>

static std::filesystem::path find_gltf(std::filesystem::path const & modelDir)
{
	static char const * const kSubdirs[] = {
		"glTF", "GLB", "glTF-Binary", "glTF-Embedded",
	};
	for (char const * subdir : kSubdirs) {
		std::filesystem::path const dir = modelDir / subdir;
		if (!std::filesystem::is_directory(dir)) { continue; }
		for (auto const & entry : std::filesystem::directory_iterator(dir)) {
			std::string const ext = entry.path().extension().string();
			if (ext == ".gltf" || ext == ".glb") {
				return entry.path();
			}
		}
	}
	return {};
}

AssetLibrary asset_library_create(std::filesystem::path const & assetsDir)
{
	AssetLibrary lib;
	std::filesystem::path const modelsDir = assetsDir / "Models";
	if (!std::filesystem::is_directory(modelsDir)) { return lib; }
	for (auto const & entry : std::filesystem::directory_iterator(modelsDir)) {
		if (!entry.is_directory()) { continue; }
		std::filesystem::path const gltf = find_gltf(entry.path());
		if (gltf.empty()) { continue; }
		lib.entries.push_back({
			.name = entry.path().filename().string(),
			.path = gltf,
		});
	}
	std::sort(lib.entries.begin(), lib.entries.end(), [](
		AssetEntry const & a, AssetEntry const & b
	) {
		return a.name < b.name;
	});
	return lib;
}

char const * asset_library_imgui(AssetLibrary const & lib)
{
	char const * selected = nullptr;
	ImGui::Begin("Assets");
	for (int i = 0; i < (int)lib.entries.size(); ++i) {
		if (ImGui::Button(lib.entries[i].name.c_str(), ImVec2(-1.0f, 0.0f))) {
			selected = lib.entries[i].path.c_str();
		}
	}
	ImGui::End();
	return selected;
}
