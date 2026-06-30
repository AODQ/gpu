#include <doctest/doctest.h>
#include <scene/scene.hpp>

#include <filesystem>
#include <fstream>

TEST_SUITE("[headless]") {

TEST_CASE("scene: EntityInstance materials default to empty") {
	scene::EntityInstance const inst {
		.filename = "foo.gltf",
		.rotation = {},
		.scale = 1.0f,
		.materials = {},
	};
	CHECK(inst.materials.empty());
}

TEST_CASE("scene: SceneMaterial round-trips through JSON") {
	auto const dir = std::filesystem::temp_directory_path() / "scene_mat_test";
	std::filesystem::create_directories(dir);
	auto const path = dir / "single_mat.json";

	GpuMorMaterial params {};
	params.baseColor = { 0.3f, 0.5f, 0.7f, 1.0f };
	params.baseMetalness = 0.42f;
	params.specularRoughness = 0.15f;
	params.specularIor = 1.8f;
	params.coatWeight = 0.3f;
	params.coatRoughness = 0.25f;
	params.geometryOpacity = 0.9f;
	params.emissiveColor = { 0.1f, 0.2f, 0.05f };
	params.emissiveLuminance = 3.5f;
	params.flags = 4u;

	{
		scene::Desc desc;
		scene::EntityInstance inst {
			.filename = "test.gltf",
			.rotation = {},
			.scale = 1.0f,
			.materials = {},
		};
		inst.materials.push_back({ .name = "MyMat", .params = params });
		scene::desc_add_entity(desc, std::move(inst), {});
		scene::desc_save(desc, path);
	}

	scene::Desc const loaded = scene::desc_load(path);
	REQUIRE(loaded.entities.size() == 1u);
	auto const * ei = std::get_if<scene::EntityInstance>(&loaded.entities[0].data);
	REQUIRE(ei != nullptr);
	REQUIRE(ei->materials.size() == 1u);

	scene::SceneMaterial const & sm = ei->materials[0];
	CHECK(sm.name == "MyMat");
	CHECK(sm.params.baseColor.x == doctest::Approx(0.3f).epsilon(0.001f));
	CHECK(sm.params.baseColor.y == doctest::Approx(0.5f).epsilon(0.001f));
	CHECK(sm.params.baseColor.z == doctest::Approx(0.7f).epsilon(0.001f));
	CHECK(sm.params.baseColor.w == doctest::Approx(1.0f).epsilon(0.001f));
	CHECK(sm.params.baseMetalness == doctest::Approx(0.42f).epsilon(0.001f));
	CHECK(sm.params.specularRoughness == doctest::Approx(0.15f).epsilon(0.001f));
	CHECK(sm.params.specularIor == doctest::Approx(1.8f).epsilon(0.001f));
	CHECK(sm.params.coatWeight == doctest::Approx(0.3f).epsilon(0.001f));
	CHECK(sm.params.coatRoughness == doctest::Approx(0.25f).epsilon(0.001f));
	CHECK(sm.params.geometryOpacity == doctest::Approx(0.9f).epsilon(0.001f));
	CHECK(sm.params.emissiveColor.x == doctest::Approx(0.1f).epsilon(0.001f));
	CHECK(sm.params.emissiveLuminance == doctest::Approx(3.5f).epsilon(0.001f));
	CHECK(sm.params.flags == 4u);
	// texture handles must always be zero after loading from JSON
	CHECK(sm.params.textureBaseColor == 0u);
	CHECK(sm.params.textureNormal == 0u);
	CHECK(sm.params.textureClearcoat == 0u);
}

TEST_CASE("scene: multiple SceneMaterials per instance round-trip") {
	auto const dir = std::filesystem::temp_directory_path() / "scene_mat_test";
	std::filesystem::create_directories(dir);
	auto const path = dir / "multi_mat.json";

	{
		scene::Desc desc;
		scene::EntityInstance inst {
			.filename = "multi.gltf",
			.rotation = {},
			.scale = 2.0f,
			.materials = {},
		};
		GpuMorMaterial p0 {};
		p0.baseColor = { 1.0f, 0.0f, 0.0f, 1.0f };
		p0.baseMetalness = 0.0f;
		inst.materials.push_back({ .name = "", .params = p0 });

		GpuMorMaterial p1 {};
		p1.baseColor = { 0.0f, 1.0f, 0.0f, 1.0f };
		p1.baseMetalness = 1.0f;
		p1.specularRoughness = 0.05f;
		inst.materials.push_back({ .name = "Green", .params = p1 });

		scene::desc_add_entity(desc, std::move(inst), {});
		scene::desc_save(desc, path);
	}

	scene::Desc const loaded = scene::desc_load(path);
	REQUIRE(loaded.entities.size() == 1u);
	auto const * ei = std::get_if<scene::EntityInstance>(&loaded.entities[0].data);
	REQUIRE(ei != nullptr);
	REQUIRE(ei->materials.size() == 2u);

	CHECK(ei->materials[0].name == "");
	CHECK(ei->materials[0].params.baseColor.x == doctest::Approx(1.0f).epsilon(0.001f));
	CHECK(ei->materials[0].params.baseMetalness == doctest::Approx(0.0f).epsilon(0.001f));

	CHECK(ei->materials[1].name == "Green");
	CHECK(ei->materials[1].params.baseColor.y == doctest::Approx(1.0f).epsilon(0.001f));
	CHECK(ei->materials[1].params.baseMetalness == doctest::Approx(1.0f).epsilon(0.001f));
	CHECK(ei->materials[1].params.specularRoughness == doctest::Approx(0.05f).epsilon(0.001f));
}

TEST_CASE("scene: legacy JSON without materials key loads with empty materials") {
	auto const dir = std::filesystem::temp_directory_path() / "scene_mat_test";
	std::filesystem::create_directories(dir);
	auto const path = dir / "legacy.json";

	{
		std::ofstream f(path);
		f << R"({
  "entities": [
    {
      "type": "instance",
      "position": [0, 0, 0],
      "filename": "foo.gltf",
      "rotation": [0, 0, 0],
      "scale": 1.0
    }
  ]
})";
	}

	scene::Desc const loaded = scene::desc_load(path);
	REQUIRE(loaded.entities.size() == 1u);
	auto const * ei = std::get_if<scene::EntityInstance>(&loaded.entities[0].data);
	REQUIRE(ei != nullptr);
	CHECK(ei->materials.empty());
}

TEST_CASE("scene: materials survive second save/load cycle unchanged") {
	auto const dir = std::filesystem::temp_directory_path() / "scene_mat_test";
	std::filesystem::create_directories(dir);
	auto const path = dir / "cycle.json";

	GpuMorMaterial original {};
	original.baseColor = { 0.6f, 0.3f, 0.1f, 0.8f };
	original.transmissionWeight = 0.5f;
	original.thinFilmIor = 2.1f;
	original.subsurfaceRadius = 7.5f;

	{
		scene::Desc desc;
		scene::EntityInstance inst {
			.filename = "cycle.gltf",
			.rotation = {},
			.scale = 1.0f,
			.materials = {},
		};
		inst.materials.push_back({ .name = "Cycle", .params = original });
		scene::desc_add_entity(desc, std::move(inst), {});
		scene::desc_save(desc, path);
	}

	// Load, then save again (simulating an editor re-save)
	{
		scene::Desc reloaded = scene::desc_load(path);
		scene::desc_save(reloaded, path);
	}

	scene::Desc const final = scene::desc_load(path);
	REQUIRE(final.entities.size() == 1u);
	auto const * ei = std::get_if<scene::EntityInstance>(&final.entities[0].data);
	REQUIRE(ei != nullptr);
	REQUIRE(ei->materials.size() == 1u);

	CHECK(ei->materials[0].params.baseColor.x == doctest::Approx(0.6f).epsilon(0.001f));
	CHECK(ei->materials[0].params.transmissionWeight == doctest::Approx(0.5f).epsilon(0.001f));
	CHECK(ei->materials[0].params.thinFilmIor == doctest::Approx(2.1f).epsilon(0.001f));
	CHECK(ei->materials[0].params.subsurfaceRadius == doctest::Approx(7.5f).epsilon(0.001f));
}

} // TEST_SUITE("[headless]")
