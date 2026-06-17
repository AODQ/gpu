#include <doctest/doctest.h>
#include <mor/mor.hpp>
#include <vkof/vkof.hpp>
#include <srat/core-types.hpp>
#include "helpers.hpp"

#include <filesystem>
#include <fstream>
#include <string>

// Writes a minimal single-triangle glTF (3 verts, 1 tri) to a temp directory
// and returns the path to the .gltf file.
static std::string write_triangle_gltf() {
	std::filesystem::path const dir = (
		std::filesystem::temp_directory_path() / "mor_test"
	);
	std::filesystem::create_directories(dir);

	// positions: (-1,-1,0), (1,-1,0), (0,1,0)  —  3 × f32v3 = 36 bytes
	f32 const positions[9] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 0.0f,  1.0f, 0.0f,
	};
	// indices: 0, 1, 2  —  3 × u16 = 6 bytes
	u16 const indices[3] = { 0, 1, 2 };

	{
		std::ofstream f(dir / "triangle.bin", std::ios::binary);
		f.write(reinterpret_cast<char const *>(positions), sizeof(positions));
		f.write(reinterpret_cast<char const *>(indices),   sizeof(indices));
	}
	{
		std::ofstream f(dir / "triangle.gltf");
		f << R"({
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes":  [{ "mesh": 0 }],
  "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0] },
    { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset":  0, "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength":  6 }
  ],
  "buffers": [{ "byteLength": 42, "uri": "triangle.bin" }]
})";
	}

	return (dir / "triangle.gltf").string();
}

// Writes a glTF with two nodes referencing two separate meshes, both using
// the same triangle geometry. Verifies multi-mesh loading in a single file.
static std::string write_two_mesh_gltf() {
	std::filesystem::path const dir = (
		std::filesystem::temp_directory_path() / "mor_test"
	);
	std::filesystem::create_directories(dir);

	f32 const positions[9] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 0.0f,  1.0f, 0.0f,
	};
	u16 const indices[3] = { 0u, 1u, 2u };
	{
		std::ofstream f(dir / "two_mesh.bin", std::ios::binary);
		f.write(reinterpret_cast<char const *>(positions), sizeof(positions));
		f.write(reinterpret_cast<char const *>(indices), sizeof(indices));
	}
	{
		std::ofstream f(dir / "two_mesh.gltf");
		f << R"({
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0, 1] }],
  "nodes":  [{ "mesh": 0 }, { "mesh": 1 }],
  "meshes": [
    { "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] },
    { "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0] },
    { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset":  0, "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength":  6 }
  ],
  "buffers": [{ "byteLength": 42, "uri": "two_mesh.bin" }]
})";
	}
	return (dir / "two_mesh.gltf").string();
}

// Writes a glTF with a triangle and an explicit material
// (baseColor=0.8/0.2/0.4/1.0, metallic=0.1, roughness=0.6).
static std::string write_triangle_gltf_with_material() {
	std::filesystem::path const dir = (
		std::filesystem::temp_directory_path() / "mor_test"
	);
	std::filesystem::create_directories(dir);

	f32 const positions[9] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 0.0f,  1.0f, 0.0f,
	};
	u16 const indices[3] = { 0u, 1u, 2u };
	{
		std::ofstream f(dir / "material_tri.bin", std::ios::binary);
		f.write(reinterpret_cast<char const *>(positions), sizeof(positions));
		f.write(reinterpret_cast<char const *>(indices), sizeof(indices));
	}
	{
		std::ofstream f(dir / "material_tri.gltf");
		f << R"({
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes":  [{ "mesh": 0 }],
  "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 }] }],
  "materials": [{
    "pbrMetallicRoughness": {
      "baseColorFactor": [0.8, 0.2, 0.4, 1.0],
      "metallicFactor":  0.1,
      "roughnessFactor": 0.6
    }
  }],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0] },
    { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset":  0, "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength":  6 }
  ],
  "buffers": [{ "byteLength": 42, "uri": "material_tri.bin" }]
})";
	}
	return (dir / "material_tri.gltf").string();
}

TEST_SUITE("[headless]") {

TEST_CASE("mor: scene create/destroy") {
	mor::Scene const scene = mor::scene_create();
	CHECK(scene.id != 0);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: empty scene has zero counts") {
	mor::Scene const scene = mor::scene_create();
	CHECK(mor::scene_instance_count(scene) == 0);
	CHECK(mor::scene_meshlet_count(scene) == 0);
	CHECK(mor::scene_vertex_count(scene) == 0);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: single triangle — 1 instance") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	CHECK(mor::scene_instance_count(scene) == 1);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: single triangle — 1 meshlet") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	CHECK(mor::scene_meshlet_count(scene) == 1);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: single triangle — 3 vertices") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	CHECK(mor::scene_vertex_count(scene) == 3);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: multiple loads accumulate") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::scene_load_gltf(scene, path.c_str());
	CHECK(mor::scene_instance_count(scene) == 2);
	CHECK(mor::scene_meshlet_count(scene) == 2);
	CHECK(mor::scene_vertex_count(scene) == 6);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: gpu upload returns valid id") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::GpuScene const gpu = mor::scene_gpu_upload(scene);
	CHECK(gpu.id != 0);
	mor::scene_gpu_destroy(gpu);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: gpu meshlet count matches cpu") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	u32 const cpuCount = mor::scene_meshlet_count(scene);
	mor::GpuScene const gpu = mor::scene_gpu_upload(scene);
	CHECK(mor::scene_gpu_meshlet_count(gpu) == cpuCount);
	mor::scene_gpu_destroy(gpu);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: all gpu buffer addresses are non-zero") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::GpuScene const gpu = mor::scene_gpu_upload(scene);
	mor::Buffers const buf = mor::scene_gpu_buffers(gpu);
	CHECK(buf.meshlets != 0);
	CHECK(buf.materials != 0);
	CHECK(buf.instances != 0);
	CHECK(buf.positions != 0);
	CHECK(buf.attributes != 0);
	CHECK(buf.meshletVerts != 0);
	CHECK(buf.meshletTris != 0);
	mor::scene_gpu_destroy(gpu);
	mor::scene_destroy(scene);
}

TEST_CASE("mor: two gpu uploads have distinct buffer addresses") {
	std::string const path = write_triangle_gltf();
	mor::Scene const s1 = mor::scene_create();
	mor::Scene const s2 = mor::scene_create();
	mor::scene_load_gltf(s1, path.c_str());
	mor::scene_load_gltf(s2, path.c_str());
	mor::GpuScene const g1 = mor::scene_gpu_upload(s1);
	mor::GpuScene const g2 = mor::scene_gpu_upload(s2);
	CHECK(
		mor::scene_gpu_buffers(g1).meshlets
		!= mor::scene_gpu_buffers(g2).meshlets
	);
	mor::scene_gpu_destroy(g1);
	mor::scene_gpu_destroy(g2);
	mor::scene_destroy(s1);
	mor::scene_destroy(s2);
}

// 11. Two nodes in one glTF file load as 2 instances with correct counts
TEST_CASE("mor: two mesh nodes in one glTF — 2 instances") {
	std::string const path = write_two_mesh_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	CHECK(mor::scene_instance_count(scene) == 2u);
	CHECK(mor::scene_meshlet_count(scene) == 2u);
	CHECK(mor::scene_vertex_count(scene) == 6u);
	mor::scene_destroy(scene);
}

// 12. scene_set_anisotropy no crash and leaves GPU buffers valid
TEST_CASE("mor: scene_set_anisotropy no crash") {
	std::string const path = write_triangle_gltf();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::GpuScene const gpu = mor::scene_gpu_upload(scene);
	REQUIRE(gpu.id != 0);

	mor::scene_set_anisotropy(scene, gpu, 4.0f);
	test::gpu_wait();

	mor::Buffers const buf = mor::scene_gpu_buffers(gpu);
	CHECK(buf.materials != 0u);
	CHECK(buf.meshlets != 0u);

	mor::scene_gpu_destroy(gpu);
	mor::scene_destroy(scene);
}

// 13. Material baseColor/metallic/roughness round-trip via GPU readback
TEST_CASE("mor: material data readback") {
	std::string const path = write_triangle_gltf_with_material();
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::GpuScene const gpu = mor::scene_gpu_upload(scene);
	REQUIRE(gpu.id != 0);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "read_material.comp",
	});
	REQUIRE(pl.id != 0);

	auto outBuf = vkof::buffer_create({
		.byteCount = 6u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 materialsVa;
		u64 outVa;
		u32 materialIndex;
	};
	Push const push {
		.materialsVa = mor::scene_gpu_buffers(gpu).materials,
		.outVa = vkof::buffer_virtual_address(outBuf),
		.materialIndex = 1u,
	};
	test::dispatch(pl, push, /*groupCountX=*/1u);

	auto const result = test::readback<f32>(outBuf, 0u, 6u);
	CHECK(result[0] == doctest::Approx(0.8f).epsilon(0.001f)); // baseColor.r
	CHECK(result[1] == doctest::Approx(0.2f).epsilon(0.001f)); // baseColor.g
	CHECK(result[2] == doctest::Approx(0.4f).epsilon(0.001f)); // baseColor.b
	CHECK(result[3] == doctest::Approx(1.0f).epsilon(0.001f)); // baseColor.a
	CHECK(result[4] == doctest::Approx(0.1f).epsilon(0.001f)); // metallic
	CHECK(result[5] == doctest::Approx(0.6f).epsilon(0.001f)); // roughness

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	mor::scene_gpu_destroy(gpu);
	mor::scene_destroy(scene);
}


} // TEST_SUITE("[headless]")
