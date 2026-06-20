#include <doctest/doctest.h>
#include <mor/mor.hpp>
#include <srat/core-types.hpp>

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

} // TEST_SUITE("[headless]")
