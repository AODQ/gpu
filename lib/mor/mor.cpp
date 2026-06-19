#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <meshoptimizer.h>

#include <mor/mor.hpp>
#include <srat/core-math.hpp>
#include <vkof/vkof.hpp>

#include <vector>

// ----------------------------------------------------------------------------
// -- private types
// ----------------------------------------------------------------------------

namespace {

struct ImplScene {
	std::vector<f32v3> positions;
	std::vector<VertexAttr> attributes;
	std::vector<u32> meshletVerts;
	std::vector<u8> meshletTris;
	std::vector<Meshlet> meshlets;
	std::vector<Instance> instances;
};

struct ImplGpuScene {
	vkof::Buffer positions;
	vkof::Buffer attributes;
	vkof::Buffer meshletVerts;
	vkof::Buffer meshletTris;
	vkof::Buffer meshlets;
	vkof::Buffer instances;
	u32 meshletCount;
};

static constexpr u32 skMaxMeshletVerts = 64;
static constexpr u32 skMaxMeshletTris = 124;

} // namespace

// ----------------------------------------------------------------------------
// -- private helpers
// ----------------------------------------------------------------------------

static void load_primitive(
	ImplScene * s,
	cgltf_primitive const & prim,
	u32 const instanceIndex
) {
	if (prim.type != cgltf_primitive_type_triangles || !prim.indices) return;

	cgltf_accessor * posAcc = nullptr;
	cgltf_accessor * normAcc = nullptr;
	cgltf_accessor * uvAcc = nullptr;
	for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
		cgltf_attribute const & attr = prim.attributes[ai];
		switch (attr.type) {
			case cgltf_attribute_type_position: posAcc = attr.data; break;
			case cgltf_attribute_type_normal: normAcc = attr.data; break;
			case cgltf_attribute_type_texcoord:
				if (attr.index == 0) uvAcc = attr.data;
				break;
			default: break;
		}
	}
	if (!posAcc) return;

	u32 const vertexBase = (u32)s->positions.size();
	u32 const vertexCount = (u32)posAcc->count;
	u32 const indexCount = (u32)prim.indices->count;

	// -- positions
	{
		std::vector<f32> tmp(vertexCount * 3);
		cgltf_accessor_unpack_floats(posAcc, tmp.data(), tmp.size());
		for (u32 i = 0; i < vertexCount; ++i) {
			s->positions.push_back({ tmp[i*3+0], tmp[i*3+1], tmp[i*3+2] });
		}
	}

	// -- normals + uvs
	{
		std::vector<f32> normals(vertexCount * 3, 0.0f);
		std::vector<f32> uvs(vertexCount * 2, 0.0f);
		if (normAcc) {
			cgltf_accessor_unpack_floats(normAcc, normals.data(), normals.size());
		}
		if (uvAcc) {
			cgltf_accessor_unpack_floats(uvAcc, uvs.data(), uvs.size());
		}
		for (u32 i = 0; i < vertexCount; ++i) {
			s->attributes.push_back({
				.normal = { normals[i*3+0], normals[i*3+1], normals[i*3+2] },
				.uv = { uvs[i*2+0], uvs[i*2+1] },
			});
		}
	}

	// -- indices
	std::vector<u32> indices(indexCount);
	for (u32 i = 0; i < indexCount; ++i) {
		indices[i] = (u32)cgltf_accessor_read_index(prim.indices, i);
	}

	// -- meshlets
	size_t const maxMeshlets = meshopt_buildMeshletsBound(
		indexCount, skMaxMeshletVerts, skMaxMeshletTris
	);
	std::vector<meshopt_Meshlet> optMeshlets(maxMeshlets);
	std::vector<u32> optVerts(maxMeshlets * skMaxMeshletVerts);
	std::vector<u8> optTris(maxMeshlets * skMaxMeshletTris * 3);

	u32 const primMeshletCount = (u32)meshopt_buildMeshlets(
		optMeshlets.data(), optVerts.data(), optTris.data(),
		indices.data(), indexCount,
		reinterpret_cast<f32 const *>(s->positions.data() + vertexBase),
		vertexCount, sizeof(f32v3),
		skMaxMeshletVerts, skMaxMeshletTris, 0.0f
	);

	// trim to actual used extents
	meshopt_Meshlet const & last = optMeshlets[primMeshletCount - 1];
	optVerts.resize(last.vertex_offset + last.vertex_count);
	optTris.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));

	u32 const vertOffsetBase = (u32)s->meshletVerts.size();
	u32 const triOffsetBase = (u32)s->meshletTris.size();

	for (u32 const v : optVerts) {
		s->meshletVerts.push_back(vertexBase + v);
	}
	for (u8 const t : optTris) {
		s->meshletTris.push_back(t);
	}

	for (u32 mi = 0; mi < primMeshletCount; ++mi) {
		meshopt_Meshlet const & m = optMeshlets[mi];
		s->meshlets.push_back({
			.vertexOffset = vertOffsetBase + m.vertex_offset,
			.vertexCount = m.vertex_count,
			.triangleOffset = triOffsetBase + m.triangle_offset,
			.triangleCount = m.triangle_count,
			.instanceIndex = instanceIndex,
		});
	}
}

static void load_node(ImplScene * s, cgltf_node const * node) {
	if (node->mesh) {
		u32 const instanceIndex = (u32)s->instances.size();
		u32 const meshletOffset = (u32)s->meshlets.size();

		for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
			load_primitive(s, node->mesh->primitives[pi], instanceIndex);
		}

		f32m44 transform {};
		cgltf_node_transform_world(node, transform.m.ptr());

		s->instances.push_back({
			.transform = transform,
			.meshletOffset = meshletOffset,
			.meshletCount = (u32)s->meshlets.size() - meshletOffset,
		});
	}

	for (cgltf_size ci = 0; ci < node->children_count; ++ci) {
		load_node(s, node->children[ci]);
	}
}

static vkof::Buffer upload_buffer(void const * data, u64 byteCount) {
	SRAT_ASSERT_ALWAYS(byteCount > 0);
	vkof::Buffer const buf = vkof::buffer_create({
		.byteCount = byteCount,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(data), byteCount
		),
	});
	return buf;
}

// ----------------------------------------------------------------------------
// -- public API
// ----------------------------------------------------------------------------

mor::Scene * mor::scene_create() {
	return reinterpret_cast<mor::Scene *>(new ImplScene());
}

void mor::scene_destroy(mor::Scene * scene) {
	delete reinterpret_cast<ImplScene *>(scene);
}

u32 mor::scene_instance_count(mor::Scene const * scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene)->instances.size();
}

u32 mor::scene_meshlet_count(mor::Scene const * scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene)->meshlets.size();
}

u32 mor::scene_vertex_count(mor::Scene const * scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene)->positions.size();
}

void mor::scene_load_gltf(mor::Scene * scene, char const * path) {
	ImplScene * s = reinterpret_cast<ImplScene *>(scene);

	cgltf_options const options {};
	cgltf_data * data = nullptr;
	SRAT_ASSERT_ALWAYS(
		cgltf_parse_file(&options, path, &data) == cgltf_result_success
	);
	SRAT_ASSERT_ALWAYS(
		cgltf_load_buffers(&options, data, path) == cgltf_result_success
	);

	for (cgltf_size si = 0; si < data->scenes_count; ++si) {
		cgltf_scene const & gltfScene = data->scenes[si];
		for (cgltf_size ni = 0; ni < gltfScene.nodes_count; ++ni) {
			load_node(s, gltfScene.nodes[ni]);
		}
	}

	cgltf_free(data);
}

mor::GpuScene mor::scene_gpu_upload(mor::Scene const * scene) {
	ImplScene const * s = reinterpret_cast<ImplScene const *>(scene);
	ImplGpuScene * gpu = new ImplGpuScene();

	gpu->positions = upload_buffer(
		s->positions.data(), s->positions.size() * sizeof(f32v3)
	);
	gpu->attributes = upload_buffer(
		s->attributes.data(), s->attributes.size() * sizeof(VertexAttr)
	);
	gpu->meshletVerts = upload_buffer(
		s->meshletVerts.data(), s->meshletVerts.size() * sizeof(u32)
	);
	gpu->meshletTris = upload_buffer(
		s->meshletTris.data(), s->meshletTris.size() * sizeof(u8)
	);
	gpu->meshlets = upload_buffer(
		s->meshlets.data(), s->meshlets.size() * sizeof(Meshlet)
	);
	gpu->instances = upload_buffer(
		s->instances.data(), s->instances.size() * sizeof(Instance)
	);
	gpu->meshletCount = (u32)s->meshlets.size();

	return mor::GpuScene { .id = reinterpret_cast<u64>(gpu) };
}

void mor::scene_gpu_destroy(mor::GpuScene scene) {
	ImplGpuScene * gpu = reinterpret_cast<ImplGpuScene *>(scene.id);
	vkof::buffer_destroy(gpu->positions);
	vkof::buffer_destroy(gpu->attributes);
	vkof::buffer_destroy(gpu->meshletVerts);
	vkof::buffer_destroy(gpu->meshletTris);
	vkof::buffer_destroy(gpu->meshlets);
	vkof::buffer_destroy(gpu->instances);
	delete gpu;
}

mor::Buffers mor::scene_gpu_buffers(mor::GpuScene scene) {
	ImplGpuScene const * gpu = reinterpret_cast<ImplGpuScene const *>(scene.id);
	return {
		.meshlets = vkof::buffer_virtual_address(gpu->meshlets),
		.instances = vkof::buffer_virtual_address(gpu->instances),
		.positions = vkof::buffer_virtual_address(gpu->positions),
		.attributes = vkof::buffer_virtual_address(gpu->attributes),
		.meshletVerts = vkof::buffer_virtual_address(gpu->meshletVerts),
		.meshletTris = vkof::buffer_virtual_address(gpu->meshletTris),
	};
}

u32 mor::scene_gpu_meshlet_count(mor::GpuScene scene) {
	return reinterpret_cast<ImplGpuScene const *>(scene.id)->meshletCount;
}
