#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <stb_image.h>


#include <meshoptimizer.h>

#include <mor/mor.hpp>
#include <srat/core-math.hpp>
#include <vkof/vkof.hpp>

#include <imgui.h>

#include <unordered_map>
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

	std::unordered_map<cgltf_texture const *, u64> textureHandles;
	std::vector<vkof::Image> images;
	std::vector<vkof::Sampler> samplers;

	std::unordered_map<cgltf_material const *, u32> materialIndices;

	std::vector<Material> materials;
};

struct ImplGpuScene {
	vkof::Buffer positions;
	vkof::Buffer attributes;
	vkof::Buffer meshletVerts;
	vkof::Buffer meshletTris;
	vkof::Buffer meshlets;
	vkof::Buffer instances;
	vkof::Buffer materials;
	vkof::Buffer textures;
	u32 meshletCount;
};

static constexpr u32 skMaxMeshletVerts = 64;
static constexpr u32 skMaxMeshletTris = 124;

} // namespace

// ----------------------------------------------------------------------------
// -- private helpers
// ----------------------------------------------------------------------------

static u64 load_texture(
	ImplScene & s,
	cgltf_texture const * tex,
	bool const srgb
) {
	if (!tex || !tex->image || !tex->image->buffer_view) return 0;

	auto const it = s.textureHandles.find(tex);
	if (it != s.textureHandles.end()) return it->second;

	cgltf_image const * img = tex->image;
	u8 const * encoded = (
		(u8 const *)img->buffer_view->buffer->data
		+ img->buffer_view->offset
	);
	int const encodedLen = (int)img->buffer_view->size;

	int w, h, channels;
	stbi_uc * const pixels = stbi_load_from_memory(
		encoded, encodedLen, &w, &h, &channels, 4
	);
	if (!pixels) return 0;

	vkof::Image const image = vkof::image_create({
		.width = (u32)w,
		.height = (u32)h,
		.format = (
			srgb
			? vkof::ImageFormat::r8g8b8a8_srgb
			: vkof::ImageFormat::r8g8b8a8_unorm
		),
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(
			pixels, (u64)w * h * 4
		),
	});
	stbi_image_free(pixels);

	auto const toFilter = [](int gl) -> vkof::SamplerFilter {
		if (gl == 9728 || gl == 9984 || gl == 9986) {
			return vkof::SamplerFilter::nearest;
		}
		return vkof::SamplerFilter::linear;
	};
	auto const toWrap = [](int gl) -> vkof::SamplerAddressMode {
		if (gl == 33071) return vkof::SamplerAddressMode::clamp_to_edge;
		if (gl == 33648) return vkof::SamplerAddressMode::mirrored_repeat;
		return vkof::SamplerAddressMode::repeat;
	};

	cgltf_sampler const * cgSamp = tex->sampler;
	vkof::Sampler const sampler = vkof::sampler_create({
		.magFilter = (
			cgSamp
			? toFilter(cgSamp->mag_filter)
			: vkof::SamplerFilter::linear
		),
		.minFilter = (
			cgSamp
			? toFilter(cgSamp->min_filter)
			: vkof::SamplerFilter::linear
		),
		.addressModeU = (
			cgSamp
			? toWrap(cgSamp->wrap_s)
			: vkof::SamplerAddressMode::repeat
		),
		.addressModeV = (
			cgSamp
			? toWrap(cgSamp->wrap_t)
			: vkof::SamplerAddressMode::repeat
		),
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});

	u64 const handle = vkof::image_sampler_handle({
		.image = image,
		.sampler = sampler,
	});

	s.images.push_back(image);
	s.samplers.push_back(sampler);
	s.textureHandles.emplace(tex, handle);
	return handle;
}

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

	// -- materials
	u32 materialIndex = 0;
	if (prim.material) {
		auto const it = s->materialIndices.find(prim.material);
		if (it != s->materialIndices.end()) {
			materialIndex = it->second;
		} else {
			materialIndex = (u32)s->materials.size();
			cgltf_material const & mat = *prim.material;
			cgltf_pbr_metallic_roughness const & mr = (
				mat.pbr_metallic_roughness
			);
			s->materials.push_back({
				.baseColor = {
					mr.base_color_factor[0],
					mr.base_color_factor[1],
					mr.base_color_factor[2],
					mr.base_color_factor[3],
				},
				.metallic = mr.metallic_factor,
				.roughness = mr.roughness_factor,
				.emissive = {
					mat.emissive_factor[0],
					mat.emissive_factor[1],
					mat.emissive_factor[2],
				},
				.textureBaseColor = load_texture(
					*s, mr.base_color_texture.texture, true
				),
				.textureNormal = load_texture(
					*s, mat.normal_texture.texture, false
				),
				.textureMetallicRoughness = load_texture(
					*s, mr.metallic_roughness_texture.texture, false
				),
				.textureEmissive = load_texture(
					*s, mat.emissive_texture.texture, true
				),
				.flags = 0,
			});
			s->materialIndices.emplace(prim.material, materialIndex);
		}
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
			.materialIndex = (u32)materialIndex,
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

mor::Scene mor::scene_create() {
	return mor::Scene { .id = reinterpret_cast<u64>(new ImplScene()) };
}

void mor::scene_destroy(mor::Scene const & scene) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	for (vkof::Image const & img : s->images) {
		vkof::image_destroy(img);
	}
	for (vkof::Sampler const & samp : s->samplers) {
		vkof::sampler_destroy(samp);
	}
	delete s;
}

u32 mor::scene_instance_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->instances.size();
}

u32 mor::scene_meshlet_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->meshlets.size();
}

u32 mor::scene_vertex_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->positions.size();
}

void mor::scene_imgui_debug(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);

	u32 const instanceCount = (u32)s->instances.size();
	u32 const meshletCount = (u32)s->meshlets.size();
	u32 const vertexCount = (u32)s->positions.size();
	u32 const materialCount = (u32)s->materials.size();
	u32 const textureCount = (u32)s->images.size();

	char label[64];

	snprintf(label, sizeof(label), "instances (%u)", instanceCount);
	if (ImGui::TreeNode(label)) {
		for (u32 i = 0; i < instanceCount; ++i) {
			Instance const & inst = s->instances[i];
			ImGui::Text(
				"[%u] meshlets: %u  offset: %u",
				i, inst.meshletCount, inst.meshletOffset
			);
		}
		ImGui::TreePop();
	}

	snprintf(label, sizeof(label), "meshlets (%u)", meshletCount);
	if (ImGui::TreeNode(label)) {
		for (u32 i = 0; i < meshletCount; ++i) {
			Meshlet const & m = s->meshlets[i];
			ImGui::Text(
				"[%u] verts: %u  tris: %u  inst: %u  mat: %u",
				i, m.vertexCount, m.triangleCount, m.instanceIndex, m.materialIndex
			);
		}
		ImGui::TreePop();
	}

	ImGui::Text("vertices:  %u", vertexCount);

	snprintf(label, sizeof(label), "materials (%u)", materialCount);
	if (ImGui::TreeNode(label)) {
		for (u32 i = 0; i < materialCount; ++i) {
			Material const & mat = s->materials[i];
			ImGui::Text(
				"[%u] base: (%.2f %.2f %.2f %.2f)  metal: %.2f  rough: %.2f",
				i,
				mat.baseColor.x, mat.baseColor.y,
				mat.baseColor.z, mat.baseColor.w,
				mat.metallic, mat.roughness
			);
		}
		ImGui::TreePop();
	}

	snprintf(label, sizeof(label), "textures (%u)", textureCount);
	if (ImGui::TreeNode(label)) {
		u32 i = 0u;
		for (auto const & [tex, handle] : s->textureHandles) {
			ImGui::Text("[%u] handle: 0x%llx", i, (unsigned long long)handle);
			++i;
		}
		ImGui::TreePop();
	}
}

void mor::scene_load_gltf(mor::Scene const & scene, char const * const path) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);

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

mor::GpuScene mor::scene_gpu_upload(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
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
	if (!s->materials.empty()) {
		gpu->materials = upload_buffer(
			s->materials.data(), s->materials.size() * sizeof(Material)
		);
	}
	{
		std::vector<u64> handles;
		handles.reserve(s->images.size());
		for (size_t i = 0; i < s->images.size(); ++i) {
			handles.emplace_back(vkof::image_sampler_handle({
				.image = s->images[i],
				.sampler = s->samplers[i],
			}));
		}
		if (!handles.empty()) {
			gpu->textures = upload_buffer(
				handles.data(), handles.size() * sizeof(u64)
			);
		}
	}
	gpu->meshletCount = (u32)s->meshlets.size();

	return mor::GpuScene { .id = reinterpret_cast<u64>(gpu) };
}

void mor::scene_gpu_destroy(mor::GpuScene const & scene) {
	ImplGpuScene * const gpu = reinterpret_cast<ImplGpuScene *>(scene.id);
	vkof::buffer_destroy(gpu->positions);
	vkof::buffer_destroy(gpu->attributes);
	vkof::buffer_destroy(gpu->meshletVerts);
	vkof::buffer_destroy(gpu->meshletTris);
	vkof::buffer_destroy(gpu->meshlets);
	vkof::buffer_destroy(gpu->instances);
	vkof::buffer_destroy(gpu->materials);
	vkof::buffer_destroy(gpu->textures);
	delete gpu;
}

mor::Buffers mor::scene_gpu_buffers(mor::GpuScene const & scene) {
	ImplGpuScene const * const gpu = reinterpret_cast<ImplGpuScene const *>(scene.id);
	return {
		.meshlets = vkof::buffer_virtual_address(gpu->meshlets),
		.materials = vkof::buffer_virtual_address(gpu->materials),
		.textures = vkof::buffer_virtual_address(gpu->textures),
		.instances = vkof::buffer_virtual_address(gpu->instances),
		.positions = vkof::buffer_virtual_address(gpu->positions),
		.attributes = vkof::buffer_virtual_address(gpu->attributes),
		.meshletVerts = vkof::buffer_virtual_address(gpu->meshletVerts),
		.meshletTris = vkof::buffer_virtual_address(gpu->meshletTris),
	};
}

u32 mor::scene_gpu_meshlet_count(mor::GpuScene const & scene) {
	return reinterpret_cast<ImplGpuScene const *>(scene.id)->meshletCount;
}
