#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <stb_image.h>


#include <meshoptimizer.h>

#include <mor/mor.hpp>
#include <srat/core-math.hpp>
#include <vkof/vkof.hpp>

#include <imgui.h>

#include <cfloat>
#include <filesystem>
#include <unordered_map>
#include <vector>

// ----------------------------------------------------------------------------
// -- private types
// ----------------------------------------------------------------------------

namespace {

struct SamplerKey
{
	vkof::SamplerFilter magFilter;
	vkof::SamplerFilter minFilter;
	vkof::SamplerAddressMode addressU;
	vkof::SamplerAddressMode addressV;
	vkof::SamplerMipmapMode mipmapMode;
	f32 maxAnisotropy;
	bool operator==(SamplerKey const & o) const {
		return (
			magFilter == o.magFilter
			&& minFilter == o.minFilter
			&& addressU == o.addressU
			&& addressV == o.addressV
			&& mipmapMode == o.mipmapMode
			&& maxAnisotropy == o.maxAnisotropy
		);
	}
};

struct SamplerKeyHash
{
	size_t operator()(SamplerKey const & k) const {
		size_t h = 0;
		auto mix = [&](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
		mix((size_t)k.magFilter);
		mix((size_t)k.minFilter);
		mix((size_t)k.addressU);
		mix((size_t)k.addressV);
		mix((size_t)k.mipmapMode);
		mix((size_t)(k.maxAnisotropy * 4.0f));
		return h;
	}
};

static std::unordered_map<SamplerKey, vkof::Sampler, SamplerKeyHash> sSamplerCache;
static vkof::Sampler sImguiDisplaySampler = { 0u };

struct ImplScene {
	std::vector<f32v3> positions;
	std::vector<GpuMorVertexAttribute> attributes;
	std::vector<u32> meshletVerts;
	std::vector<u8> meshletTris;
	std::vector<GpuMorMeshlet> meshlets;
	std::vector<GpuMorInstance> instances;

	std::string gltfDir;
	std::unordered_map<cgltf_texture const *, u32> textureHandles;
	std::unordered_map<cgltf_texture const *, vkof::Image> textureImages;
	std::unordered_map<cgltf_texture const *, SamplerKey> textureSamplerKeys;
	std::vector<vkof::Image> images;
	std::vector<std::string> imageNames;
	std::vector<ImTextureID> imguiIds;

	std::unordered_map<cgltf_material const *, u32> materialIndices;

	std::vector<GpuMorMaterial> materials;
	std::vector<std::string> materialNames;
};

struct ImplGpuMaterials
{
	vkof::Buffer buffer;
	std::vector<GpuMorMaterial> cpu;
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
	vkof::Buffer flatIndices;
	vkof::Buffer flatMeshlets;
	u32 meshletCount;
	u32 vertexCount;
	u32 triangleCount;
};

static constexpr u32 skMaxMeshletVerts = 64;
static constexpr u32 skMaxMeshletTris = 124;

} // namespace

// ----------------------------------------------------------------------------
// -- private helpers
// ----------------------------------------------------------------------------

static u32 load_texture(
	ImplScene & s,
	cgltf_texture const * tex,
	bool const srgb
) {
	if (!tex || !tex->image) { return 0; }

	auto const it = s.textureHandles.find(tex);
	if (it != s.textureHandles.end()) { return it->second; }

	cgltf_image const * img = tex->image;
	int w, h, channels;
	stbi_uc * pixels = nullptr;

	if (img->buffer_view) {
		u8 const * encoded = (
			(u8 const *)img->buffer_view->buffer->data
			+ img->buffer_view->offset
		);
		int const encodedLen = (int)img->buffer_view->size;
		pixels = stbi_load_from_memory(encoded, encodedLen, &w, &h, &channels, 4);
	} else if (img->uri && strncmp(img->uri, "data:", 5) != 0) {
		std::string const fullPath = (std::filesystem::path(s.gltfDir) / img->uri).string();
		pixels = stbi_load(fullPath.c_str(), &w, &h, &channels, 4);
	}

	if (!pixels) { return 0; }

	u32 const mipLevels = (
		(u32)std::floor(std::log2((f32)std::max(w, h))) + 1u
	);
	vkof::Image const image = vkof::image_create({
		.width = (u32)w,
		.height = (u32)h,
		.format = (
			srgb
			? vkof::ImageFormat::r8g8b8a8_srgb
			: vkof::ImageFormat::r8g8b8a8_unorm
		),
		.mipLevels = mipLevels,
		.optInitialData = srat::slice<u8 const>(
			pixels, (u64)w * h * 4
		),
	});
	stbi_image_free(pixels);
	vkof::image_generate_mipmaps(image);

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
	SamplerKey const key {
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
		.addressU = (
			cgSamp
			? toWrap(cgSamp->wrap_s)
			: vkof::SamplerAddressMode::repeat
		),
		.addressV = (
			cgSamp
			? toWrap(cgSamp->wrap_t)
			: vkof::SamplerAddressMode::repeat
		),
		.mipmapMode = vkof::SamplerMipmapMode::linear,
		.maxAnisotropy = 16.0f,
	};
	auto cacheIt = sSamplerCache.find(key);
	if (cacheIt == sSamplerCache.end()) {
		vkof::Sampler const s = vkof::sampler_create({
			.magFilter = key.magFilter,
			.minFilter = key.minFilter,
			.addressModeU = key.addressU,
			.addressModeV = key.addressV,
			.addressModeW = vkof::SamplerAddressMode::repeat,
			.mipmapMode = key.mipmapMode,
			.maxAnisotropy = key.maxAnisotropy,
		});
		cacheIt = sSamplerCache.emplace(key, s).first;
	}
	vkof::Sampler const sampler = cacheIt->second;

	u32 const handle = vkof::image_sampler_handle({
		.image = image,
		.sampler = sampler,
	});

	s.images.push_back(image);
	{
		std::string name;
		if (tex->name && tex->name[0] != '\0') {
			name = tex->name;
		} else if (img->name && img->name[0] != '\0') {
			name = img->name;
		} else if (img->uri) {
			name = std::filesystem::path(img->uri).filename().string();
		} else {
			name = "[texture " + std::to_string(s.imageNames.size()) + "]";
		}
		s.imageNames.push_back(std::move(name));
	}
	s.textureHandles.emplace(tex, handle);
	s.textureImages.emplace(tex, image);
	s.textureSamplerKeys.emplace(tex, key);
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
	cgltf_accessor * tanAcc = nullptr;
	for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
		cgltf_attribute const & attr = prim.attributes[ai];
		switch (attr.type) {
			case cgltf_attribute_type_position: posAcc = attr.data; break;
			case cgltf_attribute_type_normal: normAcc = attr.data; break;
			case cgltf_attribute_type_texcoord:
				if (attr.index == 0) uvAcc = attr.data;
				break;
			case cgltf_attribute_type_tangent: tanAcc = attr.data; break;
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

	// -- normals, uvs, tangents
	{
		std::vector<f32> normals(vertexCount * 3, 0.0f);
		std::vector<f32> uvs(vertexCount * 2, 0.0f);
		std::vector<f32> tangents(vertexCount * 4, 0.0f);
		if (normAcc) {
			cgltf_accessor_unpack_floats(normAcc, normals.data(), normals.size());
		}
		if (uvAcc) {
			cgltf_accessor_unpack_floats(uvAcc, uvs.data(), uvs.size());
		}
		if (tanAcc) {
			cgltf_accessor_unpack_floats(tanAcc, tangents.data(), tangents.size());
		} else {
			for (u32 i = 0; i < vertexCount; ++i) {
				tangents[i*4+0] = 1.0f;
				tangents[i*4+3] = 1.0f;
			}
		}
		for (u32 i = 0; i < vertexCount; ++i) {
			s->attributes.push_back({
				.normal = { normals[i*3+0], normals[i*3+1], normals[i*3+2] },
				.uv = { uvs[i*2+0], uvs[i*2+1] },
				.tangent = {
					tangents[i*4+0], tangents[i*4+1],
					tangents[i*4+2], tangents[i*4+3],
				},
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
			cgltf_pbr_metallic_roughness const & mr = mat.pbr_metallic_roughness;

			f32v3 specularColor = { 1.0f, 1.0f, 1.0f };
			f32 specularWeight = 1.0f;
			u32 textureSpecular = 0u;
			u32 textureSpecularColor = 0u;
			if (mat.has_specular) {
				specularColor = {
					mat.specular.specular_color_factor[0],
					mat.specular.specular_color_factor[1],
					mat.specular.specular_color_factor[2],
				};
				specularWeight = mat.specular.specular_factor;
				textureSpecular = load_texture(
					*s, mat.specular.specular_texture.texture, false
				);
				textureSpecularColor = load_texture(
					*s, mat.specular.specular_color_texture.texture, true
				);
			}

			f32 specularIor = 1.5f;
			if (mat.has_ior) {
				specularIor = mat.ior.ior;
			}

			f32 coatWeight = 0.0f;
			f32 coatRoughness = 0.0f;
			u32 textureClearcoat = 0u;
			u32 textureClearcoatRoughness = 0u;
			if (mat.has_clearcoat) {
				coatWeight = mat.clearcoat.clearcoat_factor;
				coatRoughness = mat.clearcoat.clearcoat_roughness_factor;
				textureClearcoat = load_texture(
					*s, mat.clearcoat.clearcoat_texture.texture, false
				);
				textureClearcoatRoughness = load_texture(
					*s, mat.clearcoat.clearcoat_roughness_texture.texture, false
				);
			}

			f32v3 fuzzColor = { 1.0f, 1.0f, 1.0f };
			f32 fuzzRoughness = 0.5f;
			f32 fuzzWeight = 0.0f;
			u32 textureFuzz = 0u;
			if (mat.has_sheen) {
				fuzzColor = {
					mat.sheen.sheen_color_factor[0],
					mat.sheen.sheen_color_factor[1],
					mat.sheen.sheen_color_factor[2],
				};
				fuzzWeight = 1.0f;
				fuzzRoughness = mat.sheen.sheen_roughness_factor;
				textureFuzz = load_texture(
					*s, mat.sheen.sheen_color_texture.texture, true
				);
			}

			f32 subsurfaceWeight = 0.0f;
			f32v3 subsurfaceColor = { 0.8f, 0.8f, 0.8f };
			f32 subsurfaceRadius = 1.0f;
			u32 textureSubsurface = 0u;
			f32v3 transmissionColor = { 1.0f, 1.0f, 1.0f };
			f32 transmissionDepth = 0.0f;
			if (mat.has_volume) {
				subsurfaceColor = {
					mat.volume.attenuation_color[0],
					mat.volume.attenuation_color[1],
					mat.volume.attenuation_color[2],
				};
				textureSubsurface = load_texture(
					*s, mat.volume.thickness_texture.texture, false
				);
				transmissionColor = subsurfaceColor;
				// attenuation_distance defaults to +Infinity in glTF when unset;
				// treat that as 0 (no Beer-Lambert absorption)
				f32 const attenuationDistance = mat.volume.attenuation_distance;
				transmissionDepth = std::isinf(attenuationDistance) ? 0.0f : attenuationDistance;
			}

			f32 transmissionWeight = 0.0f;
			u32 textureTransmission = 0u;
			if (mat.has_transmission) {
				transmissionWeight = mat.transmission.transmission_factor;
				textureTransmission = load_texture(
					*s, mat.transmission.transmission_texture.texture, false
				);
				// per KHR_materials_transmission spec: base color tints transmitted
				// light; use it when volume provides no separate attenuation color
				if (!mat.has_volume) {
					transmissionColor = {
						mr.base_color_factor[0],
						mr.base_color_factor[1],
						mr.base_color_factor[2],
					};
				}
			}

			f32 thinFilmWeight = 0.0f;
			f32 thinFilmIor = 1.4f;
			f32 thinFilmThickness = 0.5f;
			f32 thinFilmThicknessMin = 100.0f;
			u32 textureIridescence = 0u;
			u32 textureIridescenceThickness = 0u;
			if (mat.has_iridescence) {
				thinFilmWeight = mat.iridescence.iridescence_factor;
				thinFilmIor = mat.iridescence.iridescence_ior;
				thinFilmThickness = mat.iridescence.iridescence_thickness_max;
				thinFilmThicknessMin = mat.iridescence.iridescence_thickness_min;
				textureIridescence = load_texture(
					*s, mat.iridescence.iridescence_texture.texture, false
				);
				textureIridescenceThickness = load_texture(
					*s, mat.iridescence.iridescence_thickness_texture.texture, false
				);
			}

			f32 specularRoughnessAnisotropy = 0.0f;
			f32 specularAnisotropyRotation = 0.0f;
			u32 textureAnisotropy = 0u;
			if (mat.has_anisotropy) {
				specularRoughnessAnisotropy = mat.anisotropy.anisotropy_strength;
				specularAnisotropyRotation = mat.anisotropy.anisotropy_rotation;
				textureAnisotropy = load_texture(
					*s, mat.anisotropy.anisotropy_texture.texture, false
				);
			}

			u32 const textureClearcoatNormal = (
				mat.has_clearcoat
				? load_texture(
					*s, mat.clearcoat.clearcoat_normal_texture.texture, false)
				: 0u
			);
			u32 const textureFuzzRoughness = (
				mat.has_sheen
				? load_texture(*s, mat.sheen.sheen_roughness_texture.texture, false)
				: 0u
			);
			u32 const textureOcclusion = load_texture(
				*s, mat.occlusion_texture.texture, false
			);

			u32 flags = (u32)mat.alpha_mode;
			if (mat.double_sided) { flags |= 0x4u; }
			if (mat.unlit) { flags |= 0x8u; }

			f32 transmissionDispersionAbbeNumber = 20.0f;
			if (mat.has_dispersion && mat.dispersion.dispersion > 0.0f) {
				transmissionDispersionAbbeNumber = 20.0f / mat.dispersion.dispersion;
			}

			f32 emissiveLuminance = 1.0f;
			if (mat.has_emissive_strength) {
				emissiveLuminance = mat.emissive_strength.emissive_strength;
			}

			s->materials.push_back({
				.baseColor = {
					mr.base_color_factor[0],
					mr.base_color_factor[1],
					mr.base_color_factor[2],
					mr.base_color_factor[3],
				},
				.baseMetalness = mr.metallic_factor,
				.specularRoughness = mr.roughness_factor,
				.emissiveColor = {
					mat.emissive_factor[0],
					mat.emissive_factor[1],
					mat.emissive_factor[2],
				},
				.emissiveLuminance = emissiveLuminance,
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
				.specularColor = specularColor,
				.specularWeight = specularWeight,
				.textureSpecular = textureSpecular,
				.textureSpecularColor = textureSpecularColor,
				.specularIor = specularIor,
				.coatWeight = coatWeight,
				.coatRoughness = coatRoughness,
				.textureClearcoat = textureClearcoat,
				.textureClearcoatRoughness = textureClearcoatRoughness,
				.fuzzColor = fuzzColor,
				.fuzzWeight = fuzzWeight,
				.fuzzRoughness = fuzzRoughness,
				.textureFuzz = textureFuzz,
				.subsurfaceWeight = subsurfaceWeight,
				.subsurfaceColor = subsurfaceColor,
				.subsurfaceRadius = subsurfaceRadius,
				.textureSubsurface = textureSubsurface,
				.textureSubsurfaceColor = 0u,
				.transmissionWeight = transmissionWeight,
				.textureTransmission = textureTransmission,
				.textureTransmissionColor = 0u,
				.transmissionColor = transmissionColor,
				.transmissionDepth = transmissionDepth,
				.thinFilmWeight = thinFilmWeight,
				.thinFilmIor = thinFilmIor,
				.thinFilmThickness = thinFilmThickness,
				.thinFilmThicknessMin = thinFilmThicknessMin,
				.textureIridescence = textureIridescence,
				.textureIridescenceThickness = textureIridescenceThickness,
				.specularRoughnessAnisotropy = specularRoughnessAnisotropy,
				.specularAnisotropyRotation = specularAnisotropyRotation,
				.textureAnisotropy = textureAnisotropy,
				.textureClearcoatNormal = textureClearcoatNormal,
				.textureFuzzRoughness = textureFuzzRoughness,
				.textureOcclusion = textureOcclusion,
				.transmissionDispersionAbbeNumber = transmissionDispersionAbbeNumber,
				.alphaCutoff = mat.alpha_cutoff,
				.geometryOpacity = 1.0f,
				.flags = flags,
			});
			s->materialIndices.emplace(prim.material, materialIndex);
			s->materialNames.push_back(
				(mat.name && mat.name[0]) ? std::string(mat.name) : ""
			);
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
		u32 const vertexBase = (u32)s->positions.size();

		for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
			load_primitive(s, node->mesh->primitives[pi], instanceIndex);
		}

		f32m44 transform {};
		cgltf_node_transform_world(node, transform.m.ptr());

		// bake node world transform into positions and attributes so the BLAS
		// lives in GLTF-world space; instance.transform is then identity and
		// the TLAS only needs the scene-level make_model_matrix.
		u32 const vertexEnd = (u32)s->positions.size();
		for (u32 i = vertexBase; i < vertexEnd; ++i) {
			f32v3 const p = s->positions[i];
			f32v4 const tp = transform * f32v4 { p.x, p.y, p.z, 1.0f };
			s->positions[i] = { tp.x, tp.y, tp.z };

			// normals and tangent.xyz transform by the 3x3 rotation sub-matrix
			// (assumes no non-uniform scale in node transforms)
			f32 const * m = transform.m.ptr();
			auto const rot3 = [&](f32v3 const v) -> f32v3 {
				return {
					m[0]*v.x + m[4]*v.y + m[8]*v.z,
					m[1]*v.x + m[5]*v.y + m[9]*v.z,
					m[2]*v.x + m[6]*v.y + m[10]*v.z,
				};
			};
			auto const norm = [](f32v3 const v) -> f32v3 {
				f32 const inv = 1.0f / sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
				return { v.x*inv, v.y*inv, v.z*inv };
			};

			GpuMorVertexAttribute & attr = s->attributes[i];
			attr.normal = norm(rot3(attr.normal));
			f32v3 const rt = norm(rot3({ attr.tangent.x, attr.tangent.y, attr.tangent.z }));
			attr.tangent = { rt.x, rt.y, rt.z, attr.tangent.w };
		}

		s->instances.push_back({
			.transform = f32m44_identity(),
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
	ImplScene * const s = new ImplScene();
	// index 0 is always the default material; meshlets with no material use it
	s->materials.push_back({
		.baseColor = { 1.0f, 1.0f, 1.0f, 1.0f },
		.baseMetalness = 0.0f,
		.specularRoughness = 1.0f,
		.emissiveColor = { 0.0f, 0.0f, 0.0f },
		.emissiveLuminance = 1.0f,
		.textureBaseColor = 0u,
		.textureNormal = 0u,
		.textureMetallicRoughness = 0u,
		.textureEmissive = 0u,
		.specularColor = { 1.0f, 1.0f, 1.0f },
		.specularWeight = 1.0f,
		.textureSpecular = 0u,
		.textureSpecularColor = 0u,
		.specularIor = 1.5f,
		.coatWeight = 0.0f,
		.coatRoughness = 0.0f,
		.textureClearcoat = 0u,
		.textureClearcoatRoughness = 0u,
		.fuzzColor = { 1.0f, 1.0f, 1.0f },
		.fuzzWeight = 0.0f,
		.fuzzRoughness = 0.0f,
		.textureFuzz = 0u,
		.subsurfaceWeight = 0.0f,
		.subsurfaceColor = { 0.8f, 0.8f, 0.8f },
		.subsurfaceRadius = 1.0f,
		.textureSubsurface = 0u,
		.textureSubsurfaceColor = 0u,
		.transmissionWeight = 0.0f,
		.textureTransmission = 0u,
		.textureTransmissionColor = 0u,
		.transmissionColor = { 1.0f, 1.0f, 1.0f },
		.transmissionDepth = 0.0f,
		.thinFilmWeight = 0.0f,
		.thinFilmIor = 1.4f,
		.thinFilmThickness = 0.5f,
		.thinFilmThicknessMin = 100.0f,
		.textureIridescence = 0u,
		.textureIridescenceThickness = 0u,
		.specularRoughnessAnisotropy = 0.0f,
		.specularAnisotropyRotation = 0.0f,
		.textureAnisotropy = 0u,
		.textureClearcoatNormal = 0u,
		.textureFuzzRoughness = 0u,
		.textureOcclusion = 0u,
		.transmissionDispersionAbbeNumber = 20.0f,
		.alphaCutoff = 0.5f,
		.geometryOpacity = 1.0f,
		.flags = 0u,
	});
	s->materialNames.push_back("");
	return mor::Scene { .id = reinterpret_cast<u64>(s) };
}

void mor::scene_destroy(mor::Scene const & scene) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	for (ImTextureID const id : s->imguiIds) {
		if (id) { vkof::image_imgui_id_destroy(id); }
	}
	for (vkof::Image const & img : s->images) {
		vkof::image_destroy(img);
	}
	delete s;
}

void mor::sampler_cache_destroy() {
	for (auto const & [key, sampler] : sSamplerCache) {
		vkof::sampler_destroy(sampler);
	}
	sSamplerCache.clear();
	if (sImguiDisplaySampler.id) {
		vkof::sampler_destroy(sImguiDisplaySampler);
		sImguiDisplaySampler = { 0u };
	}
}

void mor::scene_set_anisotropy(
	Scene const & scene,
	GpuScene const & gpuScene,
	f32 const & anisotropy
) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	ImplGpuScene * const gpu = reinterpret_cast<ImplGpuScene *>(gpuScene.id);

	for (auto const & [key, sampler] : sSamplerCache) {
		vkof::sampler_destroy(sampler);
	}
	sSamplerCache.clear();

	std::unordered_map<u32, u32> handleRemap;
	for (auto & [tex, oldHandle] : s->textureHandles) {
		SamplerKey key = s->textureSamplerKeys.at(tex);
		key.maxAnisotropy = anisotropy;

		auto cacheIt = sSamplerCache.find(key);
		if (cacheIt == sSamplerCache.end()) {
			vkof::Sampler const sampler = vkof::sampler_create({
				.magFilter = key.magFilter,
				.minFilter = key.minFilter,
				.addressModeU = key.addressU,
				.addressModeV = key.addressV,
				.addressModeW = vkof::SamplerAddressMode::repeat,
				.mipmapMode = key.mipmapMode,
				.maxAnisotropy = key.maxAnisotropy,
			});
			cacheIt = sSamplerCache.emplace(key, sampler).first;
		}

		u32 const newHandle = vkof::image_sampler_handle({
			.image = s->textureImages.at(tex),
			.sampler = cacheIt->second,
		});
		handleRemap[oldHandle] = newHandle;
		oldHandle = newHandle;
	}

	for (GpuMorMaterial & mat : s->materials) {
		auto const remap = [&](u32 h) -> u32 {
			if (h == 0u) { return 0u; }
			auto const it = handleRemap.find(h);
			return it != handleRemap.end() ? it->second : h;
		};
		mat.textureBaseColor = remap(mat.textureBaseColor);
		mat.textureNormal = remap(mat.textureNormal);
		mat.textureMetallicRoughness = remap(mat.textureMetallicRoughness);
		mat.textureEmissive = remap(mat.textureEmissive);
		mat.textureSpecular = remap(mat.textureSpecular);
		mat.textureSpecularColor = remap(mat.textureSpecularColor);
		mat.textureClearcoat = remap(mat.textureClearcoat);
		mat.textureClearcoatRoughness = remap(mat.textureClearcoatRoughness);
		mat.textureClearcoatNormal = remap(mat.textureClearcoatNormal);
		mat.textureFuzz = remap(mat.textureFuzz);
		mat.textureFuzzRoughness = remap(mat.textureFuzzRoughness);
		mat.textureSubsurface = remap(mat.textureSubsurface);
		mat.textureTransmission = remap(mat.textureTransmission);
		mat.textureIridescence = remap(mat.textureIridescence);
		mat.textureIridescenceThickness = remap(mat.textureIridescenceThickness);
		mat.textureAnisotropy = remap(mat.textureAnisotropy);
		mat.textureOcclusion = remap(mat.textureOcclusion);
	}

	if (!s->materials.empty()) {
		vkof::buffer_upload({
			.buffer = gpu->materials,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(s->materials.data()),
				s->materials.size() * sizeof(GpuMorMaterial)
			),
		});
	}
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

void mor::scene_bounds(mor::Scene const & scene, f32v3 & outMin, f32v3 & outMax) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	outMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	outMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (f32v3 const & p : s->positions) {
		outMin.x = std::min(outMin.x, p.x);
		outMin.y = std::min(outMin.y, p.y);
		outMin.z = std::min(outMin.z, p.z);
		outMax.x = std::max(outMax.x, p.x);
		outMax.y = std::max(outMax.y, p.y);
		outMax.z = std::max(outMax.z, p.z);
	}
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
			GpuMorInstance const & inst = s->instances[i];
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
			GpuMorMeshlet const & m = s->meshlets[i];
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
			GpuMorMaterial const & mat = s->materials[i];
			ImGui::Text(
				"[%u] base: (%.2f %.2f %.2f %.2f)  metal: %.2f  rough: %.2f",
				i,
				mat.baseColor.x, mat.baseColor.y,
				mat.baseColor.z, mat.baseColor.w,
				mat.baseMetalness, mat.specularRoughness
			);
		}
		ImGui::TreePop();
	}

	snprintf(label, sizeof(label), "textures (%u)", textureCount);
	if (ImGui::TreeNode(label)) {
		u32 i = 0u;
		for (auto const & [tex, handle] : s->textureHandles) {
			ImGui::Text("[%u] handle: %u", i, handle);
			++i;
		}
		ImGui::TreePop();
	}
}

void mor::scene_imgui_textures(mor::Scene const & scene) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	if (s->images.empty()) { return; }
	if (!sImguiDisplaySampler.id) {
		sImguiDisplaySampler = vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::linear,
			.minFilter = vkof::SamplerFilter::linear,
			.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
			.mipmapMode = vkof::SamplerMipmapMode::linear,
			.maxAnisotropy = 1.0f,
		});
	}
	if (s->imguiIds.empty()) {
		s->imguiIds.resize(s->images.size(), nullptr);
		for (u32 i = 0u; i < (u32)s->images.size(); ++i) {
			s->imguiIds[i] = vkof::image_imgui_id({
				.image = s->images[i],
				.sampler = sImguiDisplaySampler,
			});
		}
	}
	constexpr f32 skThumbSize = 64.0f;
	for (u32 i = 0u; i < (u32)s->imguiIds.size(); ++i) {
		char const * const name = (
			i < (u32)s->imageNames.size()
			? s->imageNames[i].c_str()
			: "[unknown]"
		);
		ImGui::TextUnformatted(name);
		ImGui::Image(s->imguiIds[i], ImVec2(skThumbSize, skThumbSize));
	}
}

void mor::scene_load_gltf(mor::Scene const & scene, char const * const path) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);

	s->gltfDir = std::filesystem::path(path).parent_path().string();

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

	gpu->positions = (
		upload_buffer(
			s->positions.data(), s->positions.size() * sizeof(f32v3)
		)
	);
	gpu->attributes = (
		upload_buffer(
			s->attributes.data(), s->attributes.size() * sizeof(GpuMorVertexAttribute)
		)
	);
	gpu->meshletVerts = (
		upload_buffer(
			s->meshletVerts.data(), s->meshletVerts.size() * sizeof(u32)
		)
	);
	gpu->meshletTris = (
		upload_buffer(
			s->meshletTris.data(), s->meshletTris.size() * sizeof(u8)
		)
	);
	gpu->meshlets = (
		upload_buffer(
			s->meshlets.data(), s->meshlets.size() * sizeof(GpuMorMeshlet)
		)
	);
	gpu->instances = (
		upload_buffer(
			s->instances.data(), s->instances.size() * sizeof(GpuMorInstance)
		)
	);
	if (!s->materials.empty()) {
		gpu->materials = (
			upload_buffer(
				s->materials.data(), s->materials.size() * sizeof(GpuMorMaterial)
			)
		);
	}
	gpu->meshletCount = (u32)s->meshlets.size();
	gpu->vertexCount = (u32)s->positions.size();

	// -- flat u32 index buffer for BLAS geometry + parallel meshlet index per triangle
	std::vector<u32> flatIndices;
	std::vector<u32> flatMeshlets;
	for (u32 mi = 0u; mi < (u32)s->meshlets.size(); ++mi) {
		GpuMorMeshlet const & m = s->meshlets[mi];
		for (u32 tri = 0u; tri < m.triangleCount; ++tri) {
			u32 const base = m.triangleOffset + tri * 3u;
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 0u]]
			);
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 1u]]
			);
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 2u]]
			);
			flatMeshlets.emplace_back(mi);
		}
	}
	gpu->triangleCount = (u32)(flatIndices.size() / 3u);
	gpu->flatIndices = (
		upload_buffer(
			flatIndices.data(), flatIndices.size() * sizeof(u32)
		)
	);
	gpu->flatMeshlets = (
		upload_buffer(
			flatMeshlets.data(), flatMeshlets.size() * sizeof(u32)
		)
	);

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
	vkof::buffer_destroy(gpu->flatIndices);
	vkof::buffer_destroy(gpu->flatMeshlets);
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
		.flatIndices = vkof::buffer_virtual_address(gpu->flatIndices),
		.flatMeshlets = vkof::buffer_virtual_address(gpu->flatMeshlets),
		.vertexCount = gpu->vertexCount,
		.triangleCount = gpu->triangleCount,
	};
}

u32 mor::scene_gpu_meshlet_count(mor::GpuScene const & scene) {
	return reinterpret_cast<ImplGpuScene const *>(scene.id)->meshletCount;
}

u32 mor::scene_material_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->materials.size();
}

std::string mor::scene_material_name(mor::Scene const & scene, u32 const index) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	SRAT_ASSERT(index < (u32)s->materialNames.size());
	return s->materialNames[index];
}

GpuMorMaterial mor::scene_material_get(mor::Scene const & scene, u32 const index) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	SRAT_ASSERT(index < (u32)s->materials.size());
	return s->materials[index];
}

mor::GpuMaterials mor::scene_gpu_materials_create(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	ImplGpuMaterials * const m = new ImplGpuMaterials();
	m->cpu = s->materials;
	if (!m->cpu.empty()) {
		m->buffer = upload_buffer(
			m->cpu.data(), m->cpu.size() * sizeof(GpuMorMaterial)
		);
	}
	return mor::GpuMaterials { .id = reinterpret_cast<u64>(m) };
}

void mor::scene_gpu_materials_destroy(mor::GpuMaterials const & mats) {
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	vkof::buffer_destroy(m->buffer);
	delete m;
}

u64 mor::scene_gpu_materials_va(mor::GpuMaterials const & mats) {
	ImplGpuMaterials const * const m = reinterpret_cast<ImplGpuMaterials const *>(mats.id);
	return vkof::buffer_virtual_address(m->buffer);
}

void mor::scene_material_override_scalars(
	mor::GpuMaterials const & mats, u32 const index, GpuMorMaterial const & scalars
) {
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	SRAT_ASSERT(index < (u32)m->cpu.size());
	GpuMorMaterial & dst = m->cpu[index];
	dst.baseColor = scalars.baseColor;
	dst.baseMetalness = scalars.baseMetalness;
	dst.specularRoughness = scalars.specularRoughness;
	dst.emissiveColor = scalars.emissiveColor;
	dst.emissiveLuminance = scalars.emissiveLuminance;
	dst.specularColor = scalars.specularColor;
	dst.specularWeight = scalars.specularWeight;
	dst.specularIor = scalars.specularIor;
	dst.coatWeight = scalars.coatWeight;
	dst.coatRoughness = scalars.coatRoughness;
	dst.fuzzColor = scalars.fuzzColor;
	dst.fuzzWeight = scalars.fuzzWeight;
	dst.fuzzRoughness = scalars.fuzzRoughness;
	dst.subsurfaceWeight = scalars.subsurfaceWeight;
	dst.subsurfaceColor = scalars.subsurfaceColor;
	dst.subsurfaceRadius = scalars.subsurfaceRadius;
	dst.transmissionWeight = scalars.transmissionWeight;
	dst.transmissionColor = scalars.transmissionColor;
	dst.transmissionDepth = scalars.transmissionDepth;
	dst.thinFilmWeight = scalars.thinFilmWeight;
	dst.thinFilmIor = scalars.thinFilmIor;
	dst.thinFilmThickness = scalars.thinFilmThickness;
	dst.thinFilmThicknessMin = scalars.thinFilmThicknessMin;
	dst.specularRoughnessAnisotropy = scalars.specularRoughnessAnisotropy;
	dst.specularAnisotropyRotation = scalars.specularAnisotropyRotation;
	dst.transmissionDispersionAbbeNumber = scalars.transmissionDispersionAbbeNumber;
	dst.alphaCutoff = scalars.alphaCutoff;
	dst.geometryOpacity = scalars.geometryOpacity;
	dst.flags = scalars.flags;
	dst.textureBaseColor = scalars.textureBaseColor;
	dst.textureNormal = scalars.textureNormal;
	dst.textureMetallicRoughness = scalars.textureMetallicRoughness;
	dst.textureEmissive = scalars.textureEmissive;
	dst.textureSpecular = scalars.textureSpecular;
	dst.textureSpecularColor = scalars.textureSpecularColor;
	dst.textureClearcoat = scalars.textureClearcoat;
	dst.textureClearcoatRoughness = scalars.textureClearcoatRoughness;
	dst.textureClearcoatNormal = scalars.textureClearcoatNormal;
	dst.textureFuzz = scalars.textureFuzz;
	dst.textureFuzzRoughness = scalars.textureFuzzRoughness;
	dst.textureSubsurface = scalars.textureSubsurface;
	dst.textureSubsurfaceColor = scalars.textureSubsurfaceColor;
	dst.textureTransmission = scalars.textureTransmission;
	dst.textureTransmissionColor = scalars.textureTransmissionColor;
	dst.textureIridescence = scalars.textureIridescence;
	dst.textureIridescenceThickness = scalars.textureIridescenceThickness;
	dst.textureAnisotropy = scalars.textureAnisotropy;
	dst.textureOcclusion = scalars.textureOcclusion;
	vkof::buffer_upload({
		.buffer = m->buffer,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(m->cpu.data()),
			m->cpu.size() * sizeof(GpuMorMaterial)
		),
	});
}

void mor::scene_gpu_materials_sync_textures(
	mor::Scene const & scene, mor::GpuMaterials const & mats
) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	u32 const count = (u32)std::min(m->cpu.size(), s->materials.size());
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial & dst = m->cpu[i];
		GpuMorMaterial const & src = s->materials[i];
		dst.textureBaseColor = src.textureBaseColor;
		dst.textureNormal = src.textureNormal;
		dst.textureMetallicRoughness = src.textureMetallicRoughness;
		dst.textureEmissive = src.textureEmissive;
		dst.textureSpecular = src.textureSpecular;
		dst.textureSpecularColor = src.textureSpecularColor;
		dst.textureClearcoat = src.textureClearcoat;
		dst.textureClearcoatRoughness = src.textureClearcoatRoughness;
		dst.textureClearcoatNormal = src.textureClearcoatNormal;
		dst.textureFuzz = src.textureFuzz;
		dst.textureFuzzRoughness = src.textureFuzzRoughness;
		dst.textureSubsurface = src.textureSubsurface;
		dst.textureTransmission = src.textureTransmission;
		dst.textureIridescence = src.textureIridescence;
		dst.textureIridescenceThickness = src.textureIridescenceThickness;
		dst.textureAnisotropy = src.textureAnisotropy;
		dst.textureOcclusion = src.textureOcclusion;
	}
	if (!m->cpu.empty()) {
		vkof::buffer_upload({
			.buffer = m->buffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(m->cpu.data()),
				m->cpu.size() * sizeof(GpuMorMaterial)
			),
		});
	}
}
