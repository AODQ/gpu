#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

layout(set = 0, binding = 0) uniform sampler2D vkofTextures[];

#include "scene_shared.h"
#include "probe.glsl"
#include "material.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inTangent;
layout(location = 2) in vec2 inUv;
layout(location = 3) in flat uint inMaterialIndex;
layout(location = 4) in flat uint inMeshletIndex;
layout(location = 5) in flat uint inInstanceIndex;
layout(location = 6) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

vec3 index_color(uint idx) {
	uint h = idx * 2654435761u;
	h ^= h >> 16u;
	return (
		vec3(
			float(h & 0xFFu),
			float((h >> 8u) & 0xFFu),
			float((h >> 16u) & 0xFFu)
		) / 255.0
	);
}

void main() {
	MaterialBuf materialBuf = MaterialBuf(pc.draw.materials);
	const Material mat = materialBuf.data[inMaterialIndex];
	const f32v3 N = normalize(inNormal);
	const f32v3 T = normalize(inTangent.xyz);
	const f32v3 B = cross(N, T) * inTangent.w;
	const mat3 TBN = mat3(T, B, N);
	const BsdfMaterial bmat = bsdf_material_from_gltf(mat, inUv);
	const vec3 worldNormal = normalize(TBN * bmat.modelNormal);
	const f32v3 wi = normalize(inWorldPos - pc.global.cameraPos);
	const f32v3 sun = normalize(f32v3(0.4, 1.0, 0.6));
	const f32 dotNL = max(dot(worldNormal, sun), 0.0);
	const f32v3 directLight = (
		fnBsdfF(worldNormal, wi, sun, bmat) * f32v3(3.0) * dotNL
	);
	const f32v3 ambient = bmat.albedo * 0.02;
	outColor = f32v4(directLight + ambient + bmat.emissive, 1.0);

	if (pc.global.debugMode == 1u) {
		outColor = vec4(index_color(inMeshletIndex), 1.0);
	} else if (pc.global.debugMode == 2u) {
		outColor = vec4(index_color(inMaterialIndex), 1.0);
	} else if (pc.global.debugMode == 3u) {
		outColor = vec4(index_color(inInstanceIndex), 1.0);
	} else if (pc.global.debugMode == 4u) {
		const float d = pow(1.0 - gl_FragCoord.z, 1.0f/2.2);
		outColor = vec4(d, d, d, 1.0);
	}

	if (shader_probe_is_active_pixel()) {
		debugPrintfEXT(
			"meshlet: %u\nmaterial: %u\ninstance: %u\ndepth: %f\nbase color: %.2f %.2f %.2f\ntexture: %u\n",
			inMeshletIndex, inMaterialIndex, inInstanceIndex, gl_FragCoord.z,
			mat.baseColor.r, mat.baseColor.g, mat.baseColor.b,
			mat.textureBaseColor
		);
		outColor = vec4(
			sin(pc.global.time * 10.0f) * 0.5f + 0.5f,
			cos(pc.global.time * 10.0f) * 0.5f + 0.5f,
			sin(pc.global.time * 10.0f + 3.14f) * 0.5f + 0.5f,
			1.0f);
	}
}
