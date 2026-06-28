// Requires: Material struct (from mor/mor_shared.h)
// Requires: GL_EXT_nonuniform_qualifier enabled in including shader
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]
// Requires: pc.global (GpuGlobalPC) accessible in including shader scope

// imports glTF material into BsdfMaterial

#ifndef UTIL_MATERIAL_GLTF_GLSL
#define UTIL_MATERIAL_GLTF_GLSL

const f32 PI  = 3.14159265358979323846;
const f32 TAU = 6.28318530717958647692;
const f32 IPI = 0.31830988618379067154;

f32v4 fnSampleTexture(
	const uint handle,
	const f32v2 uv,
	const f32v2 uvDx,
	const f32v2 uvDy
) {
	const GpuDebugPC dbg = GpuDebugPCBuffer(pc.global.debug).data;
	if (dbg.mipOverrideActive != 0u) {
		return textureLod(
			vkofTextures[nonuniformEXT(handle)],
			uv,
			dbg.mipLodOverride
		);
	}
	return textureGrad(vkofTextures[nonuniformEXT(handle)], uv, uvDx, uvDy);
}

f32v4 fnSampleTextureWithLod(
	const uint handle,
	const f32v2 uv,
	const uint lod
) {
	const GpuDebugPC dbg = GpuDebugPCBuffer(pc.global.debug).data;
	if (dbg.mipOverrideActive != 0u) {
		return textureLod(
			vkofTextures[nonuniformEXT(handle)],
			uv,
			dbg.mipLodOverride
		);
	}
	return textureLod(vkofTextures[nonuniformEXT(handle)], uv, lod);
}

struct BsdfMaterial {
	// normal from normal map
	f32v3 normalWorld;
	f32v3 normalGeometrical;
	f32v3 albedo;
	f32v3 emissive;
	f32 alpha;
	f32 diffuse;
	f32 fresnel;
	f32 transmittive;
};

BsdfMaterial bsdfMaterialFromGltf(
	const GpuMorMaterial mat,
	const f32v3 normalGeometrical,
	const mat3 tbn,
	const f32v2 uv,
	const f32v2 uvDx,
	const f32v2 uvDy
) {
	f32v4 baseColor = mat.baseColor;
	if (mat.textureBaseColor != 0u) {
		baseColor *= fnSampleTexture(mat.textureBaseColor, uv, uvDx, uvDy);
	}
	f32 roughness = mat.roughness;
	f32 metallic = mat.metallic;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = (
			fnSampleTexture(mat.textureMetallicRoughness, uv, uvDx, uvDy)
		);
		roughness *= mr.g;
		metallic *= mr.b;
	}
	f32v3 emissive = mat.emissive;
	if (mat.textureEmissive != 0u) {
		emissive *= (
			fnSampleTexture(mat.textureEmissive, uv, uvDx, uvDy).rgb
		);
	}
	f32v3 normal = f32v3(0.0, 0.0, 1.0);
	if (mat.textureNormal != 0u) {
		normal = (
			fnSampleTexture(mat.textureNormal, uv, uvDx, uvDy).rgb * 2.0 - 1.0
		);
	}
	BsdfMaterial bmat;
	bmat.albedo = baseColor.rgb;
	bmat.normalWorld = normalize(tbn * normal);
	bmat.normalGeometrical = normalize(normalGeometrical);
	bmat.emissive = emissive;
	bmat.alpha = roughness;
	bmat.diffuse = clamp(1.0 - metallic, 0.01f, 0.99f);
	bmat.fresnel = mix(0.04, 1.0, metallic);
	bmat.transmittive = 0.0;
	return bmat;
}

BsdfMaterial bsdfMaterialFromGltfWithLod(
	const GpuMorMaterial mat,
	const f32v3 normalGeometrical,
	mat3 tbn,
	const f32v2 uv,
	const uint lod
) {
	f32v4 baseColor = mat.baseColor;
	if (mat.textureBaseColor != 0u) {
		baseColor *= fnSampleTextureWithLod(mat.textureBaseColor, uv, lod);
	}
	f32 roughness = mat.roughness;
	f32 metallic = mat.metallic;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = (
			fnSampleTextureWithLod(mat.textureMetallicRoughness, uv, lod)
		);
		roughness *= mr.g;
		metallic *= mr.b;
	}
	f32v3 emissive = mat.emissive;
	if (mat.textureEmissive != 0u) {
		emissive *= (
			fnSampleTextureWithLod(mat.textureEmissive, uv, lod).rgb
		);
	}
	f32v3 normal = f32v3(0.0, 0.0, 1.0);
	if (mat.textureNormal != 0u) {
		normal = (
			fnSampleTextureWithLod(mat.textureNormal, uv, lod).rgb * 2.0 - 1.0
		);
	}
	BsdfMaterial bmat;
	bmat.albedo = baseColor.rgb;
	bmat.normalWorld = normalize(tbn * normal);
	bmat.normalGeometrical = normalize(normalGeometrical);
	bmat.emissive = emissive;
	bmat.alpha = roughness;
	bmat.diffuse = clamp(1.0 - metallic, 0.01f, 0.99f);
	bmat.fresnel = mix(0.04, 1.0, metallic);
	bmat.transmittive = 0.0;
	return bmat;
}

#endif // UTIL_MATERIAL_GLTF_GLSL
