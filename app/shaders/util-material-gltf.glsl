// Requires: Material struct (from mor/mor_shared.h)
// Requires: GL_EXT_nonuniform_qualifier enabled in including shader
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]
// Requires: pc.global (GpuGlobalPC) accessible in including shader scope

// imports glTF material into OpenPBR material

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_debug_printf : require

#ifndef UTIL_MATERIAL_GLTF_GLSL
#define UTIL_MATERIAL_GLTF_GLSL

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

#include "util-material-openpbr.glsl"

OpenPbrMaterial openPbrMaterialFromGltf(
	const GpuMorMaterial mat,
	const f32v2 uv,
	const f32v2 uvDx,
	const f32v2 uvDy,
	out f32v3 modelNormal
) {
#define SAMPLER(h) fnSampleTexture(h, uv, uvDx, uvDy)

#define SAMPLE_RGBA(label, field, texField) \
	f32v4 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField); }
#define SAMPLE_RGB(label, field, texField) \
	f32v3 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField).rgb; }
#define SAMPLE_SCALAR(label, field, texField, swizzle) \
	f32 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField).swizzle; }

	MOR_MATERIAL_RGBA_PARAMS(SAMPLE_RGBA)
	MOR_MATERIAL_RGB_PARAMS(SAMPLE_RGB)
	MOR_MATERIAL_SCALAR_TEX_PARAMS(SAMPLE_SCALAR)

#undef SAMPLE_RGBA
#undef SAMPLE_RGB
#undef SAMPLE_SCALAR

	// metallic roughness: shared texture, two channels
	f32 roughness = mat.specularRoughness;
	f32 metallic = mat.baseMetalness;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = SAMPLER(mat.textureMetallicRoughness);
		roughness *= mr.g;
		metallic *= mr.b;
	}

	// normal map: unpack [-1,1] not a multiply
	f32v3 normal = f32v3(0.0f, 0.0f, 1.0f);
	if (mat.textureNormal != 0u) {
		normal = (
			SAMPLER(mat.textureNormal).rgb * 2.0f - 1.0f
		);
	}
	modelNormal = normal;

	// thin film thickness: mix between min/max, not a multiply
	f32 thinFilmThickness = mat.thinFilmThickness;
	if (mat.textureIridescenceThickness != 0u) {
		const f32 t = SAMPLER(mat.textureIridescenceThickness).r;
		thinFilmThickness = mix(mat.thinFilmThicknessMin, mat.thinFilmThickness, t);
	}

#undef SAMPLER

	const uint alphaMode = mat.flags & 0x3u;
	const f32 geometryOpacity = (
		alphaMode == 1u ? ((baseColor.a >= mat.alphaCutoff) ? 1.0f : 0.0f) :
		alphaMode == 2u ? baseColor.a :
		1.0f
	);

	OpenPbrMaterial pbrMat;
	pbrMat.baseColor = baseColor.rgb;
	pbrMat.baseWeight = 1.0f - mat.transmissionWeight;
	pbrMat.baseMetallic = metallic;
	pbrMat.baseDiffuseRoughness = 0.04f;
	pbrMat.specularColor = specularColor;
	pbrMat.specularIor = mat.specularIor;
	pbrMat.specularWeight = specularWeight;
	pbrMat.specularRoughnessScale = clamp(roughness, 0.04f, 1.0f);
	pbrMat.specularRoughnessAnisotropy = specularRoughnessAnisotropy;
	pbrMat.subsurfaceWeight = subsurfaceWeight;
	pbrMat.subsurfaceColor = subsurfaceColor;
	pbrMat.subsurfaceRadius = mat.subsurfaceRadius;
	pbrMat.subsurfaceRadiusScale = f32v3(1.0f, 0.5f, 0.25f);
	pbrMat.subsurfaceScatterAnisotropy = 0.0f;
	pbrMat.transmissionWeight = transmissionWeight;
	pbrMat.transmissionColor = transmissionColor;
	pbrMat.transmissionDepth = mat.transmissionDepth;
	pbrMat.transmissionDispersionAbbeNumber = mat.transmissionDispersionAbbeNumber;
	pbrMat.coatWeight = coatWeight;
	pbrMat.coatColor = subsurfaceColor;
	pbrMat.coatRoughness = clamp(coatRoughness, 0.04f, 1.0f);
	pbrMat.coatIor = 1.6f;
	pbrMat.fuzzWeight = mat.fuzzWeight;
	pbrMat.fuzzColor = mat.fuzzColor;
	pbrMat.fuzzRoughness = clamp(fuzzRoughness, 0.04f, 1.0f);
	pbrMat.thinFilmWeight = thinFilmWeight;
	pbrMat.thinFilmIor = mat.thinFilmIor;
	pbrMat.thinFilmThickness = thinFilmThickness;
	pbrMat.emissionColor = emissiveColor;
	pbrMat.emissionLuminance = mat.emissiveLuminance;
	pbrMat.geometryOpacity = geometryOpacity;
	pbrMat.geometryThinWalled = f32((mat.flags & 0x4u) != 0u);

	// one thing to note; specular color factor in openpbr is only a factor
	// in metallic, but in glTF it is a factor for both metallic and dielectric.
	// so specular color factor needs to be applied to metallic
	// this is a hack to try to match glTF
	if (any(notEqual(specularColor, f32v3(1.0f)))) {
		pbrMat.baseMetallic = mix(pbrMat.baseMetallic, 1.0f, max(specularColor.r, max(specularColor.g, specularColor.b))*0.5f);
	}

	const GpuDebugPC unf = GpuDebugPCBuffer(pc.global.debug).data;
	if (gl_GlobalInvocationID.x == unf.probeX
		&& gl_GlobalInvocationID.y == unf.probeY
		&& unf.probeActive == 1
	) {
		debugPrintfEXT(
			"baseColor=%f,%f,%f,; baseWeight=%f; baseMetallic=%f;\n baseDiffuseRoughness=%f; specularColor=%f,%f,%f; specularIor=%f;\n specularWeight=%f; specularRoughnessScale=%f; specularRoughnessAnisotropy=%f;\n subsurfaceWeight=%f; subsurfaceColor=%f,%f,%f; subsurfaceRadius=%f;\n subsurfaceRadiusScale=%f,%f,%f; subsurfaceScatterAnisotropy=%f; transmissionWeight=%f;\n transmissionColor=%f,%f,%f; transmissionDepth=%f; transmissionDispersionAbbeNumber=%f;\n coatWeight=%f; coatColor=%f,%f,%f; coatRoughness=%f;\n coatIor=%f; fuzzWeight=%f; fuzzColor=%f,%f,%f;\n fuzzRoughness=%f; thinFilmWeight=%f; thinFilmIor=%f;\n thinFilmThickness=%f; emissionColor=%f,%f,%f; emissionLuminance=%f;\n geometryOpacity=%f, geometryThinWalled=%d\n",
			pbrMat.baseColor.r, pbrMat.baseColor.g, pbrMat.baseColor.b,
			pbrMat.baseWeight, pbrMat.baseMetallic, pbrMat.baseDiffuseRoughness,
			pbrMat.specularColor.r, pbrMat.specularColor.g, pbrMat.specularColor.b,
			pbrMat.specularIor, pbrMat.specularWeight, pbrMat.specularRoughnessScale, pbrMat.specularRoughnessAnisotropy,
			pbrMat.subsurfaceWeight, pbrMat.subsurfaceColor.r, pbrMat.subsurfaceColor.g, pbrMat.subsurfaceColor.b,
			pbrMat.subsurfaceRadius,
			pbrMat.subsurfaceRadiusScale.r, pbrMat.subsurfaceRadiusScale.g, pbrMat.subsurfaceRadiusScale.b,
			pbrMat.subsurfaceScatterAnisotropy, pbrMat.transmissionWeight, pbrMat.transmissionColor.r, pbrMat.transmissionColor.g, pbrMat.transmissionColor.b,
			pbrMat.transmissionDepth, pbrMat.transmissionDispersionAbbeNumber,
			pbrMat.coatWeight, pbrMat.coatColor.r, pbrMat.coatColor.g, pbrMat.coatColor.b, pbrMat.coatRoughness, pbrMat.coatIor,
			pbrMat.fuzzWeight, pbrMat.fuzzColor.r, pbrMat.fuzzColor.g, pbrMat.fuzzColor.b, pbrMat.fuzzRoughness,
			pbrMat.thinFilmWeight, pbrMat.thinFilmIor, pbrMat.thinFilmThickness,
			pbrMat.emissionColor.r, pbrMat.emissionColor.g, pbrMat.emissionColor.b, pbrMat.emissionLuminance,
			pbrMat.geometryOpacity, int(pbrMat.geometryThinWalled)
		);
	}

	return pbrMat;
}

OpenPbrMaterial openPbrMaterialFromGltfWithLod(
	const GpuMorMaterial mat,
	const f32v2 uv,
	const uint lod,
	out f32v3 modelNormal
) {
#define SAMPLER(h) fnSampleTextureWithLod(h, uv, lod)

#define SAMPLE_RGBA(label, field, texField) \
	f32v4 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField); }
#define SAMPLE_RGB(label, field, texField) \
	f32v3 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField).rgb; }
#define SAMPLE_SCALAR(label, field, texField, swizzle) \
	f32 field = mat.field; \
	if (mat.texField != 0u) { field *= SAMPLER(mat.texField).swizzle; }

	MOR_MATERIAL_RGBA_PARAMS(SAMPLE_RGBA)
	MOR_MATERIAL_RGB_PARAMS(SAMPLE_RGB)
	MOR_MATERIAL_SCALAR_TEX_PARAMS(SAMPLE_SCALAR)

#undef SAMPLE_RGBA
#undef SAMPLE_RGB
#undef SAMPLE_SCALAR

	// metallic roughness: shared texture, two channels
	f32 roughness = mat.specularRoughness;
	f32 metallic = mat.baseMetalness;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = SAMPLER(mat.textureMetallicRoughness);
		roughness *= mr.g;
		metallic *= mr.b;
	}

	// normal map: unpack [-1,1] not a multiply
	f32v3 normal = f32v3(0.0f, 0.0f, 1.0f);
	if (mat.textureNormal != 0u) {
		normal = (
			SAMPLER(mat.textureNormal).rgb * 2.0f - 1.0f
		);
	}
	modelNormal = normal;

	// thin film thickness: mix between min/max, not a multiply
	f32 thinFilmThickness = mat.thinFilmThickness;
	if (mat.textureIridescenceThickness != 0u) {
		const f32 t = SAMPLER(mat.textureIridescenceThickness).r;
		thinFilmThickness = mix(mat.thinFilmThicknessMin, mat.thinFilmThickness, t);
	}

#undef SAMPLER

	// fuzz weight = max component of (texture-modulated) fuzz color
	const f32 fuzzWeight = max(fuzzColor.r, max(fuzzColor.g, fuzzColor.b));
	fuzzColor = (fuzzWeight > 0.0f) ? (fuzzColor / fuzzWeight) : f32v3(1.0f);

	const uint alphaMode = mat.flags & 0x3u;
	const f32 geometryOpacity = (
		alphaMode == 1u ? ((baseColor.a >= mat.alphaCutoff) ? 1.0f : 0.0f) :
		alphaMode == 2u ? baseColor.a :
		1.0f
	);

	OpenPbrMaterial pbrMat;
	pbrMat.baseColor = baseColor.rgb;
	pbrMat.baseWeight = 1.0f - mat.transmissionWeight;
	pbrMat.baseMetallic = metallic;
	pbrMat.baseDiffuseRoughness = 0.04f;
	pbrMat.specularColor = specularColor;
	pbrMat.specularIor = mat.specularIor;
	pbrMat.specularWeight = specularWeight;
	pbrMat.specularRoughnessScale = clamp(roughness, 0.04f, 1.0f);
	pbrMat.specularRoughnessAnisotropy = specularRoughnessAnisotropy;
	pbrMat.subsurfaceWeight = subsurfaceWeight;
	pbrMat.subsurfaceColor = subsurfaceColor;
	pbrMat.subsurfaceRadius = mat.subsurfaceRadius;
	pbrMat.subsurfaceRadiusScale = f32v3(1.0f, 0.5f, 0.25f);
	pbrMat.subsurfaceScatterAnisotropy = 0.0f;
	pbrMat.transmissionWeight = transmissionWeight;
	pbrMat.transmissionColor = transmissionColor;
	pbrMat.transmissionDepth = mat.transmissionDepth;
	pbrMat.transmissionDispersionAbbeNumber = mat.transmissionDispersionAbbeNumber;
	pbrMat.coatWeight = coatWeight;
	pbrMat.coatColor = subsurfaceColor;
	pbrMat.coatRoughness = clamp(coatRoughness, 0.04f, 1.0f);
	pbrMat.coatIor = 1.6f;
	pbrMat.fuzzWeight = fuzzWeight;
	pbrMat.fuzzColor = fuzzColor;
	pbrMat.fuzzRoughness = clamp(fuzzRoughness, 0.04f, 1.0f);
	pbrMat.thinFilmWeight = thinFilmWeight;
	pbrMat.thinFilmIor = mat.thinFilmIor;
	pbrMat.thinFilmThickness = thinFilmThickness;
	pbrMat.emissionColor = emissiveColor;
	pbrMat.emissionLuminance = mat.emissiveLuminance;
	pbrMat.geometryOpacity = geometryOpacity;
	pbrMat.geometryThinWalled = f32((mat.flags & 0x4u) != 0u);
	return pbrMat;
}

#endif // UTIL_MATERIAL_GLTF_GLSL
