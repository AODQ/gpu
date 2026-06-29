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
	f32v4 baseColor = mat.baseColor;
	if (mat.textureBaseColor != 0u) {
		baseColor *= fnSampleTexture(mat.textureBaseColor, uv, uvDx, uvDy);
	}
	f32 roughness = mat.specularRoughness;
	f32 metallic = mat.baseMetalness;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = (
			fnSampleTexture(mat.textureMetallicRoughness, uv, uvDx, uvDy)
		);
		roughness *= mr.g;
		metallic *= mr.b;
	}
	f32v3 emissiveColor = mat.emissiveColor;
	if (mat.textureEmissive != 0u) {
		emissiveColor *= fnSampleTexture(mat.textureEmissive, uv, uvDx, uvDy).rgb;
	}
	f32v3 normal = f32v3(0.0f, 0.0f, 1.0f);
	if (mat.textureNormal != 0u) {
		normal = (
			fnSampleTexture(mat.textureNormal, uv, uvDx, uvDy).rgb * 2.0f - 1.0f
		);
	}
	modelNormal = normal;

	f32v3 specularColor = mat.specularColor;
	f32 specularWeight = mat.specularWeight;
	if (mat.textureSpecularColor != 0u) {
		specularColor *= (
			fnSampleTexture(mat.textureSpecularColor, uv, uvDx, uvDy).rgb
		);
	}
	if (mat.textureSpecular != 0u) {
		specularWeight *= fnSampleTexture(mat.textureSpecular, uv, uvDx, uvDy).a;
	}

	f32 coatWeight = mat.coatWeight;
	f32 coatRoughness = mat.coatRoughness;
	if (mat.textureClearcoat != 0u) {
		coatWeight *= fnSampleTexture(mat.textureClearcoat, uv, uvDx, uvDy).r;
	}
	if (mat.textureClearcoatRoughness != 0u) {
		coatRoughness *= (
			fnSampleTexture(mat.textureClearcoatRoughness, uv, uvDx, uvDy).g
		);
	}

	f32v3 fuzzColor = mat.fuzzColor;
	f32 fuzzRoughness = mat.fuzzRoughness;
	if (mat.textureFuzz != 0u) {
		fuzzColor *= fnSampleTexture(mat.textureFuzz, uv, uvDx, uvDy).rgb;
	}
	if (mat.textureFuzzRoughness != 0u) {
		fuzzRoughness *= (
			fnSampleTexture(mat.textureFuzzRoughness, uv, uvDx, uvDy).g
		);
	}
	const f32 fuzzWeight = max(fuzzColor.r, max(fuzzColor.g, fuzzColor.b));
	fuzzColor = (fuzzWeight > 0.0f) ? (fuzzColor / fuzzWeight) : f32v3(1.0f);

	f32 subsurfaceWeight = mat.subsurfaceWeight;
	if (mat.textureSubsurface != 0u) {
		subsurfaceWeight *= fnSampleTexture(mat.textureSubsurface, uv, uvDx, uvDy).r;
	}

	f32 transmissionWeight = mat.transmissionWeight;
	if (mat.textureTransmission != 0u) {
		transmissionWeight *= (
			fnSampleTexture(mat.textureTransmission, uv, uvDx, uvDy).r
		);
	}

	f32 thinFilmWeight = mat.thinFilmWeight;
	if (mat.textureIridescence != 0u) {
		thinFilmWeight *= fnSampleTexture(mat.textureIridescence, uv, uvDx, uvDy).r;
	}
	f32 thinFilmThickness = mat.thinFilmThickness;
	if (mat.textureIridescenceThickness != 0u) {
		const f32 t = (
			fnSampleTexture(mat.textureIridescenceThickness, uv, uvDx, uvDy).r
		);
		thinFilmThickness = mix(mat.thinFilmThicknessMin, mat.thinFilmThickness, t);
	}

	f32 specularRoughnessAnisotropy = mat.specularRoughnessAnisotropy;
	if (mat.textureAnisotropy != 0u) {
		specularRoughnessAnisotropy *= (
			fnSampleTexture(mat.textureAnisotropy, uv, uvDx, uvDy).b
		);
	}

	const uint alphaMode = mat.flags & 0x3u;
	const f32 geometryOpacity = (
		alphaMode == 1u ? ((baseColor.a >= mat.alphaCutoff) ? 1.0f : 0.0f) :
		alphaMode == 2u ? baseColor.a :
		1.0f
	);

	OpenPbrMaterial pbrMat;
	pbrMat.baseColor = baseColor.rgb;
	pbrMat.baseWeight = 1.0f;
	pbrMat.baseMetallic = metallic;
	pbrMat.baseDiffuseRoughness = 0.04f;
	pbrMat.specularColor = specularColor;
	pbrMat.specularIor = mat.specularIor;
	pbrMat.specularWeight = specularWeight;
	pbrMat.specularRoughnessScale = clamp(roughness, 0.04f, 1.0f);
	pbrMat.specularRoughnessAnisotropy = specularRoughnessAnisotropy;
	pbrMat.subsurfaceWeight = subsurfaceWeight;
	pbrMat.subsurfaceColor = mat.subsurfaceColor;
	pbrMat.subsurfaceRadius = mat.subsurfaceRadius;
	pbrMat.subsurfaceRadiusScale = f32v3(1.0f, 0.5f, 0.25f);
	pbrMat.subsurfaceScatterAnisotropy = 0.0f;
	pbrMat.transmissionWeight = transmissionWeight;
	pbrMat.transmissionColor = mat.transmissionColor;
	pbrMat.transmissionDepth = mat.transmissionDepth;
	pbrMat.transmissionDispersionAbbeNumber = mat.transmissionDispersionAbbeNumber;
	pbrMat.coatWeight = coatWeight;
	pbrMat.coatColor = mat.subsurfaceColor;
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
	f32v4 baseColor = mat.baseColor;
	if (mat.textureBaseColor != 0u) {
		baseColor *= fnSampleTextureWithLod(mat.textureBaseColor, uv, lod);
	}
	f32 roughness = mat.specularRoughness;
	f32 metallic = mat.baseMetalness;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = (
			fnSampleTextureWithLod(mat.textureMetallicRoughness, uv, lod)
		);
		roughness *= mr.g;
		metallic *= mr.b;
	}
	f32v3 emissiveColor = mat.emissiveColor;
	if (mat.textureEmissive != 0u) {
		emissiveColor *= (
			fnSampleTextureWithLod(mat.textureEmissive, uv, lod).rgb
		);
	}
	f32v3 normal = f32v3(0.0f, 0.0f, 1.0f);
	if (mat.textureNormal != 0u) {
		normal = (
			fnSampleTextureWithLod(mat.textureNormal, uv, lod).rgb * 2.0f - 1.0f
		);
	}
	modelNormal = normal;

	f32v3 specularColor = mat.specularColor;
	f32 specularWeight = mat.specularWeight;
	if (mat.textureSpecularColor != 0u) {
		specularColor *= (
			fnSampleTextureWithLod(mat.textureSpecularColor, uv, lod).rgb
		);
	}
	if (mat.textureSpecular != 0u) {
		specularWeight *= fnSampleTextureWithLod(mat.textureSpecular, uv, lod).a;
	}

	f32 coatWeight = mat.coatWeight;
	f32 coatRoughness = mat.coatRoughness;
	if (mat.textureClearcoat != 0u) {
		coatWeight *= fnSampleTextureWithLod(mat.textureClearcoat, uv, lod).r;
	}
	if (mat.textureClearcoatRoughness != 0u) {
		coatRoughness *= (
			fnSampleTextureWithLod(mat.textureClearcoatRoughness, uv, lod).g
		);
	}

	f32v3 fuzzColor = mat.fuzzColor;
	f32 fuzzRoughness = mat.fuzzRoughness;
	if (mat.textureFuzz != 0u) {
		fuzzColor *= fnSampleTextureWithLod(mat.textureFuzz, uv, lod).rgb;
	}
	if (mat.textureFuzzRoughness != 0u) {
		fuzzRoughness *= fnSampleTextureWithLod(mat.textureFuzzRoughness, uv, lod).g;
	}
	const f32 fuzzWeight = max(fuzzColor.r, max(fuzzColor.g, fuzzColor.b));
	fuzzColor = (fuzzWeight > 0.0f) ? (fuzzColor / fuzzWeight) : f32v3(1.0f);

	f32 subsurfaceWeight = mat.subsurfaceWeight;
	if (mat.textureSubsurface != 0u) {
		subsurfaceWeight *= (
			fnSampleTextureWithLod(mat.textureSubsurface, uv, lod).r
		);
	}

	f32 transmissionWeight = mat.transmissionWeight;
	if (mat.textureTransmission != 0u) {
		transmissionWeight *= (
			fnSampleTextureWithLod(mat.textureTransmission, uv, lod).r
		);
	}

	f32 thinFilmWeight = mat.thinFilmWeight;
	if (mat.textureIridescence != 0u) {
		thinFilmWeight *= fnSampleTextureWithLod(mat.textureIridescence, uv, lod).r;
	}
	f32 thinFilmThickness = mat.thinFilmThickness;
	if (mat.textureIridescenceThickness != 0u) {
		const f32 t = (
			fnSampleTextureWithLod(mat.textureIridescenceThickness, uv, lod).r
		);
		thinFilmThickness = mix(mat.thinFilmThicknessMin, mat.thinFilmThickness, t);
	}

	f32 specularRoughnessAnisotropy = mat.specularRoughnessAnisotropy;
	if (mat.textureAnisotropy != 0u) {
		specularRoughnessAnisotropy *= (
			fnSampleTextureWithLod(mat.textureAnisotropy, uv, lod).b
		);
	}

	const uint alphaMode = mat.flags & 0x3u;
	const f32 geometryOpacity = (
		alphaMode == 1u ? ((baseColor.a >= mat.alphaCutoff) ? 1.0f : 0.0f) :
		alphaMode == 2u ? baseColor.a :
		1.0f
	);

	OpenPbrMaterial pbrMat;
	pbrMat.baseColor = baseColor.rgb;
	pbrMat.baseWeight = 1.0f;
	pbrMat.baseMetallic = metallic;
	pbrMat.baseDiffuseRoughness = 0.04f;
	pbrMat.specularColor = specularColor;
	pbrMat.specularIor = mat.specularIor;
	pbrMat.specularWeight = specularWeight;
	pbrMat.specularRoughnessScale = clamp(roughness, 0.04f, 1.0f);
	pbrMat.specularRoughnessAnisotropy = specularRoughnessAnisotropy;
	pbrMat.subsurfaceWeight = subsurfaceWeight;
	pbrMat.subsurfaceColor = mat.subsurfaceColor;
	pbrMat.subsurfaceRadius = mat.subsurfaceRadius;
	pbrMat.subsurfaceRadiusScale = f32v3(1.0f, 0.5f, 0.25f);
	pbrMat.subsurfaceScatterAnisotropy = 0.0f;
	pbrMat.transmissionWeight = transmissionWeight;
	pbrMat.transmissionColor = mat.transmissionColor;
	pbrMat.transmissionDepth = mat.transmissionDepth;
	pbrMat.transmissionDispersionAbbeNumber = mat.transmissionDispersionAbbeNumber;
	pbrMat.coatWeight = coatWeight;
	pbrMat.coatColor = mat.subsurfaceColor;
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
