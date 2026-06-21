// Requires: type aliases f32, f32v2, f32v3, f32v4
//           (from global_pc.h via scene_shared.h)
// Requires: Material struct (from mor/mor_shared.h)
// Requires: GL_EXT_nonuniform_qualifier enabled in including shader
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]

const f32 PI  = 3.14159265358979323846;
const f32 TAU = 6.28318530717958647692;
const f32 IPI = 0.31830988618379067154;

struct BsdfMaterial {
	// normal from normal map
	f32v3 modelNormal;
	f32v3 albedo;
	f32v3 emissive;
	f32 alpha;
	f32 diffuse;
	f32 fresnel;
	f32 transmittive;
};

BsdfMaterial bsdf_material_from_gltf(
	GpuMorMaterial mat,
	f32v2 uv
) {
	f32v4 baseColor = mat.baseColor;
	if (mat.textureBaseColor != 0u) {
		baseColor *= texture(vkofTextures[nonuniformEXT(mat.textureBaseColor)], uv);
	}
	f32 roughness = mat.roughness;
	f32 metallic = mat.metallic;
	if (mat.textureMetallicRoughness != 0u) {
		const f32v4 mr = texture(vkofTextures[nonuniformEXT(mat.textureMetallicRoughness)], uv);
		roughness *= mr.g;
		metallic *= mr.b;
	}
	f32v3 emissive = mat.emissive;
	if (mat.textureEmissive != 0u) {
		emissive *= texture(vkofTextures[nonuniformEXT(mat.textureEmissive)], uv).rgb;
	}
	f32v3 normal = f32v3(0.0, 0.0, 1.0);
	if (mat.textureNormal != 0u) {
		normal = (
			texture(vkofTextures[nonuniformEXT(mat.textureNormal)], uv).rgb
			* 2.0 - 1.0
		);
	}
	BsdfMaterial bmat;
	bmat.albedo = baseColor.rgb;
	bmat.modelNormal = normal;
	bmat.emissive = emissive;
	bmat.alpha = roughness;
	bmat.diffuse = 1.0 - metallic;
	bmat.fresnel = mix(0.04, 1.0, metallic);
	bmat.transmittive = 0.0;
	return bmat;
}

f32 fnGgxDistribution(f32 dotNorH, f32 a2) {
	f32 d = dotNorH * (a2 - 1.0) + 1.0;
	d = max(d, 1e-5);
	return a2 / (PI * d * d);
}

// Height-correlated Smith visibility; includes the 1/(4 dotNorWi dotNorWo) denominator.
f32 fnSmithGgxVisibility(f32 dotNorWi, f32 dotNorWo, f32 a2) {
	const f32 ggxV = dotNorWo * sqrt(dotNorWi * dotNorWi * (1.0 - a2) + a2);
	const f32 ggxL = dotNorWi * sqrt(dotNorWo * dotNorWo * (1.0 - a2) + a2);
	return 0.5 / max(ggxV + ggxL, 1e-5);
}

f32 fnSchlickFresnel(f32 dotHWi, f32 f0) {
	return f0 + (1.0 - f0) * pow(1.0 - dotHWi, 5.0);
}

// Build orthonormal basis from N and reorient v (local hemisphere sample) into world space.
f32v3 fnReorientHemisphere(f32v3 v, f32v3 N) {
	const f32v3 up = abs(N.z) < 0.999 ? f32v3(0.0, 0.0, 1.0) : f32v3(1.0, 0.0, 0.0);
	const f32v3 T = normalize(cross(up, N));
	const f32v3 B = cross(N, T);
	return normalize(T * v.x + B * v.y + N * v.z);
}

// Sample a GGX half-vector in world space. xi is a 2D uniform random sample.
f32v3 fnSampleGgxH(f32v3 N, f32 a2, f32v2 xi) {
	const f32 phi = TAU * xi.x;
	const f32 cosT = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
	const f32 sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
	const f32v3 hLocal = f32v3(sinT * cos(phi), sinT * sin(phi), cosT);
	return fnReorientHemisphere(hLocal, N);
}

// PDF of a wi/wo pair under the mixed diffuse+GGX specular model.
f32 fnBsdfPdf(f32v3 N, f32v3 wi, f32v3 wo, BsdfMaterial mat) {
	if (mat.transmittive > 0.0) {
		return 1.0;
	}
	const f32 dotNorWo = dot(N, wo);
	if (dotNorWo <= 0.0) {
		return 0.0;
	}
	f32v3 H = -wi + wo;
	{
		const f32 hlen = length(H);
		if (hlen < 1e-4) {
			return 0.0;
		}
		H /= hlen;
	}
	const f32 dotNorH = max(dot(N, H), 0.0);
	const f32 dotHWi = clamp(dot(-wi, H), 0.0, 1.0);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001);
	const f32 specPdf = fnGgxDistribution(dotNorH, a2) * dotNorH / (4.0 * dotHWi);
	const f32 diffPdf = dotNorWo * IPI;
	return mat.diffuse * diffPdf + (1.0 - mat.diffuse) * specPdf;
}

// Evaluate the BSDF (diffuse + GGX specular).
// wi: eye-to-surface (incoming).  wo: surface-to-light (outgoing).
// Returns f(wi, wo) — multiply by cos(theta_i) and light radiance for full lighting.
f32v3 fnBsdfF(f32v3 N, f32v3 wi, f32v3 wo, BsdfMaterial mat) {
	if (mat.transmittive > 0.0) {
		return IPI * mat.albedo;
	}
	const f32v3 wiV = -wi;
	const f32 dotNorWi = dot(N, wiV);
	const f32 dotNorWo = dot(N, wo);
	if (dotNorWi <= 0.0 || dotNorWo <= 0.0) {
		return f32v3(0.0);
	}
	f32v3 H = wiV + wo;
	{
		const f32 hlen = length(H);
		if (hlen < 1e-4) {
			return f32v3(0.0);
		}
		H /= hlen;
	}
	const f32 dotNorH = clamp(dot(N, H), 0.0, 1.0);
	const f32 dotHWi = clamp(dot(wiV, H), 0.0, 1.0);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001);
	const f32 D = fnGgxDistribution(dotNorH, a2);
	const f32 G = fnSmithGgxVisibility(dotNorWi, dotNorWo, a2);
	const f32 F = fnSchlickFresnel(dotHWi, mat.fresnel);
	const f32 spec = D * G * F;
	return (
		(1.0 - mat.diffuse) * spec * mat.albedo
		+ mat.diffuse * mat.albedo * IPI
	);
}

// Path-tracing sampling: requires fnSampleUniform(inout f32 seed),
// fnSampleUniform2(inout f32v2 seed2), and
// fnSampleHemisphereCos(f32v3 N, out f32 pdf, inout f32v2 seed2).
#ifdef MATERIAL_PATH_TRACING
f32v3 fnBsdfSample(
	f32v3 N, f32v3 wi, f32v3 P, BsdfMaterial mat, out f32 pdf,
	inout f32 seed, inout f32v2 seed2
) {
	if (mat.transmittive > 0.0) {
		f32v3 t = refract(wi, N, mat.transmittive);
		if (dot(t, t) < 1e-8) {
			t = reflect(wi, N);
		}
		pdf = 1.0;
		return t;
	}
	const f32 diffChance = fnSampleUniform(seed);
	if (diffChance < mat.diffuse) {
		f32v3 wo = fnSampleHemisphereCos(N, pdf, seed2);
		pdf = fnBsdfPdf(N, wi, wo, mat);
		return wo;
	}
	const f32v2 xi = fnSampleUniform2(seed2);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001);
	const f32v3 H = fnSampleGgxH(N, a2, xi);
	f32v3 wo = 2.0 * dot(-wi, H) * H - -wi;
	pdf = fnBsdfPdf(N, wi, wo, mat);
	return wo;
}
#endif
