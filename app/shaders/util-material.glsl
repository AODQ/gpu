// my shitty and mathematically flawed microfacet bsdf

#ifndef UTIL_MATERIAL_GLSL
#define UTIL_MATERIAL_GLSL

#include "util-material-gltf.glsl"
#include "util-raytrace.glsl"

#define sqr(x) ((x)*(x))

// -----------------------------------------------------------------------------
// -- microfacet bsdf utilities
// -----------------------------------------------------------------------------

float fnSampleUniform(inout float seed) {
#if skRandom == skRandomSine
	return fract(sin(seed += 0.1)*43758.5453123);
#else
	return fract(sin(seed += 0.1)*43758.5453123);
#endif
}

vec2 fnSampleUniform2(inout f32v2 seed) {
#if skRandom == skRandomSine
	return (
		  fract(sin(vec2(seed.r+=0.1,seed.g+=0.1))
		* vec2(43758.5453123,22578.1459123))
	);
#else
	seed = vec2(
		fract(sin(dot(seed, vec2(127.1, 311.7))) * 43758.5453),
		fract(sin(dot(seed, vec2(269.5, 183.3))) * 22151.0)
	);
	return seed;
#endif
}

// -----------------------------------------------------------------------------
// -- samplers
// -----------------------------------------------------------------------------

void Calculate_XY ( in f32v3 N, inout f32v3 binormal, inout f32v3 bitangent){
  binormal = abs(N.y) < 0.99f ? f32v3(0.0, 1.0, 0.0) : f32v3(1.0, 0.0, 0.0);
  binormal = normalize(cross(N, binormal));
  bitangent = cross(binormal, N);
}

vec3 Reorient_Hemisphere ( vec3 wo, vec3 N ) {
  f32v3 binormal, bitangent;
  Calculate_XY(N, binormal, bitangent);
  return bitangent*wo.x + binormal*wo.y + wo.z*N;
}

float PDF_Cosine_Hemisphere ( f32v3 wi, f32v3 N ) {
  return abs(dot(wi, N)) * IPI;
}

f32v3 To_Cartesian_T ( float theta, float phi ) {
  return f32v3(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
}
f32v3 To_Cartesian ( float cos_theta, float phi ) {
  float sin_theta = sqrt(max(0.0, 1.0 - cos_theta*cos_theta));
  return f32v3(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
}

vec3 fnSampleHemisphereCos(
	f32v3 N,
	out float pdf, inout f32v2 seed2
) {
  vec2 u = fnSampleUniform2(seed2);
  f32v3 wo = Reorient_Hemisphere(
                normalize(To_Cartesian(sqrt(u.y), TAU*u.x)), N);
  pdf = PDF_Cosine_Hemisphere(wo, N);
  return wo;
}

float PDF_Cone ( float lobe ) {
  if ( lobe < 0.001 ) return 1.0;
  return (TAU*sqr(sin(0.5*lobe)));
}

f32v3 Sample_Uniform_Cone ( float lobe, out float pdf, inout f32v2 seed2 ) {
  f32v2 u = fnSampleUniform2(seed2);
  float phi = TAU*u.x,
        cos_theta = 1.0 - u.y*(1.0 - cos(lobe));
  pdf = PDF_Cone(lobe);
  return To_Cartesian(cos_theta, phi);
}

f32v2 Normal_Sampler ( in sampler2D s, in f32v2 uv ) {
  f32v2 eps = f32v2(0.003, 0.0);
  return f32v2(length(texture(s, uv+eps.xy)) - length(texture(s, uv-eps.xy)),
                length(texture(s, uv+eps.yx)) - length(texture(s, uv-eps.yx)));
}

// -----------------------------------------------------------------------------
// -- microfacet bsdf
// -----------------------------------------------------------------------------

// GGX microfacet BSDF. wi is eye -> surface, wo leaves surface.
// view direction is -wi. H is normalize(wi + wo).

// GGX normal distribution.
f32 fnGgxDistribution(f32 dotNorH, f32 a2) {
	f32 d = dotNorH * (a2 - 1.0) + 1.0;
	d = max(d, 1e-5);
	return a2 / (PI * d * d);
}

// Height-correlated Smith visibility.
// Folds in 1/(4*dotNorWo*dotNorWi) so don't have to do that calculation
// at the end of the microfacet (something i was previously doubling lmao)
f32 fnSmithGgxVisibility(f32 dotNorWi, f32 dotNorWo, f32 a2) {
	const f32 ggxV = dotNorWo * sqrt(dotNorWi * dotNorWi * (1.0 - a2) + a2);
	const f32 ggxL = dotNorWi * sqrt(dotNorWo * dotNorWo * (1.0 - a2) + a2);
	return 0.5 / max(ggxV + ggxL, 1e-5);
}

// Schlick Fresnel (scalar F0 to match mat.fresnel).
f32 fnSchlickFresnel(f32 dotHWi, f32 f0) {
	return f0 + (1.0 - f0) * pow(1.0 - dotHWi, 5.0);
}

// Sample a GGX half-vector around N from the NDF.
f32v3 fnSampleGgxH ( f32v3 N, f32 a2, f32v2 xi ) {
	f32 phi = TAU * xi.x;
	f32 cosT = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
	f32 sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
	f32v3 hLocal = f32v3(sinT * cos(phi), sinT * sin(phi), cosT);
	return Reorient_Hemisphere(hLocal, N);
}

f32 fnBsdfPdf ( f32v3 N, f32v3 wi, f32v3 wo, BsdfMaterial mat ) {
	// Transmission is treated as a near-delta lobe; handled by the caller.
	if ( mat.transmittive > 0.0 ) {
		return 1.0;
	}
	f32 dotNorWo = dot(N, wo);
	// Below the hemisphere contributes no reflection density.
	if ( dotNorWo <= 0.0 ) {
		return 0.0;
	}
	f32v3 H = -wi + wo;
	{
		// NaN guard for degenerate half-vector
		const f32 hlen = length(H);
		if (hlen < 1e-4) {
			// degenerate pdf
			return 0.0f;
		}
		H /= hlen;
	}
	const f32 dotNorH = max(dot(N, H), 0.0);
	const f32 dotHWi = clamp(dot(-wi, H), 0.0f, 1.0f);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001f);
	// Half-vector NDF pdf converted to the outgoing direction.
	const f32 specPdf = (
		fnGgxDistribution(dotNorH, a2) * dotNorH / (4.0 * dotHWi)
	);
	const f32 diffPdf = dotNorWo * IPI;
	return mat.diffuse * diffPdf + (1.0 - mat.diffuse) * specPdf;
}

f32v3 fnBsdfSample (
	f32v3 N, f32v3 wi, f32v3 P, BsdfMaterial mat, out f32 pdf,
	inout f32 seed, inout f32v2 seed2
) {
	if ( mat.transmittive > 0.0 ) {
		f32v3 t = refract(wi, N, mat.transmittive);
		// refract returns 0 on total internal reflection; fall back.
		if ( dot(t, t) < 1e-8 ) {
			t = reflect(wi, N);
		}
		pdf = 1.0;
		return t;
	}
	f32 diffChance = fnSampleUniform(seed);
	// Diffuse lobe: sample direction, then report the shared combined pdf.
	if ( false ) {
		f32v3 wo = fnSampleHemisphereCos(N, pdf, seed2);
		pdf = fnBsdfPdf(N, wi, wo, mat);
		return wo;
	}
	// Specular lobe: sample a half-vector, reflect the view across it.
	f32v2 xi = fnSampleUniform2(seed2);
	f32 a2 = max(mat.alpha * mat.alpha, 0.0001f);
	f32v3 H = fnSampleGgxH(N, a2, xi);
	f32v3 wo = 2.0 * dot(-wi, H) * H + -wi;
	pdf = fnBsdfPdf(N, wi, wo, mat);
	return wo;
}

f32v3 fnBsdfF ( f32v3 N, f32v3 wi, f32v3 wo, BsdfMaterial mat ) {
	if ( mat.transmittive > 0.0 ) {
		return IPI * mat.albedo;
	}
	f32v3 wiV = -wi;
	f32 dotNorWi = dot(N, wiV);
	f32 dotNorWo = dot(N, wo);
	if (dotNorWi <= 0.0 || dotNorWo <= 0.0) {
		return f32v3(0.0f);
	}
	f32v3 H = wiV + wo;
	{
		// NaN guard for degenerate half-vector
		const f32 hlen = length(H);
		if (hlen < 1e-4) {
			// degenerate half-vector, no specular contribution
			return f32v3(0.0f);
		} else {
			H /= hlen;
		}
	}
	const f32 dotNorH = clamp(dot(N, H), 0.0, 1.0f);
	const f32 dotHWi = clamp(dot(wiV, H), 0.0, 1.0f);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001f);
	const f32 D = fnGgxDistribution(dotNorH, a2);
	const f32 G = fnSmithGgxVisibility(dotNorWi, dotNorWo, a2);
	const f32 F = fnSchlickFresnel(dotHWi, mat.fresnel);
	// no need for 1/(4*dotNorWo*dotNorWi), it's folded into G
	const f32 spec = D * G * F;
	const f32 diffuse = mat.diffuse;
	return (
		 (1.0 - diffuse) * spec * mat.albedo
		+ diffuse * mat.albedo * IPI
	);
}

#endif // UTIL_MATERIAL_GLSL
