// my shitty and mathematically flawed microfacet bsdf

#ifndef UTIL_MATERIAL_GLSL
#define UTIL_MATERIAL_GLSL

#include "util-material-gltf.glsl"
#include "util-raytrace.glsl"
#include "util-material-openpbr-tables.glsl"

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

// Schlick Fresnel with vec3 F0 for coloured specular.
f32v3 fnSchlickFresnel(const f32 dotHV, const f32v3 f0) {
	return f0 + (f32v3(1.0f) - f0) * pow(1.0f - dotHV, 5.0f);
}

// One-sided Smith G1 masking term.
f32 fnSmithG1(const f32 dotNorV, const f32 a2) {
	return 2.0f * dotNorV / (dotNorV + sqrt(a2 + (1.0f - a2) * dotNorV * dotNorV));
}

// Kulla-Conty multi-scatter energy compensation.
// E(mu, alpha): GGX white-furnace directional albedo.
// mu   axis: (i+0.5)/16   i in 0..15  (grazing->normal)
// alpha axis: (i/15)^2    i in 0..15  (smooth->rough)
const f32 kETable[256] = f32[256](
	1.000000f, 0.988071f, 0.905022f, 0.896458f, 0.922060f, 0.940172f, 0.951033f, 0.953774f, 0.952107f, 0.947958f, 0.942332f, 0.934545f, 0.925498f, 0.914913f, 0.903202f, 0.890433f,
	1.000000f, 0.998627f, 0.977963f, 0.924440f, 0.889060f, 0.886391f, 0.891276f, 0.894988f, 0.894870f, 0.886240f, 0.876163f, 0.858471f, 0.840526f, 0.817747f, 0.794919f, 0.769773f,
	1.000000f, 0.999548f, 0.992477f, 0.961348f, 0.915559f, 0.883532f, 0.870649f, 0.864415f, 0.858654f, 0.846603f, 0.828853f, 0.808554f, 0.783298f, 0.752251f, 0.720449f, 0.687631f,
	1.000000f, 0.999813f, 0.996053f, 0.978904f, 0.942906f, 0.901699f, 0.871593f, 0.852289f, 0.836613f, 0.816990f, 0.796714f, 0.768985f, 0.735167f, 0.701014f, 0.663193f, 0.624795f,
	1.000000f, 0.999856f, 0.997469f, 0.987360f, 0.960689f, 0.920756f, 0.881842f, 0.849685f, 0.824418f, 0.800921f, 0.770320f, 0.738004f, 0.701490f, 0.660073f, 0.619153f, 0.572905f,
	1.000000f, 0.999878f, 0.998320f, 0.991305f, 0.972025f, 0.937171f, 0.895881f, 0.857017f, 0.822157f, 0.787703f, 0.755183f, 0.715447f, 0.670402f, 0.624560f, 0.579233f, 0.532599f,
	1.000000f, 0.999895f, 0.998738f, 0.993239f, 0.978505f, 0.949633f, 0.908792f, 0.865747f, 0.825601f, 0.782747f, 0.741589f, 0.698455f, 0.648912f, 0.597526f, 0.546716f, 0.495422f,
	1.000000f, 0.999962f, 0.999180f, 0.994909f, 0.982861f, 0.958860f, 0.921529f, 0.876808f, 0.828128f, 0.781164f, 0.733090f, 0.682346f, 0.628276f, 0.576641f, 0.520170f, 0.461832f,
	1.000000f, 0.999967f, 0.999143f, 0.996110f, 0.986222f, 0.965474f, 0.931820f, 0.886617f, 0.837798f, 0.783723f, 0.729957f, 0.672340f, 0.615165f, 0.555356f, 0.493722f, 0.438420f,
	1.000000f, 0.999950f, 0.999315f, 0.996472f, 0.988006f, 0.971015f, 0.939302f, 0.896990f, 0.844288f, 0.786255f, 0.726462f, 0.663577f, 0.601563f, 0.537290f, 0.471909f, 0.415300f,
	1.000000f, 0.999953f, 0.999497f, 0.997208f, 0.990300f, 0.974460f, 0.947030f, 0.905052f, 0.852056f, 0.790519f, 0.726396f, 0.657019f, 0.587188f, 0.523369f, 0.454441f, 0.391267f,
	1.000000f, 0.999994f, 0.999558f, 0.997375f, 0.991503f, 0.977419f, 0.951576f, 0.912996f, 0.860437f, 0.796573f, 0.726529f, 0.655100f, 0.581643f, 0.509683f, 0.437398f, 0.372621f,
	1.000000f, 0.999956f, 0.999568f, 0.997899f, 0.992178f, 0.980203f, 0.957207f, 0.919148f, 0.867690f, 0.802882f, 0.730433f, 0.650576f, 0.571347f, 0.495317f, 0.423144f, 0.354806f,
	1.000000f, 0.999978f, 0.999652f, 0.997975f, 0.992989f, 0.982050f, 0.960359f, 0.924918f, 0.875282f, 0.807544f, 0.732619f, 0.649359f, 0.564623f, 0.483091f, 0.408167f, 0.339617f,
	1.000000f, 0.999969f, 0.999708f, 0.998276f, 0.993703f, 0.983939f, 0.964226f, 0.930900f, 0.879820f, 0.816623f, 0.736827f, 0.649661f, 0.560940f, 0.477082f, 0.397979f, 0.324305f,
	1.000000f, 0.999960f, 0.999734f, 0.998373f, 0.994381f, 0.985219f, 0.966582f, 0.935156f, 0.886950f, 0.821114f, 0.742601f, 0.650476f, 0.559640f, 0.466482f, 0.385075f, 0.313566f
);
const f32 kEavgTable[16] = f32[16](
	1.000000f, 0.999880f, 0.998510f, 0.994123f, 0.984244f, 0.967015f, 0.940452f, 0.904163f, 0.858230f, 0.802818f, 0.741311f, 0.673700f, 0.604681f, 0.536525f, 0.470419f, 0.408946f
);

f32 fnGgxAlbedo(const f32 mu, const f32 alpha) {
	const f32 ai = clamp(sqrt(alpha) * 15.0f, 0.0f, 14.999f);
	const f32 mi = clamp(mu * 16.0f - 0.5f, 0.0f, 14.999f);
	const int ai0 = int(ai); const int ai1 = min(ai0 + 1, 15);
	const int mi0 = int(mi); const int mi1 = min(mi0 + 1, 15);
	const f32 ta = fract(ai); const f32 tm = fract(mi);
	return (
		mix(
			mix(kETable[mi0*16 + ai0], kETable[mi0*16 + ai1], ta),
			mix(kETable[mi1*16 + ai0], kETable[mi1*16 + ai1], ta),
			tm
		)
	);
}

f32 fnGgxAlbedoAvg(const f32 alpha) {
	const f32 ai = clamp(sqrt(alpha) * 15.0f, 0.0f, 14.999f);
	const int ai0 = int(ai);
	return mix(kEavgTable[ai0], kEavgTable[min(ai0 + 1, 15)], fract(ai));
}

// GGX VNDF sampling (Heitz 2018);
// guarantees dot(wi, H) < 0 so reflect(wi, H) stays above surface.
f32v3 fnSampleGgxVndf(const f32v3 N, const f32v3 wi, const f32 alpha, const f32v2 xi) {
	f32v3 binormal, bitangent;
	Calculate_XY(N, binormal, bitangent);
	const f32v3 wiLocal = f32v3(dot(wi, bitangent), dot(wi, binormal), dot(wi, N));
	const f32v3 V = -wiLocal;
	const f32v3 Vh = normalize(f32v3(alpha * V.x, alpha * V.y, V.z));
	const f32 lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	const f32v3 T1 = (
		lensq > 1e-7f
		? f32v3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrt(lensq))
		: f32v3(1.0f, 0.0f, 0.0f)
	);
	const f32v3 T2 = cross(Vh, T1);
	const f32 r = sqrt(xi.x);
	const f32 phi = TAU * xi.y;
	const f32 t1 = r * cos(phi);
	f32 t2 = r * sin(phi);
	const f32 s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * sqrt(max(1.0f - t1 * t1, 0.0f)) + s * t2;
	const f32v3 Nh = (
		t1 * T1 + t2 * T2
		+ sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh
	);
	const f32v3 Hlocal = normalize(f32v3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
	return Hlocal.x * bitangent + Hlocal.y * binormal + Hlocal.z * N;
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
	const f32 dotNorH = max(dot(N, H), 0.0f);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001f);
	const f32 dotNorWi = max(-dot(N, wi), 1e-5f);
	const f32 G1 = fnSmithG1(dotNorWi, a2);
	const f32 specPdf = G1 * fnGgxDistribution(dotNorH, a2) / (4.0f * dotNorWi);
	const f32 diffPdf = dotNorWo * IPI;
	const f32v3 Fview = fnSchlickFresnel(dotNorWi, mat.f0);
	const f32 lumF = dot(Fview, f32v3(0.2126f, 0.7152f, 0.0722f));
	return lumF * specPdf + (1.0f - lumF) * diffPdf;
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
	const f32 dotNorWi = max(-dot(N, wi), 1e-5f);
	const f32v3 Fview = fnSchlickFresnel(dotNorWi, mat.f0);
	const f32 lumF = dot(Fview, f32v3(0.2126f, 0.7152f, 0.0722f));
	const f32 diffChance = fnSampleUniform(seed);
	if (diffChance >= lumF) {
		const f32v3 wo = fnSampleHemisphereCos(N, pdf, seed2);
		pdf = fnBsdfPdf(N, wi, wo, mat);
		return wo;
	}
	const f32v2 xi = fnSampleUniform2(seed2);
	const f32 alpha = max(mat.alpha, 0.01f);
	const f32v3 H = fnSampleGgxVndf(N, wi, alpha, xi);
	const f32v3 wo = wi - 2.0f * dot(wi, H) * H;
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
	const f32 dotNorH = clamp(dot(N, H), 0.0f, 1.0f);
	const f32 dotHWi = clamp(dot(wiV, H), 0.0f, 1.0f);
	const f32 a2 = max(mat.alpha * mat.alpha, 0.0001f);
	const f32 D = fnGgxDistribution(dotNorH, a2);
	const f32 G = fnSmithGgxVisibility(dotNorWi, dotNorWo, a2);
	const f32v3 F = fnSchlickFresnel(dotHWi, mat.f0);
	// no need for 1/(4*dotNorWo*dotNorWi), it's folded into G
	const f32v3 spec = D * G * F;
	const f32v3 diffuseColor = mat.albedo * mat.diffuse;
	const f32v3 result = spec + (f32v3(1.0f) - F) * diffuseColor * IPI;
	const f32 Eo = fnGgxAlbedo(dotNorWo, mat.alpha);
	const f32 Ei = fnGgxAlbedo(dotNorWi, mat.alpha);
	const f32 Eavg = fnGgxAlbedoAvg(mat.alpha);
	const f32 fMs = (1.0f - Eo) * (1.0f - Ei) / (PI * max(1.0f - Eavg, 1e-5f));
	const f32v3 Favg = mat.f0 + (f32v3(1.0f) - mat.f0) / 21.0f;
	const f32v3 fMsColor = (
		Favg * Favg * Eavg
		/ max(f32v3(1.0f) - Favg * (f32v3(1.0f) - Eavg), f32v3(1e-5f))
	);
	return result + fMs * fMsColor;
}

#endif // UTIL_MATERIAL_GLSL
