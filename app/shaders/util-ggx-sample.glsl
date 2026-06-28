// GGX microfacet importance sampling and STBN blue-noise helpers.
//
// Requires: GL_EXT_nonuniform_qualifier enabled in including shader
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]

// -----------------------------------------------------------------------------
// -- STBN blue-noise lookup
// -----------------------------------------------------------------------------

// fetch one scalar blue-noise value for pixel px from a pre-loaded STBN frame
float stbnScalar(const uint handle, const ivec2 px) {
	const ivec2 texSize = textureSize(vkofTextures[nonuniformEXT(handle)], 0);
	return texelFetch(vkofTextures[nonuniformEXT(handle)], px % texSize, 0).r;
}

// fetch a vec2 blue-noise value for pixel px from a pre-loaded STBN frame
vec2 stbnVec2(const uint handle, const ivec2 px) {
	const ivec2 texSize = textureSize(vkofTextures[nonuniformEXT(handle)], 0);
	return texelFetch(vkofTextures[nonuniformEXT(handle)], px % texSize, 0).rg;
}

// -----------------------------------------------------------------------------
// -- GGX microfacet importance sampling
// -----------------------------------------------------------------------------

// build orthonormal tangent frame from normal
void ggxBuildFrame(const vec3 N, out vec3 T, out vec3 B) {
	const vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	T = normalize(cross(up, N));
	B = cross(N, T);
}

// sample a GGX-distributed half-vector in world space
// u: uniform [0,1] sample
// alpha: GGX roughness (linearRoughness)
// N: surface normal
vec3 ggxSampleH(const vec2 u, const float alpha, const vec3 N) {
	const float a2 = alpha * alpha;
	const float phi = TAU * u.x;
	const float cosTheta2 = (1.0 - u.y) / max(1.0 + (a2 - 1.0) * u.y, 1e-7);
	const float cosTheta = sqrt(max(cosTheta2, 0.0));
	const float sinTheta = sqrt(max(1.0 - cosTheta2, 0.0));
	const vec3 Hlocal = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
	vec3 T, B;
	ggxBuildFrame(N, T, B);
	return normalize(T * Hlocal.x + B * Hlocal.y + N * Hlocal.z);
}

// GGX normal distribution function (D term)
float ggxD(const float alpha, const float dotNH) {
	const float a2 = alpha * alpha;
	const float d = dotNH * dotNH * (a2 - 1.0) + 1.0;
	return a2 / max(PI * d * d, 1e-7);
}

// PDF for a GGX-sampled reflection direction
// dotNH: dot(N, H) where H = normalize(reflDir - incident)
// dotVH: dot(-incident, H)
float ggxReflectPdf(const float alpha, const float dotNH, const float dotVH) {
	if (dotVH < 1e-5) { return 0.0; }
	return ggxD(alpha, dotNH) * dotNH / (4.0 * dotVH);
}

// sample a reflection direction with GGX importance sampling
// incident: direction FROM camera TOWARD surface (normalize(worldPos - camPos))
// N: surface normal
// alpha: GGX roughness (linearRoughness^2, not perceptual roughness)
// u: uniform [0,1] sample
// outPdf: output PDF for the sampled direction (0 if below-surface reflection)
vec3 ggxSampleReflect(
	const vec3 incident,
	const vec3 N,
	const float alpha,
	const vec2 u,
	out float outPdf
) {
	const vec3 H = ggxSampleH(u, alpha, N);
	const vec3 reflDir = reflect(incident, H);
	const float dotNH = max(dot(N, H), 0.0);
	const float dotVH = max(dot(-incident, H), 0.0);
	if (dot(N, reflDir) <= 0.0) {
		outPdf = 0.0;
		return reflDir;
	}
	outPdf = ggxReflectPdf(alpha, dotNH, dotVH);
	return reflDir;
}
