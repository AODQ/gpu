// A-trous wavelet spatial denoiser with edge-stopping weights.
// Run as 2–4 passes with increasing passIndex for progressive filtering.
//
// Requires: GL_EXT_nonuniform_qualifier enabled in including shader
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]

// A-trous 5-tap cross filter pass.
//
// inputHandle:  sampler2D of accumulated RGB color (RGBA16F; alpha unused)
// posHandle:    sampler2D of world-space positions (RGB; history position buffer)
// norHandle:    sampler2D of world-space normals   (RGB; history normal buffer)
// momentHandle: sampler2D of moment data (R=M1, G=M2, B=variance)
// px:           current pixel coordinate
// screenSize:   render resolution in pixels
// passIndex:    iteration index 0..N; controls kernel step size
//
// returns: filtered RGB radiance
vec3 taaAtrousFilter(
	const uint inputHandle,
	const uint posHandle,
	const uint norHandle,
	const uint momentHandle,
	const ivec2 px,
	const ivec2 screenSize,
	const int passIndex
) {
	const vec2 uvC = (vec2(px) + 0.5) / vec2(screenSize);
	const vec4 colC = texture(vkofTextures[nonuniformEXT(inputHandle)], uvC);
	const vec3 posC = texture(vkofTextures[nonuniformEXT(posHandle)], uvC).xyz;
	const vec3 norC = texture(vkofTextures[nonuniformEXT(norHandle)], uvC).xyz;
	const float variance = (
		max(texture(vkofTextures[nonuniformEXT(momentHandle)], uvC).b, 0.001)
	);

	const float sigmaL = 4.0;
	const float sigmaP = 1.0;
	const float sigmaN = 128.0;

	const float varianceScale = clamp(variance / 0.1, 0.5, 2.0);
	const int step = int(mix(1.0, float(1 << passIndex), varianceScale));

	const ivec2 offsets[5] = ivec2[](
		ivec2(0, 0),
		ivec2(1, 0),
		ivec2(-1, 0),
		ivec2(0, 1),
		ivec2(0, -1)
	);

	const float lumC = dot(colC.rgb, vec3(0.2126, 0.7152, 0.0722));

	vec3 sumRgb = vec3(0.0);
	float sumWeight = 0.0;

	for (int i = 0; i < 5; ++i) {
		const ivec2 npx = px + offsets[i] * step;
		if (
			any(lessThan(npx, ivec2(0)))
			|| any(greaterThanEqual(npx, screenSize))
		) {
			continue;
		}

		const vec2 uvQ = (vec2(npx) + 0.5) / vec2(screenSize);
		const vec4 colQ = texture(vkofTextures[nonuniformEXT(inputHandle)], uvQ);
		const vec3 posQ = texture(vkofTextures[nonuniformEXT(posHandle)], uvQ).xyz;
		const vec3 norQ = texture(vkofTextures[nonuniformEXT(norHandle)], uvQ).xyz;

		const float rawLumQ = dot(colQ.rgb, vec3(0.2126, 0.7152, 0.0722));
		const float maxLum = 10.0;
		const vec3 colQclamped = colQ.rgb * min(1.0, maxLum / max(rawLumQ, 1e-4));
		const float lumQ = dot(colQclamped, vec3(0.2126, 0.7152, 0.0722));

		const float wN = pow(max(dot(norC, norQ), 0.0), sigmaN);
		const float wP = exp(-distance(posC, posQ) / (sigmaP + 1e-4));
		const float wL = exp(-abs(lumC - lumQ) / (sigmaL * sqrt(variance) + 1e-4));
		const float w = wN * wP * wL;

		sumRgb += colQclamped * w;
		sumWeight += w;
	}

	return sumRgb / max(sumWeight, 1e-4);
}
