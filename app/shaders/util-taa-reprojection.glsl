// Temporal reprojection, history validation, and accumulation blend helpers.
// Used by temporal accumulation passes (specular, AO, etc.).
//
// Requires: layout(set=0, binding=0) uniform sampler2D vkofTextures[]

// reproject worldPos into previous-frame UV [0,1]²
// returns false if the point is behind the camera or off screen
bool taaReproject(const vec3 worldPos, const mat4 prevViewProj, out vec2 prevUv) {
	const vec4 clip = prevViewProj * vec4(worldPos, 1.0);
	if (clip.w <= 0.0) { return false; }
	const vec2 ndc = clip.xy / clip.w;
	prevUv = ndc * 0.5 + 0.5;
	return (
		all(greaterThan(prevUv, vec2(0.001)))
		&& all(lessThan(prevUv, vec2(0.999)))
	);
}

// validate that history at prevUv corresponds to the same surface as the current pixel
// prevWorldPos: fetched from the history position buffer at prevUv
// prevWorldNormal: fetched from the history normal buffer at prevUv
// cameraPos: current camera position (used to scale the spatial footprint)
bool taaValidate(
	const vec3 worldPos,
	const vec3 worldNormal,
	const vec3 prevWorldPos,
	const vec3 prevWorldNormal,
	const vec3 cameraPos
) {
	const float footprint = max(0.005 * distance(worldPos, cameraPos), 1e-3);
	const bool spatialOk = distance(worldPos, prevWorldPos) < footprint;
	const bool normalOk = dot(worldNormal, prevWorldNormal) > 0.9;
	return spatialOk && normalOk;
}

// exponential blend weight: 1.0 on first sample, converges toward 0 as count grows
// maxHistory: cap on accumulated samples (16–64 typical; pass 0.0 for unlimited)
float taaBlendAlpha(const float sampleWeight, const float historyCount, const float maxHistory) {
	const float count = (
		(maxHistory > 0.0) ? min(historyCount, maxHistory) : historyCount
	);
	return sampleWeight / max(count + sampleWeight, 1e-5);
}

// clamp history color to a k-sigma box around the 3x3 spatial neighborhood
// to suppress ghosting without hard rejection
// k: aggressiveness — 4.0 is typical for low-spp stochastic inputs
vec3 taaColorClamp(
	const vec3 histColor,
	const vec3 neighborMin,
	const vec3 neighborMax,
	const float k
) {
	const vec3 sigma = (neighborMax - neighborMin) * 0.5;
	const vec3 mean = (neighborMax + neighborMin) * 0.5;
	return clamp(histColor, mean - k * sigma, mean + k * sigma);
}
