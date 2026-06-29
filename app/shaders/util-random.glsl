#ifndef UTIL_RANDOM_GLSL
#define UTIL_RANDOM_GLSL

float fnSampleUniform(inout float seed) {
	return fract(sin(seed += 0.1)*43758.5453123);
}

vec2 fnSampleUniform2(inout f32v2 seed) {
	seed = vec2(
		fract(sin(dot(seed, vec2(127.1, 311.7))) * 43758.5453),
		fract(sin(dot(seed, vec2(269.5, 183.3))) * 22151.0)
	);
	return seed;
}

float fnSampleBluenoise(u64 va, u32 count, u32 frameIndex, ivec2 coord) {
	if (count == 0u || va == 0ul) { return 0.5f; }
	const u32 handle = GpuBluenoiseHandleBuffer(va).data[frameIndex % count];
	const vec2 uv = (vec2(coord % ivec2(128)) + 0.5f) / 128.0f;
	return texture(vkofTextures[nonuniformEXT(handle)], uv).r;
}

#endif // UTIL_RANDOM_GLSL
