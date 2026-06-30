#ifndef UTIL_RANDOM_GLSL
#define UTIL_RANDOM_GLSL

u32 fnPcgHash(u32 s) {
	s = s * 747796405u + 2891336453u;
	s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (s >> 22u) ^ s;
}

float fnSampleUniform(inout float seed) {
	u32 s = fnPcgHash(floatBitsToUint(seed));
	seed = uintBitsToFloat((s >> 9u) | 0x3F800000u) - 1.0f;
	return seed;
}

f32v2 fnSampleUniform2(inout f32v2 seed) {
	u32 s = fnPcgHash(floatBitsToUint(seed.x) ^ floatBitsToUint(seed.y));
	u32 s2 = fnPcgHash(s);
	seed.x = uintBitsToFloat((s >> 9u) | 0x3F800000u) - 1.0f;
	seed.y = uintBitsToFloat((s2 >> 9u) | 0x3F800000u) - 1.0f;
	return seed;
}

float fnSampleBluenoise(u64 va, u32 count, u32 frameIndex, ivec2 coord) {
	if (count == 0u || va == 0ul) { return 0.5f; }
	const u32 handle = GpuBluenoiseHandleBuffer(va).data[frameIndex % count];
	const vec2 uv = (vec2(coord % ivec2(128)) + 0.5f) / 128.0f;
	return texture(vkofTextures[nonuniformEXT(handle)], uv).r;
}

#endif // UTIL_RANDOM_GLSL
