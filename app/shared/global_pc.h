#ifdef __cplusplus
#pragma once
#endif

#ifdef __cplusplus
#include <srat/core-math.hpp>
#else
#define f32    float
#define f32v2  vec2
#define f32v3  vec3
#define f32v4  vec4
#define f32m44 mat4
#define u32    uint
#define u64    uint64_t
#define i32    int
#endif

#define BDA(Type) u64

struct GpuGlobalPC {
	f32 time;
	i32 probeX;
	i32 probeY;
	u32 probeActive;
	u32 debugMode;
	f32v3 cameraPos;
	f32 _pad[8];
	f32m44 viewProj;
};

#ifdef __cplusplus
static_assert(sizeof(GpuGlobalPC) == 128, "GpuGlobalPC must be 128 bytes");
#endif
