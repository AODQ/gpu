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

#define VA(Type) u64

struct GpuDebugPC {
	i32 debugMode;
	i32 probeX;
	i32 probeY;
	u32 probeActive;
	f32 mipLodBias;
	u32 mipOverrideActive;
	f32 mipLodOverride;
};

struct GpuGlobalPC {
	f32 time;
	f32v3 cameraPos;
	u32 lightCount;
	u64 lightsVa;
	f32 exposure;
	u32 selectedObject;
	f32m44 viewProj;
	VA(GpuDebugPC) debug;
	u32 shadowsEnabled;
	u32 _reserved[3];
};

#ifdef __cplusplus
static_assert(sizeof(GpuGlobalPC) == 128, "GpuGlobalPC must be 128 bytes");
#else
layout(buffer_reference, scalar) buffer GpuDebugPCBuffer {
	GpuDebugPC data;
};
#endif
