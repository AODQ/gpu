#ifdef __cplusplus
#pragma once
#endif

#ifdef __cplusplus
#include <srat/core-types.hpp>
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

struct GlobalPC {
	f32 time;
	f32 _pad[31];
};

#ifdef __cplusplus
static_assert(sizeof(GlobalPC) == 128, "GlobalPC must be 128 bytes");
#endif
