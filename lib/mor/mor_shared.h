#ifdef __cplusplus
#pragma once
#endif

#define MOR_DEBUG_VIEW_MESHLET_ID 1
#define MOR_DEBUG_VIEW_MODEL_ID 2
#define MOR_DEBUG_VIEW_TRIANGLE_ID 3
#define MOR_DEBUG_VIEW_MIP_HEATMAP 4
#define MOR_DEBUG_VIEW_ALBEDO 5
#define MOR_DEBUG_VIEW_WORLD_NORMAL 6
#define MOR_DEBUG_VIEW_ROUGHNESS 7
#define MOR_DEBUG_VIEW_METALLIC 8
#define MOR_DEBUG_VIEW_EMISSIVE 9
#define MOR_DEBUG_VIEW_UV 10

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define f32   float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#endif
#endif

struct GpuMorVertexAttribute {
	f32v3 normal;
	f32v2 uv;
	f32v4 tangent;
};

struct GpuMorMeshlet {
	u32 vertexOffset;
	u32 vertexCount;
	u32 triangleOffset;
	u32 triangleCount;
	u32 instanceIndex;
	u32 materialIndex;
};

struct GpuMorMaterial {
	f32v4 baseColor;
	f32 metallic;
	f32 roughness;
	f32v3 emissive;
	u32 textureBaseColor;
	u32 textureNormal;
	u32 textureMetallicRoughness;
	u32 textureEmissive;
	u32 flags;
};

struct GpuMorInstance {
	f32m44 transform;
	u32 meshletOffset;
	u32 meshletCount;
};

#ifdef __cplusplus
using GpuMorMeshletBuffer = u64;
using GpuMorInstanceBuffer = u64;
using GpuMorPositionBuffer = u64;
using GpuMorVertexAttributeBuffer = u64;
using GpuMorMeshletVertBuffer = u64;
using GpuMorMeshletTriBuffer = u64;
using GpuMorMaterialBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuMorMeshletBuffer {
	GpuMorMeshlet data[];
};
layout(buffer_reference, scalar) buffer GpuMorInstanceBuffer {
	GpuMorInstance data[];
};
layout(buffer_reference, scalar) buffer GpuMorPositionBuffer {
	f32v3 data[];
};
layout(buffer_reference, scalar) buffer GpuMorVertexAttributeBuffer {
	GpuMorVertexAttribute data[];
};
layout(buffer_reference, scalar) buffer GpuMorMeshletVertBuffer {
	u32 data[];
};
layout(buffer_reference, scalar) buffer GpuMorMeshletTriBuffer {
	uint8_t data[];
};
layout(buffer_reference, scalar) buffer GpuMorMaterialBuffer {
	GpuMorMaterial data[];
};
#endif
