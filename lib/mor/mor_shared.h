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
	f32 baseMetalness;
	f32 specularRoughness;
	f32v3 emissiveColor;
	f32 emissiveLuminance;
	u32 textureBaseColor;
	u32 textureNormal;
	u32 textureMetallicRoughness;
	u32 textureEmissive;
	f32v3 specularColor;
	f32 specularWeight;
	u32 textureSpecular;
	u32 textureSpecularColor;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	u32 textureClearcoat;
	u32 textureClearcoatRoughness;
	f32v3 fuzzColor;
	f32 fuzzRoughness;
	u32 textureFuzz;
	f32 subsurfaceWeight;
	f32v3 subsurfaceColor;
	f32 subsurfaceRadius;
	u32 textureSubsurface;
	f32 transmissionWeight;
	u32 textureTransmission;
	f32v3 transmissionColor;
	f32 transmissionDepth;
	f32 thinFilmWeight;
	f32 thinFilmIor;
	f32 thinFilmThickness;
	f32 thinFilmThicknessMin;
	u32 textureIridescence;
	u32 textureIridescenceThickness;
	f32 specularRoughnessAnisotropy;
	f32 specularAnisotropyRotation;
	u32 textureAnisotropy;
	u32 textureClearcoatNormal;
	u32 textureFuzzRoughness;
	u32 textureOcclusion;
	f32 transmissionDispersionAbbeNumber;
	f32 alphaCutoff;
	f32 geometryOpacity;
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
