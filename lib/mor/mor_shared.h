#ifdef __cplusplus
#pragma once
#endif

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define f32   float
#define f32v2 vec2
#define f32v3 vec3
#define f32m44 mat4
#define u32 uint
#endif
#endif

struct VertexAttr {
	f32v3 normal;
	f32v2 uv;
	f32v4 tangent;
};

struct Meshlet {
	u32 vertexOffset;
	u32 vertexCount;
	u32 triangleOffset;
	u32 triangleCount;
	u32 instanceIndex;
	u32 materialIndex;
};

struct Material {
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

struct Instance {
	f32m44 transform;
	u32 meshletOffset;
	u32 meshletCount;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer MeshletBuf     { Meshlet    data[]; };
layout(buffer_reference, scalar) buffer InstanceBuf    { Instance   data[]; };
layout(buffer_reference, scalar) buffer PositionBuf    { f32v3      data[]; };
layout(buffer_reference, scalar) buffer AttrBuf        { VertexAttr data[]; };
layout(buffer_reference, scalar) buffer MeshletVertBuf { u32        data[]; };
layout(buffer_reference, scalar) buffer MeshletTriBuf  { uint8_t    data[]; };
layout(buffer_reference, scalar) buffer MaterialBuf    { Material   data[]; };
#endif
