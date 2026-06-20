#version 450
#extension GL_EXT_mesh_shader                      : require
#extension GL_EXT_buffer_reference                 : require
#extension GL_EXT_scalar_block_layout              : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_GOOGLE_include_directive             : require

#include "scene_shared.h"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in flat int inMaterialIndex;

layout(location = 0) out vec4 outColor;

void main() {
	MaterialBuf materialBuf = MaterialBuf(pc.draw.materials);
	const Material mat = materialBuf.data[inMaterialIndex];
	const vec3 n = normalize(inNormal);
	const vec3 sun = normalize(vec3(0.4, 1.0, 0.6));
	const float diff = max(dot(n, sun), 0.0);
	outColor = vec4(vec3(0.15 + diff * 0.85), 1.0) * mat.baseColor;
}
