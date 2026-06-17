#version 450
#extension GL_EXT_mesh_shader : require

layout(location = 0) in vec3 inNormal;
layout(location = 1) perprimitiveEXT flat in uint inSphereIdx;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(inNormal * 0.5 + vec3(0.5), 1.0);
}
