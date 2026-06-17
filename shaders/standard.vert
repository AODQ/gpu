#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(push_constant, scalar) uniform PushConstants {
	mat4 viewProj;
	vec3 cameraForward;
	vec3 cameraPosWorld;
	vec4 drawTintColor;
	uint64_t instanceCount;
	BufferVertexAttribute vertexBufferVas;
	BufferInstance instanceBufferVa;
	BufferModel modelBufferVa;
	BufferCommand commandBufferVa;
	BufferCommandCount commandCountBufferVa;
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColor;

void main() {
	const Instance instance = pc.instanceBufferVa.instances[gl_InstanceIndex];
	const VertexAttribute attr = pc.vertexBufferVas.vertexAttributes[gl_VertexIndex];
	gl_Position = pc.viewProj * instance.transform * vec4(attr.position, 1.0);
	outNormal = attr.normal;
	outColor = mix(instance.color.rgb, pc.drawTintColor.rgb, pc.drawTintColor.a);
}
