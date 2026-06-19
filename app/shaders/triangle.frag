#version 450
#extension GL_EXT_scalar_block_layout  : require
#extension GL_GOOGLE_include_directive : require

#include "triangle_shared.h"

layout(location = 0) out vec4 outColor;

void main() {
	float pulse = 0.5 + 0.5 * sin(pc.global.time * 2.0);
	outColor = vec4(pc.draw.color.rgb * pulse, pc.draw.color.a);
}
