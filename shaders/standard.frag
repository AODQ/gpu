#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main() {
	const float dotWoNor = (
		max(dot(normalize(inNormal), normalize(vec3(0.4f, 0.8f, 0.4f))), 0.1f)
	);

	outColor = vec4(inColor * dotWoNor, 1.0f);
}
