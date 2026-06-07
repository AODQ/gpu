struct VertexAttribute {
	vec3 position;
	vec3 normal;
};

struct Instance {
	mat4 transform;
	vec4 color;
	uint modelIndex;
};

struct Model {
	vec4 boundingSphere;
	uint indexOffset;
	uint indexCount;
	int vertexOffset;
};

struct CommandDrawIndexed {
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	int vertexOffset;
	uint firstInstance;
};

layout(buffer_reference, scalar) readonly buffer BufferInstance {
	Instance instances[];
};

layout(buffer_reference, scalar) readonly buffer BufferModel {
	Model models[];
};

layout(buffer_reference, scalar) buffer BufferCommand {
	CommandDrawIndexed commands[];
};

layout(buffer_reference, scalar) buffer BufferCommandCount {
	uint commandCount;
};

layout(buffer_reference, scalar) readonly buffer BufferVertexAttribute {
	VertexAttribute vertexAttributes[];
};
