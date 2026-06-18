#ifdef __cplusplus

#include <srat/core-types.hpp>

#define VKOFF_GLSL_BUFFER(body) struct body;
#define VKOFF_GLSL_PTR(type) alignas(8) type *

#else

#define VKOFF_GLSL_BUFFER(body) \
	layout(buffer_reference, scalar) buffer body;
#define VKOFF_GLSL_PTR(type) type

#define f32m44 mat4
#define f32v3 vec3
#define f32v4 vec4
#define u32 uint
#define i32 int

#endif

// -----------------------------------------------------------------------------

VKOFF_GLSL_BUFFER(
	PushConstantRootScene {
		u32 unused;
	}
);

VKOFF_GLSL_BUFFER(
	PushConstantRootPass {
		f32m44 viewProj;
	}
);

VKOFF_GLSL_BUFFER(
	PushConstantRootLight {
		u32 unused;
	}
);

VKOFF_GLSL_BUFFER(
	PushConstRoot,
	{
		VKOFF_GLSL_PTR(PushConstantRootScene) scene;
		VKOFF_GLSL_PTR(PushConstantRootPass) pass;
		VKOFF_GLSL_PTR(PushConstantRootLight) light;
		f32 time;
		u32 frameIndex;
	}
);
