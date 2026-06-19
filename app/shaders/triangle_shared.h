#pragma once

#include "shared/global_pc.h"

#ifdef __cplusplus
#include <srat/core-math.hpp>
#endif

struct TriangleDrawPC {
	f32v4 color;
};

#ifndef __cplusplus
layout(push_constant, scalar) uniform PC {
	GlobalPC       global;
	TriangleDrawPC draw;
} pc;
#endif
