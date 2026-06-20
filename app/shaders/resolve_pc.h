#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct ModelIndirectDesc {
	u64 meshlets; // mor_shared / Meshlet
	u64 materials;
};

struct ResolvePC {
	u32 visibilityImageHandle;
	u32 outputImageHandle;
	u64 models;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer ModelIndirectDescBuf {
	ModelIndirectDesc data[];
};

layout(push_constant, scalar) uniform PC {
	GlobalPC global;
	ResolvePC resolve;
} pc;
#endif
