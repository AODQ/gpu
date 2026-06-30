#ifdef __cplusplus
#pragma once
#include <mor/mor_shared.h>
#endif

#include "shared/global_pc.h"
#include "shared/light_shared.h"

#ifdef __cplusplus
using GpuFlatIndexBuffer = u64;
using GpuFlatMeshletBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuFlatIndexBuffer {
	u32 data[];
};
layout(buffer_reference, scalar) buffer GpuFlatMeshletBuffer {
	u32 data[];
};
#endif

struct GpuResolveModelIndirect {
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorMaterialBuffer) materials;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorVertexAttributeBuffer) attributes;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	VA(GpuFlatIndexBuffer) flatIndices;
	VA(GpuFlatMeshletBuffer) flatMeshlets;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
using GpuResolveModelIndirectBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuResolveModelIndirectBuffer {
	GpuResolveModelIndirect data[];
};
#endif

struct GpuResolvePC {
	u32 visibilityImageHandle;
	u32 outputImageHandle;
	u32 prevFrameNormalStorageHandle;
	u32 prevFrameDepthStorageHandle;
	u32 prevFrameSpecularStorageHandle;
	u32 prevFrameMomentStorageHandle;
	u32 frameIndex;
	u32 bluenoiseCount;
	u64 bluenoiseVa;
	u32 ptMode;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuBluenoiseHandleBuffer {
	u32 data[];
};
#ifndef RESOLVE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuResolvePC resolve;
} pc;
#endif
#endif
