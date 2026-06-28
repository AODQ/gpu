// util to trace rays in the scene using ray queries

#ifndef UTIL_RAYTRACE_GLSL
#define UTIL_RAYTRACE_GLSL

#include "util-material.glsl"
#include "util-mesh.glsl"

// -----------------------------------------------------------------------------
// -- ray-query utility
// -----------------------------------------------------------------------------

struct RayQueryResult {
	float dist;
	uint modelDrawIndex;
	uint primitiveIndex;
	f32v2 barycentric;
};

RayQueryResult utilTraceRay(
	const vec3 origin,
	const vec3 dir,
	const float maxDist,
	const bool isShadowRay
) {
	rayQueryEXT rq;
	uint rayFlags = gl_RayFlagsOpaqueEXT;
	if (isShadowRay) {
		rayFlags |= gl_RayFlagsTerminateOnFirstHitEXT;
	}
	rayQueryInitializeEXT(
		rq,
		/*tlas*/ vkofTlas,
		/*flags*/ rayFlags,
		/*cullMask*/ 0xFF,
		/*origin*/ origin,
		/*minT*/ 0.00001f,
		/*dir*/ dir,
		/*maxT*/ maxDist
	);
	rayQueryProceedEXT(rq);
	if (
		rayQueryGetIntersectionTypeEXT(rq, true)
		== gl_RayQueryCommittedIntersectionNoneEXT
	) {
		return RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
	}

	// get the distance to the intersection
	RayQueryResult result;
	result.dist = rayQueryGetIntersectionTEXT(rq, true);
	result.modelDrawIndex = (
		rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true)
	);
	result.primitiveIndex = (
		rayQueryGetIntersectionPrimitiveIndexEXT(rq, true)
	);
	result.barycentric = (
		rayQueryGetIntersectionBarycentricsEXT(rq, true)
	);
	return result;
}

// -----------------------------------------------------------------------------
// -- irradiance propagation
// -----------------------------------------------------------------------------

// no contributions were found. It's still okay to continue bouncing as well
// it's okay to use the accumulated irradiance. If there's no accumulated
// irradiance and the path is terminated, then no overall contribution
#define skHitNone 0

// a direct contribution was found. The path is terminated and the
// accumulated irradiance can be used. No further bouncing.
#define skHitDirect 1

// an indirect contribution was found. The path can continue bouncing, and
// on path termination the accumulated irradiance can be used.
#define skHitIndirect 2

// the path should be terminated. Any contribution can be used however
// if there is none accumulated, then no overall contribution.
#define skHitTerminate 3

struct Ray {
	f32v3 origin;
	f32v3 dir;
};

int utilIrradiance(
	const BsdfMaterial material,
	inout f32v3 itOri,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	out f32v3 bsdfWo,
	out Ray rayWo,
	inout f32 seed,
	inout f32v2 seed2,
	out RayQueryResult bsdfRq,
	const int iteration
) {
	int returnEnum = skHitNone;
	f32 bsdfPdf;
	f32 emitPdf;

	// -- propagate irradiance throughput along bsdf
	bsdfWo = (
		fnBsdfSample(
			/*N=*/ material.normalGeometrical,
			/*wi=*/ itWi,
			/*P=*/ itOri,
			/*mat=*/ material,
			/*pdf=*/ bsdfPdf,
			seed, seed2
		)
	);
	const f32v3 irradianceThroughputPrevious = irradianceThroughput;
	const f32 dotNorWo = dot(material.normalGeometrical, bsdfWo);

	// terminate path if this produced a bad direction
	if (dotNorWo <= 0.0f) {
		irradianceAccumulator = f32v3(0.0f, 1.0f, 0.0f);
		return skHitTerminate;
	}

	irradianceThroughput *= (
		fnBsdfF(material.normalGeometrical, itWi, bsdfWo, material)
		* dotNorWo / bsdfPdf
	);

	// terminate path if throughput is too low to contribute; this also helps
	// prevent NaN accumulation
	if (length(irradianceThroughput) < 1e-4f) {
		return skHitTerminate;
	}

	// -- russian roulette
	if (iteration > 2) {
		const f32 rrChanceUnclamped = (
			max(
				irradianceThroughput.x,
				max(irradianceThroughput.y, irradianceThroughput.z)
			)
		);
		const f32 rrChance = clamp(rrChanceUnclamped, 0.0f, 1.0f);
		if (fnSampleUniform(seed) > rrChance) {
			return skHitTerminate;
		}
		irradianceThroughput /= rrChance;
	}

	// -- indirect irradiance propagation (bsdf path)
	{
		bsdfRq = (
			utilTraceRay(
				// TODO below replace with the geometrical normal
				itOri + material.normalGeometrical * 0.01f,
				bsdfWo,
				/*maxDist=*/999999.0f,
				/*isShadowRay=*/false
			)
		);

		// miss shader hits the sky for now
		if (bsdfRq.dist <= 0.0f) {
			// -- sky contribution
			irradianceAccumulator += (
				f32v3(0.2f) * irradianceThroughput
			);
			return skHitDirect;
		}

		// TODO apply bsdf contribution to emitter when hitting a light 
		// if (fnValidEmitter(bsdfLightIndex, bsdfWo)) {
		// 	lightIndex = bsdfLightIndex;
		// 	const f32 emitterDist = hitWo.x;

		// 	// use NEE sampling strategy for PDF so that it can be combined
		// 	// with direct irradiance
		// 	const f32 emitPdf = (
		// 		  fnEmitPdf(lightIndex, bsdfWo, emitterDist)
		// 		/ skLightsInSceneInclSky
		// 	);

		// 	// balance heuristic weight for BSDF strategy
		// 	const f32 misWeightDenom = bsdfPdf + emitPdf;
		// 	const f32 misWeight = (
		// 		misWeightDenom > 0.0f ? bsdfPdf / misWeightDenom : 0.0f
		// 	);

		// 	// throughput contains the bounce's bsdfF.
		// 	// arriving light is just emission, throughput and MIS weight.
		// 	const f32v3 radiance = (
		// 		  fnEmitterF(lightIndex, bsdfWo)
		// 		* irradianceThroughput
		// 		* misWeight
		// 	);
		// 	irradianceAccumulator += radiance;
		// 	returnEnum = skHitIndirect;
		// }
	}

	// NEE direct radiance isn't implemented yet
	// // -- direct irradiance (NEE path)
	// {
	// 	// if hit a light, then use the NEE sampling strategy on the same
	// 	// light. otherwise just grab random light index
	// 	if (lightIndex == -1) {
	// 		lightIndex = int(fnSampleEmitter(seed, emitPdf));
	// 	}

	// 	const f32v3 emitterPos = fnEmitterSample(lightIndex, itNor, seed2);
	// 	const f32v3 emitterWo = normalize(emitterPos - itOri);
	// 	const f32 emitterDist = length(emitterPos - itOri);

	// 	f32v2 lightMarch = f32v2(999999.0f, -1.0f);

	// 	// only march if the emitter is in the same hemisphere as the normal
	// 	const bool sameHemisphere = dot(itNor, emitterWo) > 0.0f;
	// 	if (sameHemisphere) {
	// 		lightMarch = (
	// 			fnSceneMarch(
	// 				Ray(itOri + itNor*0.01f, emitterWo),
	// 				skSceneMarchLq
	// 			)
	// 		);
	// 	}

	// 	bool visible = (
	// 			(LIGHT_IDX(lightMarch.y) == lightIndex)
	// 		&& fnValidEmitter(lightIndex, emitterWo)
	// 	);
	// 	if (sameHemisphere && lightIndex == skLightIndexSky) {
	// 		// sky ray must miss otherwise non-visible to sky
	// 		 visible = (lightMarch.y == -1.0f);
	// 	}
	// 	if (visible) {
	// 		const f32 pdf = (
	// 			  fnEmitPdf(lightIndex, emitterWo, emitterDist)
	// 			/ skLightsInSceneInclSky
	// 		);
	// 		// reject degenerate PDFs to prevent NaNs
	// 		if (pdf <= 0.0f) {
	// 			return returnEnum;
	// 		}

	// 		// compute MISweight
	// 		const f32 bsdfPdf = fnBsdfPdf(itNor, itWi, emitterWo, mat);
	// 		const f32 misWeight = pdf / (pdf + bsdfPdf);

	// 		const f32 dotNorWo = max(dot(itNor, emitterWo), 0.0f);
	// 		const f32v3 radiance = (
	// 			  fnEmitterF(lightIndex, emitterWo)
	// 			* fnBsdfF(itNor, itWi, emitterWo, mat)
	// 			* dotNorWo
	// 			/ pdf
	// 		);
	// 		// use previous irradiance throughput as that one doesn't have the
	// 		// bsdfF of the future bounce
	// 		irradianceAccumulator += (
	// 			radiance * irradianceThroughputPrevious * misWeight
	// 		);
	// 		returnEnum = skHitDirect;
	// 	}
	// }

	return returnEnum;
}

int utilIrradiancePropagate(
	inout BsdfMaterial material,
	inout f32v3 itOri,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	inout f32 seed,
	inout f32v2 seed2,
	const int iteration
) {
	// -- compute irradiance on surface
	f32v3 bsdfWo;
	Ray rayWo;
	RayQueryResult bsdfRq;
	int returnEnum = (
		utilIrradiance(
			material,
			itOri,
			itWi,
			irradianceThroughput,
			irradianceAccumulator,
			bsdfWo,
			rayWo,
			seed, seed2,
			bsdfRq,
			iteration
		)
	);

	// -- update the material and ray for the next bounce
	if (returnEnum == skHitIndirect || returnEnum == skHitNone) {
		const UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				bsdfRq.modelDrawIndex,
				bsdfRq.primitiveIndex,
				bsdfRq.barycentric
			)
		);
		const f32v3 bitangent = (
			cross(
				rqData.normal,
				rqData.tangent.xyz
			)
			* rqData.tangent.w
		);
		const mat3 tbn = (
			mat3(
				normalize(rqData.tangent.xyz),
				normalize(bitangent),
				normalize(rqData.normal)
			)
		);
		const BsdfMaterial rqBsdfMaterial = (
			bsdfMaterialFromGltfWithLod(
				/*material=*/rqData.material,
				/*normalGeometrical=*/rqData.normalGeometrical,
				/*tbn=*/ tbn,
				/*uv=*/rqData.uv,
				/*lod=*/ uint(sqrt(material.alpha) * 10.0f + 0.5f)
			)
		);
		material = rqBsdfMaterial;
		itOri = itOri + bsdfWo * bsdfRq.dist;
		itWi = -bsdfWo;
	}

	return returnEnum;

	// if path is terminated, then return early
	// if (returnEnum == skHitTerminate) {
	// 	return skHitTerminate;
	// }
}

f32v4 utilIrradiancePropagationEntry(
	BsdfMaterial originalMaterial,
	f32v3 origin,
	f32v3 wi,
	const uint propagationDepth
) {
	f32v3 irradianceThroughput = f32v3(1.0);
	f32v3 irradianceAccumulator = f32v3(0.0);

	f32 seed = (
		float(gl_WorkGroupID.x * gl_WorkGroupID.y * gl_WorkGroupID.z)
		+ float(gl_LocalInvocationIndex)
		+ pc.global.time
	);
	f32v2 seed2 = f32v2(seed, seed * 1.61803398875);

	int hit = 0;
	for (int i = 0; i < propagationDepth; ++i) {
		int returnEnum = (
			utilIrradiancePropagate(
				originalMaterial,
				origin,
				wi,
				irradianceThroughput,
				irradianceAccumulator,
				seed, seed2,
				i
			)
		);

		if (returnEnum == skHitNone) {
			continue;
		}

		if (returnEnum == skHitIndirect) {
			hit = 1;
			continue;
		}

		if (returnEnum == skHitDirect) {
			hit = 1;
			break;
		}

		if (returnEnum == skHitTerminate) {
			break;
		}
	}

	// debug for NaN hunts
	if (any(isnan(irradianceAccumulator)) && hit > 0) {
		return f32v4(1.0f, 0.0f, 0.0f, 1.0f);
	}

	return f32v4(irradianceAccumulator, float(hit));
}

#endif // UTIL_RAYTRACE_GLSL
