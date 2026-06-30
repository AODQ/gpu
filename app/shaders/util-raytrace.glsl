// util to trace rays in the scene using ray queries

#ifndef UTIL_RAYTRACE_GLSL
#define UTIL_RAYTRACE_GLSL

#include "util-mesh.glsl"

#include "util-material-openpbr.glsl"

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
	const bool isShadowRay,
	const bool testAlphaCutoff
) {
	rayQueryEXT rq;
	uint rayFlags = 0;
	if (!testAlphaCutoff) {
		rayFlags |= gl_RayFlagsOpaqueEXT;
	}
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

	RayQueryResult result;
	while (rayQueryProceedEXT(rq)) {
		if (
			rayQueryGetIntersectionTypeEXT(rq, false)
			!= gl_RayQueryCandidateIntersectionTriangleEXT
		) {
			return RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
		}

		// get the distance to the intersection
		result.dist = rayQueryGetIntersectionTEXT(rq, false);
		result.modelDrawIndex = (
			rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false)
		);
		result.primitiveIndex = (
			rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)
		);
		result.barycentric = (
			rayQueryGetIntersectionBarycentricsEXT(rq, false)
		);
		if (!testAlphaCutoff) {
			return result;
		}

		const UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				result.modelDrawIndex,
				result.primitiveIndex,
				result.barycentric
			)
		);
		f32v3 modelNormal;
		const OpenPbrMaterial mat = (
			openPbrMaterialFromGltfWithLod(
				/*material=*/rqData.material,
				/*uv=*/rqData.uv,
				/*lod=*/ 0,
				/*modelNormal=*/modelNormal
			)
		);

		if (mat.geometryOpacity > 0.01f) {
			rayQueryConfirmIntersectionEXT(rq);
			if (isShadowRay) {
				rayQueryTerminateEXT(rq);
			}
		}
	}

	if (
		rayQueryGetIntersectionTypeEXT(rq, true)
		== gl_RayQueryCommittedIntersectionTriangleEXT
	) {
		return result;
	}

	return RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
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
	const MaterialTableHandles tables,
	const OpenPbrMaterial material,
	inout f32v3 itOri,
	inout f32v3 itNormalWorld,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	out f32v3 bsdfWo,
	out Ray rayWo,
	inout f32 seed,
	inout f32v2 seed2,
	out RayQueryResult bsdfRq,
	out bool sampledTransmission,
	const int iteration
) {
	int returnEnum = skHitNone;
	f32 bsdfPdf;
	f32 emitPdf;

	// -- propagate irradiance throughput along bsdf
	bsdfWo = (
		openPbrSampleWo(
			/*tables=*/ tables,
			/*nor=*/ itNormalWorld,
			/*wi=*/ itWi,
			/*mat=*/ material,
			/*pdf=*/ bsdfPdf,
			/*sampledTransmission=*/ sampledTransmission,
			seed, seed2
		)
	);
	const f32v3 irradianceThroughputPrevious = irradianceThroughput;

	// uncomment this block to test itNormalWorld vs itNormalGeometrical
#if 0
	if (dot(itNormalGeometrical, bsdfWo) < 0.0f)
	{
		// if the sampled wo is below the surface, then terminate path
		irradianceAccumulator = (itNormalGeometrical-itNormalWorld);
		return skHitTerminate;
	}
#endif

	// below-surface check: valid for reflection but not for transmission
	// (refracted wo correctly points below the surface normal)
	f32 dotNorWo = dot(itNormalWorld, bsdfWo);
	if (!sampledTransmission && dotNorWo <= 0.0f) {
		return skHitTerminate;
	}

	// propagate irradiance throughput along bsdf; use abs() so transmission
	// cosine factor is positive (wo is below normal for refracted rays)
	irradianceThroughput *= (
		openPbrEvaluateF(tables, itNormalWorld, itWi, bsdfWo, material)
		* abs(dotNorWo) / bsdfPdf
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
				itOri + itNormalGeometrical * (sampledTransmission ? -0.0005f : 0.0005f),
				bsdfWo,
				/*maxDist=*/999999.0f,
				/*isShadowRay=*/false,
				/*testAlphaCutoff=*/true
			)
		);

		// miss shader hits the sky for now
		if (bsdfRq.dist <= 0.0f) {
			// -- sky contribution
			const GpuDebugPC unf = GpuDebugPCBuffer(pc.global.debug).data;
			irradianceAccumulator += (
				// f32v3(0.3f) * irradianceThroughput
				sampleSky(
					/*dir=*/ bsdfWo,
					/*sunDir=*/ unf.sunDir,
					/*turbidity=*/ unf.skyTurbidity,
					/*intensity=*/ unf.skyIntensity
				) * irradianceThroughput
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
	const MaterialTableHandles tables,
	inout OpenPbrMaterial material,
	inout f32v3 itOri,
	inout f32v3 itNormalWorld,
	inout f32v3 itNormalGeometrical,
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
	bool sampledTransmission;
	int returnEnum = (
		utilIrradiance(
			tables,
			material,
			itOri,
			itNormalWorld,
			itNormalGeometrical,
			itWi,
			irradianceThroughput,
			irradianceAccumulator,
			bsdfWo,
			rayWo,
			seed, seed2,
			bsdfRq,
			sampledTransmission,
			iteration
		)
	);

	// -- Beer-Lambert attenuation through transmissive medium
	if (
		sampledTransmission
		&& material.transmissionDepth > 0.0f
		&& bsdfRq.dist > 0.0f
		&& returnEnum != skHitTerminate
	) {
		/*
			Beer-Lambert law (spec 3.7.4):\\
			\mu_t = -\frac{\ln T}{\lambda},
			\quad T = transmissionColor,
			\quad \lambda = transmissionDepth\\
			attenuation = \exp(-\mu_t \cdot d) = T^{d / \lambda}
		*/
		const f32v3 mu = (
			-log(max(material.transmissionColor, f32v3(1e-6f)))
			/ material.transmissionDepth
		);
		irradianceThroughput *= exp(-mu * bsdfRq.dist);
	}

	// -- update the material and ray for the next bounce
	if (returnEnum == skHitIndirect || returnEnum == skHitNone) {
		const UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				bsdfRq.modelDrawIndex,
				bsdfRq.primitiveIndex,
				bsdfRq.barycentric
			)
		);
		const mat3 tbn = (
			utilCalculateTbnBasis(rqData.normal, rqData.tangent)
		);
		f32v3 modelNormal;
		material = (
			openPbrMaterialFromGltfWithLod(
				/*material=*/rqData.material,
				/*uv=*/rqData.uv,
				/*lod=*/ 0,
				/*modelNormal=*/modelNormal
			)
		);
		itOri = itOri + bsdfWo * bsdfRq.dist;
		itNormalWorld = normalize(tbn * modelNormal);
		itNormalGeometrical = rqData.normalGeometrical;
		itWi = -bsdfWo;
		// after transmission the mesh normal faces away from the incoming ray
		// (back face of the object); flip so the next bounce sees a consistent frame
		if (sampledTransmission && dot(itWi, itNormalGeometrical) < 0.0f) {
			itNormalWorld = -itNormalWorld;
			itNormalGeometrical = -itNormalGeometrical;
		}
	}

	return returnEnum;
}

f32v4 utilIrradiancePropagationEntry(
	const MaterialTableHandles tables,
	OpenPbrMaterial originalMaterial,
	f32v3 origin,
	f32v3 normalWorld,
	f32v3 normalGeometrical,
	f32v3 wi,
	const uint propagationDepth,
	inout f32 seed,
	inout f32v2 seed2
) {
	f32v3 irradianceThroughput = f32v3(1.0);
	f32v3 irradianceAccumulator = f32v3(0.0);

	int hit = 0;
	for (int i = 0; i < propagationDepth; ++i) {
		int returnEnum = (
			utilIrradiancePropagate(
				tables,
				originalMaterial,
				origin,
				normalWorld,
				normalGeometrical,
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
