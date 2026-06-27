vec3 sampleGridIrradiance(
	const vec3 worldPos,
	const vec3 worldNormal,
	const GpuDdgiGrid grid,
	const vec2 localUv
) {
	// -- scale the world position into probe grid space
	// -- subtract 0.5 to account for probes being placed at (px+0.5)*spacing
	const vec3 gridPos = (
		(worldPos - grid.origin) / vec3(grid.probeSpacing) - 0.5
	);
	const ivec3 baseProbe = ivec3(floor(gridPos));
	const vec3 alpha = fract(gridPos);

	// -- loop over the 8 surrounding probes and trilinearly interpolate
	vec3 irradiance = vec3(0.0);
	float weightSum = 0.0;
	for (int i = 0; i < 8; ++i) {
		const ivec3 probeOffset = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
		const ivec3 probeIdx = (
			clamp(
				baseProbe + probeOffset,
				ivec3(0),
				ivec3(grid.probeCounts) - ivec3(1)
			)
		);

		// -- trilinear weight: (1-alpha) for low neighbor, alpha for high
		const vec3 w = mix(vec3(1.0) - alpha, alpha, vec3(probeOffset));
		float weight = w.x * w.y * w.z;

		// -- chebyshev visibility weight + backface weight
		{
			const vec3 probeWorldOrigin = (
				grid.origin + (vec3(probeIdx) + vec3(0.5)) * vec3(grid.probeSpacing)
			);
			const vec3 probeToSurface = worldPos - probeWorldOrigin;

			// -- backface: fade probes behind the surface to zero
			const vec3 probeDir = normalize(-probeToSurface);
			weight *= 0.5 * (1.0 + dot(worldNormal, probeDir));

			const vec2 depthOctUv = octahedronEncode(normalize(probeToSurface));
			const vec2 depthLocalUv = (
				clamp(
					(depthOctUv * 0.5 + 0.5) * 16.0,
					0.5,
					15.5
				)
			);
			const uvec3 dSlot = (
				uvec3(
					(uint(probeIdx.x) + grid.scrollOffset.x) % grid.probeCounts.x,
					(uint(probeIdx.y) + grid.scrollOffset.y) % grid.probeCounts.y,
					(uint(probeIdx.z) + grid.scrollOffset.z) % grid.probeCounts.z
				)
			);
			const vec3 depthAtlasUv = (
				vec3(
					(
						(float(dSlot.x) * 16.0 + depthLocalUv.x)
						/ (float(grid.probeCounts.x) * 16.0)
					),
					(
						(float(dSlot.y) * 16.0 + depthLocalUv.y)
						/ (float(grid.probeCounts.y) * 16.0)
					),
					(float(dSlot.z) + 0.5) / float(grid.probeCounts.z)
				)
			);
			const vec2 depthTexel = (
				texture(
					vkofTextures3D[nonuniformEXT(grid.depthSamplerHandle)],
					depthAtlasUv
				).rg
			);
			const float avgDepth = depthTexel.r;
			const float avgDepthSqr = depthTexel.g;

			// compute variance of depth samples
			const float variance = abs(avgDepthSqr - avgDepth * avgDepth);

			// apply chebyshev to compute probability of visibility
			const float depthToSurface = length(probeToSurface);
			const float chebyshev = (
				depthToSurface <= avgDepth
				? 1.0f
				: (
					variance
					/ (
						variance
						+ (depthToSurface - avgDepth) * (depthToSurface - avgDepth)
					)
				)
			);

			// crush the weight to reduce light leaks
			weight *= chebyshev * chebyshev;
		}

		// -- sample the probe irradiance
		const uvec3 iSlot = (
			uvec3(
				(uint(probeIdx.x) + grid.scrollOffset.x) % grid.probeCounts.x,
				(uint(probeIdx.y) + grid.scrollOffset.y) % grid.probeCounts.y,
				(uint(probeIdx.z) + grid.scrollOffset.z) % grid.probeCounts.z
			)
		);
		const vec3 atlasUv = (
			vec3(
				(float(iSlot.x) * 8.0 + localUv.x) / (float(grid.probeCounts.x) * 8.0),
				(float(iSlot.y) * 8.0 + localUv.y) / (float(grid.probeCounts.y) * 8.0),
				(float(iSlot.z) + 0.5) / float(grid.probeCounts.z)
			)
		);

		// -- accumulate
		irradiance += (
			weight
			* (
				texture(
					vkofTextures3D[nonuniformEXT(grid.irradianceSamplerHandle)],
					atlasUv
				).rgb
			)
		);
		weightSum += weight;
	}

	// -- normalize by the weight sum
	irradiance /= max(weightSum, 0.0001f);

	return irradiance;
}

vec3 sampleProbeIrradiance(
	const vec3 worldPos,
	const vec3 worldNormal
) {
	if (pc.global.ddgiGrid == 0u) { return vec3(0.0); }
	const GpuDdgiCascades cascades = (
		GpuDdgiCascadesBuffer(pc.global.ddgiGrid).data
	);

	const vec2 octahedronUv = octahedronEncode(normalize(worldNormal));
	const vec2 localUv = clamp((octahedronUv * 0.5 + 0.5) * 8.0, 0.5, 7.5);

	// -- sample from each cascade (finest first), weight it if near the edge
	vec3 irradiance = vec3(0.0);
	// share of irradiance not yet claimed; coarser cascades fill the rest
	float remainingWeight = 1.0;
	for (int ci = 0; ci < int(cascades.count); ++ci) {
		const GpuDdgiGrid grid = cascades.grids[ci];
		const vec3 cascadeSize = (
			vec3(grid.probeCounts) * vec3(grid.probeSpacing)
		);
		const vec3 relPos = worldPos - grid.origin;

		if (
			any(lessThan(relPos, vec3(0.0)))
			|| any(greaterThan(relPos, cascadeSize))
		) {
			continue;
		}

		// -- blend zone: fade out toward cascade edge before sampling next cascade
		const vec3 center = cascadeSize * 0.5;
		const vec3 fromCenter = abs(relPos - center) / center;
		const float d = max(fromCenter.x, max(fromCenter.y, fromCenter.z));
		const float blendFraction = 0.1;
		const float blendAlpha = (
			clamp(
				(d - (1.0 - blendFraction)) / blendFraction,
				0.0, 1.0
			)
		);

		const float cascadeWeight = remainingWeight * (1.0 - blendAlpha);
		irradiance += (
			cascadeWeight
			* sampleGridIrradiance(worldPos, worldNormal, grid, localUv)
		);
		remainingWeight -= cascadeWeight;

		if (remainingWeight < 0.001) { break; }
	}

	return irradiance;
}
