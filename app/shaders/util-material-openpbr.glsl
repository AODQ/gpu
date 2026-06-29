// shitty openbr implementation
// https://academysoftwarefoundation.github.io/OpenPBR/index.html

#ifndef UTIL_MATERIAL_OPENPBR_GLSL
#define UTIL_MATERIAL_OPENPBR_GLSL

#include "util-random.glsl"
#include "util-material-openpbr-tables.glsl"

#define IPI 0.31830988618379067154
#define PI  3.14159265358979323846
#define TAU 6.28318530717958647692
#define kEpsi1 (0.99999999999999999f)
const float skFon1 = 0.5f - 2.0f / (3.0f * PI);
const float skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

/*
	                                                 ^ emission
	.---------.--------------------------------------|-------.
	|         | fuzz                                 |       |
	|         '--------------------------------------|-------' <- thin film
	|         | clearcoat                            |       |
	|         '-------'--------------'------------'----------'
	|         | metal |              | subsurface | gloss    |
	| ambient '       | translucent  |            |----------'
	| medium  |       | base         |            | diffuse  |
	'---------'-------'--------------'------------'----------'
	                                 <--- opaque base ------->
	                  < ---------- dielectric base ---------->
	          < ------------- base -------------------------->

	the resulting material is denoted M_{pbr}

	M_{pbr} = mix(S_{ambient-medium}, M_{surface}, \alpha)\\
	M_{surface} = mix(M_{coated-base}, S_{fuzz}, F)\\
	M_{coated-base} = mix(M_{base-substrate}, S_{coat}, C)\\
	M_{base-substrate} = mix(M_{dielectric-base}, S_{metal}, M)\\
	M_{dielectric-base} = mix(M_{opaque-base}, S_{translucent-base}, T)\\
	M_{opaque-base} = mix(M_{glossy-diffuse}, S_{subsurface}, S)\\
	M_{glossy-diffuse} = layer(S_{diffuse}, S_{gloss})\\

	where,
		\alpha = geometry opacity,
		F = fuzz weight,
		C = clearcoat weight,
		M = base metallic,
		T = transmission weight,
		S = subsurface weight,
*/

struct OpenPbrMaterial {
	// base layer
	f32v3 baseColor;
	f32 baseWeight;
	f32 baseDiffuseRoughness;
	f32 baseMetallic;

	// specular layer
	f32v3 specularColor;
	f32 specularIor;
	f32 specularWeight;
	f32 specularRoughnessScale;
	f32 specularRoughnessAnisotropy;

	// subsurface layer
	f32 subsurfaceWeight;
	f32v3 subsurfaceColor;
	f32 subsurfaceRadius;
	f32v3 subsurfaceRadiusScale;
	f32 subsurfaceScatterAnisotropy;

	// transmission layer
	f32 transmissionWeight;
	f32v3 transmissionColor;
	f32 transmissionDepth;
	f32 transmissionDispersionAbbeNumber;

	// coat layer
	f32 coatWeight;
	f32v3 coatColor;
	f32 coatRoughness;
	f32 coatIor;

	// fuzz layer
	f32 fuzzWeight;
	f32v3 fuzzColor;
	f32 fuzzRoughness;

	// thin film layer
	f32 thinFilmWeight;
	f32 thinFilmIor;
	f32 thinFilmThickness;

	// emission layer
	f32v3 emissionColor;
	f32 emissionLuminance;

	// geometry
	f32 geometryOpacity;
	f32 geometryThinWalled;
};


// -----------------------------------------------------------------------------
// -- utility samplers/orthogonal basis
// -----------------------------------------------------------------------------

void utilCalculateXy(
	const f32v3 nor,
	out f32v3 binormal,
	out f32v3 bitangent
) {
	// frisvad duff 2017 orthornormal basis revisted
	const f32 sgn = nor.z >= 0.0f ? 1.0f : -1.0f;
	const f32 a = -1.0f / (sgn + nor.z);
	const f32 b = nor.x * nor.y * a;
	binormal = f32v3(1.0f + sgn*nor.x*nor.x*a, sgn*b, -sgn*nor.x);
	bitangent = f32v3(b, sgn + nor.y*nor.y*a, -nor.y);
}

mat3 utilCalculateTbnBasis(
	const f32v3 nor,
	const f32v4 tangent
) {
	// gram-schmidt re-orthogonalization
	const f32v3 n = normalize(nor);
	const f32v3 tRaw = tangent.xyz - n * dot(n, tangent.xyz);
	const f32 tLen = length(tRaw);
	if (tLen < 1e-6) {
		// degerate tangent, fall-back to Frisvad basis
		f32v3 binormal, bitangent;
		utilCalculateXy(n, binormal, bitangent);
		return mat3(binormal, bitangent, n);
	}
	const f32v3 t = tRaw / tLen;
	const f32v3 bitangent = (cross(n, t) * tangent.w);
	return mat3(t, bitangent, n);
}

f32v3 utilReorientHemisphere(f32v3 wo, f32v3 nor) {
	f32v3 binormal, bitangent;
	const f32v3 n = normalize(nor);
	utilCalculateXy(n, binormal, bitangent);
	return bitangent*wo.x + binormal*wo.y + wo.z*n;
}

f32v3 utilToCartesian(const f32 cosTheta, const f32 phi) {
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return f32v3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
}

f32v3 utilCosineHemisphereSampleWo(
	f32v3 nor, inout f32v2 seed2
) {
	const f32v2 u = fnSampleUniform2(seed2);
	const f32v3 wo = (
		utilReorientHemisphere(
			normalize(utilToCartesian(sqrt(u.x), TAU*u.y)),
			nor
		)
	);
	return wo;
}

// -----------------------------------------------------------------------------
// -- utility implementation detail microfacet distribution evaluations
// -----------------------------------------------------------------------------
// the OpenPBR doesn't really get much into details about the microfacet model;
// this is just ported from my old microfacet code which didn't really leave
// any documentation

f32 utilMicrofacetGgxDistribution(const f32 dotNorH, const f32 a2) {
	const f32 d = max(dotNorH * dotNorH * (a2 - 1.0) + 1.0, 1e-5);
	return a2 / (PI * d * d);
}

// height-correlated smith visibility
// folds the \fract{1}{4 G_1(\omega_i) G_1(\omega_o)} term into the distribution
f32 utilMicrofacetSmithGgxVisibility(
	const f32 dotNorWi,
	const f32 dotNorWo,
	const f32 a2
) {
	const f32 ggxWi = dotNorWo * sqrt(dotNorWi * (dotNorWi - dotNorWi*a2) + a2);
	const f32 ggxWo = dotNorWi * sqrt(dotNorWo * (dotNorWo - dotNorWo*a2) + a2);
	return 0.5f / max(ggxWi + ggxWo, 1e-5f);
}

// dielectric fresnel;
// eta is the specularIor for reflection and 1.0/specularIor for refraction
f32 utilMicrofacetFresnelDielectric(const f32 cosTheta, const f32 eta) {
	const f32 cosTheta2 = cosTheta * cosTheta;
	const f32 sinTheta2 = 1.0f - cosTheta2;
	const f32 eta2 = eta * eta;
	const f32 sinThetaT2 = sinTheta2 / eta2;
	if (sinThetaT2 > 1.0f) {
		return 1.0f; // total internal reflection
	}
	const f32 cosThetaT = sqrt(1.0f - sinThetaT2);
	const f32 rParallel = (
		(eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT)
	);
	const f32 rPerspective = (
		(cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT)
	);
	return 0.5f * (rParallel*rParallel + rPerspective*rPerspective);
}

// ggx vndf sampling from heitz 2018
// guaruntees dot(wi, H) > 0.0f
f32v3 utilMicrofacetSampleGgxVndf(
	const f32v3 N,
	const f32v3 wi,
	const f32 roughness,
	const f32v2 xi
) {
	f32v3 binormal, bitangent;
	utilCalculateXy(N, binormal, bitangent);
	const f32v3 wiLocal = (
		f32v3(dot(wi, bitangent), dot(wi, binormal), dot(wi, N))
	);
	const f32v3 wiH = (
		normalize(f32v3(roughness * wiLocal.x, roughness * wiLocal.y, wiLocal.z))
	);
	const f32 lensq = wiH.x * wiH.x + wiH.y * wiH.y;
	const f32v3 T1 = (
		lensq > 1e-7f
		? f32v3(-wiH.y, wiH.x, 0.0f) * (1.0f / sqrt(lensq))
		: f32v3(1.0f, 0.0f, 0.0f)
	);
	const f32v3 T2 = cross(wiH, T1);
	const f32 r = sqrt(xi.x);
	const f32 phi = TAU * xi.y;
	const f32 t1 = r * cos(phi);
	f32 t2 = r * sin(phi);
	const f32 s = 0.5f * (1.0f + wiH.z);
	t2 = (1.0f - s) * sqrt(max(1.0f - t1 * t1, 0.0f)) + s * t2;
	const f32v3 Nh = (
		t1 * T1 + t2 * T2
		+ sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wiH
	);
	const f32v3 Hlocal = (
		normalize(f32v3(roughness * Nh.x, roughness * Nh.y, max(0.0f, Nh.z)))
	);
	return Hlocal.x * bitangent + Hlocal.y * binormal + Hlocal.z * N;
}

// one-sided smith geometrical masking term
f32 utilMicrofacetSmithG1(const f32 dotNorWi, const f32 alpha2) {
	return (
		2.0f
		* dotNorWi
		/ (dotNorWi + sqrt(alpha2 + (1.0f - alpha2) * dotNorWi * dotNorWi))
	);
}

// returns the fractional energy to mix diffuse and specular
// uses kulla-conty multi-scatter energy compensation tables
f32 utilMicrofacetDielectricEnergyCompensate(
	const f32 mu,
	const f32 roughness
) {
	const f32 ai = clamp(roughness * 15.0f, 0.0f, 14.999f);
	const f32 mi = clamp(mu*16.0f - 0.5f, 0.0f, 14.999f);
	const int ai0 = int(ai);
	const int ai1 = min(ai0 + 1, 15);
	const int mi0 = int(mi);
	const int mi1 = min(mi0 + 1, 15);
	const f32 ta = fract(ai);
	const f32 tm = fract(mi);
	return (
		mix(
			mix(
				skKullaContyEnergyTable[mi0*16 + ai0],
				skKullaContyEnergyTable[mi0*16 + ai1],
				ta
			),
			mix(
				skKullaContyEnergyTable[mi1*16 + ai0],
				skKullaContyEnergyTable[mi1*16 + ai1],
				ta
			),
			tm
		)
	);
}

f32 utilMicrofacetPdf(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo
) {
	f32 dotNorWo = dot(nor, wo);
	// Below the hemisphere contributes no reflection density.
	if ( dotNorWo <= 0.0 ) {
		return 0.0;
	}
	const f32v3 h = normalize(wi + wo);
	const f32 dotNorH = max(dot(nor, h), 0.0f);
	const f32 a2 = (
		max(mat.specularRoughnessScale * mat.specularRoughnessScale, 0.0001f)
	);
	const f32 dotNorWi = max(dot(nor, wi), 1e-5f);
	const f32 G1 = utilMicrofacetSmithG1(dotNorWi, a2);
	const f32 specPdf = (
		G1 * utilMicrofacetGgxDistribution(dotNorH, a2) / (4.0f * dotNorWi)
	);
	const f32 diffPdf = dotNorWo * IPI;
	const f32 iorRatio = (mat.specularIor - 1.0f) / (mat.specularIor + 1.0f);
	f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f) f0 = 0.04f; // dielectric default
	const f32 dielectricProbability = (
		mat.specularWeight * f0
		* utilMicrofacetDielectricEnergyCompensate(
			dotNorWi,
			mat.specularRoughnessScale
		)
	);
	const f32 specProbability = (
		max(dielectricProbability, mat.baseMetallic)
	);
	const f32 coatA2 = mat.coatRoughness * mat.coatRoughness;
	const f32 coatG1 = (
		utilMicrofacetSmithG1(dotNorWi, coatA2)
	);
	const f32 coatSpecularProbability = (
		coatG1
		* utilMicrofacetGgxDistribution(dotNorH, coatA2) / (4.0f * dotNorWi)
	);
	const f32 coatIorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	const f32 coatF0 = coatIorRatio * coatIorRatio;
	const f32 coatProbability = (
		mat.coatWeight * coatF0
		* utilMicrofacetDielectricEnergyCompensate(dotNorWi, mat.coatRoughness)
	);
	return (
		coatProbability * coatSpecularProbability
		+ specProbability * specPdf
		+ (1.0f - max(coatProbability, specProbability)) * diffPdf
	);
	return specProbability * specPdf + (1.0f - specProbability) * diffPdf;
}

// -----------------------------------------------------------------------------
// -- utility implementation detail material evaluations
// -----------------------------------------------------------------------------

f32 utilOrenNayerEnergyCompensate(const f32 mu, const f32 r) {
	// portsmouth 2024 approximate form;
	// in future can either use a lookup table or a more accurate approximation
	const float mucomp = 1.0f - mu;
	const float g1 = 0.0571085289f;
	const float g2 = 0.491881867f;
	const float g3 = -0.332181442f;
	const float g4 = 0.0714429953f;
	float gOverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
	return (1.0f + r * gOverPi) / (1.0f + r * skFon1);
}

// -----------------------------------------------------------------------------
// -- utility material evaluation
// -----------------------------------------------------------------------------

f32v3 openPbrFresnelMetallicEvaluateF(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo
) {
	/*
		S_w = specular\ weight\\
		S_c = specular\ color\\
		f_0 = base\ weight * base\ color\\
		u = \omega_i \cdot n\\
		F_s = f_0 + (1 - f_0) (1 - u)^5\\
		F = S_c * F_s\\
		F_c = F_s(\frac{1}{7}) - F(\frac{1}{7})\\
		F_d = \frac{1}{7} (1 - \frac{1}{7})^6\\
		F_{82} = F_s - \frac{u (1 - u)^6}{F_d} * F_c\\
		F_m = S_w * F_{82}\\
	*/

	const f32v3 h = normalize(wi + wo);
	const f32 dotHNor = dot(h, nor);

	// 1/7 * (1 - 1/7)^6
	const f32 fresnelDenom = 0.056653f;

	const f32v3 f0 = mat.baseWeight * mat.baseColor;

	// f0 + (1 - f0) * (1 - 1/7)^5
	const f32v3 fresnelConstS = f0 + (f32v3(1.0f) - f0) * 0.462664f;

	// F_s(1/7) - F_s * (1/7)
	const f32v3 fresnelConst = (
		fresnelConstS - (mat.specularColor * fresnelConstS)
	);

	// fs = f0 + (1 - f0) * (1 - u)^5
	const f32v3 fs = f0 + (f32v3(1.0f) - f0)*pow(1.0f-dotHNor, 5.0f);

	// f82 = fs - (u * (1 - u)^6 / fresnelDenom) * fresnelConst
	const f32v3 f82 = (
		fs
		- (
			((dotHNor * pow(1.0f - dotHNor, 6.0f) / fresnelDenom))
			* fresnelConst
		)
		
	);

	// apply the rest of the microfacet
	const f32 a2 = mat.specularRoughnessScale * mat.specularRoughnessScale;
	return (
		mat.specularWeight
		* mat.specularColor
		* f82
	);
}

f32v3 openPbrCoatEvaluateF(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo,
	const f32v3 baseSubstrate
) {
	// coat specular reflection
	const f32v3 h = normalize(wi + wo);
	const f32 iorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f) {
		return baseSubstrate; // no coat layer
	}
	const f32 coatFresnel = (
		utilMicrofacetFresnelDielectric(dot(h, wi), mat.coatIor)
	);
	const f32 coatDistribution = (
		utilMicrofacetGgxDistribution(
			dot(nor, h),
			mat.coatRoughness * mat.coatRoughness
		)
	);
	const f32 coatVisibility = (
		utilMicrofacetSmithGgxVisibility(
			dot(nor, wi),
			dot(nor, wo),
			mat.coatRoughness * mat.coatRoughness
		)
	);
	const f32 fCoat = (
		mat.coatWeight * coatFresnel * coatDistribution * coatVisibility
	);
	const f32 coatEnergyCompensation = (
		f0
		* utilMicrofacetDielectricEnergyCompensate(
			dot(nor, wo),
			mat.coatRoughness
		)
	);
	const f32v3 coatAbsorption = mat.coatColor * mat.coatColor;
	const f32v3 transmittedBase = (
		(1.0f - coatEnergyCompensation) * baseSubstrate * coatAbsorption
	);
	return fCoat + mix(baseSubstrate, transmittedBase, mat.coatWeight);
}

f32v3 openPbrBaseSubstrate(
	const OpenPbrMaterial mat
) {
	/*
		M_{bs} = mix( M_{db}, S_{metal}, M)\\
		M_{db} = mix( M_{opaque-base}, S_{translucent-base}, T)\\

		where,
			bs = base substrate
			db = dielectric base
			M = base metallic
			S = specular,
			T = transmission weight
	*/

	const f32v3 metallicDielectricBase = f32v3(0.0f);//TODO
	const f32v3 surfaceMetallic = f32v3(0.0f);//TODO
	const f32v3 subsurfaceTranslucentBase = f32v3(0.0f);//TODO
	const f32v3 metallicOpaqueBase = f32v3(0.0f);//TODO

	return f32v3(0.0f);
}

f32v3 openPbrGlossyDiffuseOrenNayerEvaluateF(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo
) {
	/*
		layer of dielectric gloss on top of an opaque diffuse slab.
		OpenPBR implements f_{diffuse} as energy-preserving Oren-Nayer-Fujii.

		M_{glossy-diffuse} = layer(S_{diffuse}, S_{gloss})\\
		S_{gloss} = Slab(f_{dielectric}, V_{dielectric})\\

		S_{diffuse} = Slab(f_{diffuse})\\

		f_{diffuse}(\omega_i, \omega_o) =
			f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
		\\
		F_{EON}(\omega_i, \omega_o) =
			\frac{w_d C}{\pi}
			(A(\sigma) + B(\sigma) \frac{s}{t})
		\\
		A(\sigma) = 1 - \sigma^2 / (2 * (\sigma^2 + 0.33))\\
		B(\sigma) = 0.45 * \sigma^2 / (\sigma^2 + 0.09)\\
		s = max(0, \omega_{i_{proj}} \cdot \omega_{o_{proj}})\\
		t = max(N \cdot \omega_i, N \cdot \omega_o)\\
		f^{Fujii}_{EON}(\omega_i, \omega_o) =
			\frac{w_d \rho_{ms}}{\pi}
			(1 - \hat{E}_{ON}(\omega_i))
			(1 - \hat{E}_{ON}(\omega_o))
		\\
		\hat{E}_{ON}(\omega_i) = AF * (1.0 + FON2 * r)\\
		\rho_{ms} =
			\frac{C^2}{\pi}
			\frac
				{\langle \hat{E}_{ON} \rangle / (1 - \langle \hat{E}_{ON} \rangle)}
				{1 - C(1 - \langle \hat{E}_{ON} \rangle)}
		\\

		where,
			\rho_{ms} = multiple scattering on microfacet surface
			\sigma = base diffuse roughness
			\hat{E}_{ON} = energy compensation avg
			w_d = diffuse weight
			C = base color
			t is the maximum of the cosine of the incident and outgoing angles
			FON2 = energy conservation constant
			s is the azimuthal difference between \omega_i and \omega_o ;
				oren-nayer scattering surfaces retroreflects
	*/

	// -- step 1: fujii oren-nayer scattering

	// \sigma
	const f32 sigma = mat.baseDiffuseRoughness;
	const f32 sigma2 = sigma * sigma;

	// A(\sigma)
	const f32 A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));

	// B(\sigma)
	const f32 B = 0.45f * sigma2 / (sigma2 + 0.09f);

	// t = max( N \cdot \omega_i, N \cdot \omega_o)
	const f32 t = max(max(dot(nor, wi), dot(nor, wo)), 0.001f);

	// s = max(0, \omega_{i_{proj}} \cdot \omega_{o_{proj}})
	const f32v3 wiProj = normalize(wi - nor * dot(nor, wi));
	const f32v3 woProj = normalize(wo - nor * dot(nor, wo));
	// wiProj/woProj can be NaN if wi/wo is parallel to nor
	const f32 s = (
		(dot(nor, wi) >= kEpsi1) ? 0.0f : max(0.0f, dot(wiProj, woProj))
	);

	// f_{EON}
	const f32v3 fOrenNayer = (
		// \frac{w_d C}{\pi}
		mat.baseColor * mat.baseWeight * IPI
		* (
			// A(\sigma) + B(\sigma) \frac{s}{t}
			A + B * (s / t)
		)
	);

	// -- step 2: fujii oren-nayer energy compensation

	// \hat{E}_{ON}(\omega_i)
	const f32 energyCompWi = (
		utilOrenNayerEnergyCompensate(dot(nor, wi), sigma)
	);
	// \hat{E}_{ON}(\omega_o)
	const f32 energyCompWo = (
		utilOrenNayerEnergyCompensate(dot(nor, wo), sigma)
	);

	// \hat{E}_{ON}, this comes from portsmouth 2024
	const f32 albedoAvg = (
		1.0f / (1.0f + skFon1 * sigma) * (1.0f + skFon2 * sigma)
	);

	// \rho_{ms}
	const f32v3 rhoMs = (
		// \frac{C^2}{\pi}
		mat.baseColor*mat.baseColor * IPI
		// {\langle \hat{E}_{ON} \rangle / (1 - \langle \hat{E}_{ON} \rangle)}
		* (albedoAvg / max(1.0f - albedoAvg, 1e-6f))
		// / {1 - C(1 - \langle \hat{E}_{ON} \rangle)}
		/ (1.0f - mat.baseColor * (1.0f - albedoAvg))
	);

	// f^{Fujii}_{EON}(\omega_i, \omega_o)
	// note this does double IPI mul; this is what the openPBR spec specifies
	const f32v3 fOrenNayerFujii = (
		// \frac{w_d \rho_{ms}}{\pi}
		mat.baseWeight * rhoMs * IPI
		// (1 - \hat{E}_{ON}(\omega_i))
		* (1.0f - energyCompWi)
		// (1 - \hat{E}_{ON}(\omega_o))
		* (1.0f - energyCompWo)
	);

	// f_{diffuse}(\omega_i, \omega_o)
	const f32v3 fDiffuse = (
		// f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
		fOrenNayer + fOrenNayerFujii
	);

	return fDiffuse;
}

f32v3 openPbrGlossyDiffuseEvaluateF(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo
) {
	/*
		E_{glossy-diffuse} = E_{dielectric} + E_{diffuse}\\
		E_{diffuse} = (1 - E_{dielectric} * C\\

		f_{glossy-diffuse}(\omega_i, \omega_o) =
			f_{dielectric}(\omega_i, \omega_o)
			+ (1 - E_{dielectric}(\omega_o)) f_{diffuse}(\omega_i, \omega_o)
		\\

		where,
			E_{dielectric} = dielectric reflection without macrosopic transmission
			E_{diffuse} = the remaining macroscopic transmission energy
	*/

	// -- step 1: oren-nayer, complex enough for its own function
	const f32v3 fDiffuse = (
		openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, wi, wo)
	);

	// -- step 2: microfacet dielectric reflection
	/*
	   typical microfacet;
	   f_0 = (\frac{\eta - 1}{\eta + 1})^2\\
	   f_{dielectric} =
	     S_w * mf_f * mf_g * mf_v
	     / (4 * (\omega_wi \cdot n) * (\omega_wo \cdot n))
	   \\
	   where,
	     \eta = specularIor
	     S_w = specular weight
	*/
	const f32v3 h = normalize(wi + wo);
	const f32 iorRatio = (mat.specularIor - 1.0f) / (mat.specularIor + 1.0f);
	f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f) {
		f0 = 0.1f;
	}
	const f32 mfFresnel = (
		utilMicrofacetFresnelDielectric(dot(h, wi), mat.specularIor)
	);
	const f32 mfDistribution = (
		utilMicrofacetGgxDistribution(
			dot(nor, h),
			mat.specularRoughnessScale * mat.specularRoughnessScale
		)
	);
	const f32 mfVisibility = (
		utilMicrofacetSmithGgxVisibility(
			dot(nor, wi),
			dot(nor, wo),
			mat.specularRoughnessScale * mat.specularRoughnessScale
		)
	);

	// f_{dielectric}
	const f32 fDielectric = (
		// S_w * mf_f * mf_g * mf_v
		mat.specularWeight * mfFresnel * mfDistribution * mfVisibility
		// / (4 * (\omega_wi \cdot n) * (\omega_wo \cdot n))
		// this part is already folded into the smith visibility term
	);

	// -- step 3: energy compensation for the diffuse layer
	// E_{dielectric} = dielectric reflection energy fraction at \omega_o
	const f32 dielectricEnergyCompensate = (
		mat.specularWeight * f0
		* utilMicrofacetDielectricEnergyCompensate(
			dot(nor, wo),
			mat.specularRoughnessScale
		)
	);

	// f_{glossy-diffuse}
	return f32v3(fDielectric) + (1.0f - dielectricEnergyCompensate) * fDiffuse;
}

f32v3 openPbrSubsurfaceEvaluateF(
	const OpenPbrMaterial mat,
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo
) {
	/*
		E_{subsurface} = E_{spec} + E_{multiscatter}\\
		E_{multiscatter} = (1 - E_{spec}) * C

		where,
			C = subsurface color
	*/

	// -- step 1: microfacet dielectric reflection (identical to glossy-diffuse)
	const f32v3 h = normalize(wi + wo);
	const f32 mfFresnel = (
		utilMicrofacetFresnelDielectric(dot(h, wi), mat.specularIor)
	);
	const f32 mfDistribution = (
		utilMicrofacetGgxDistribution(
			dot(nor, h),
			mat.specularRoughnessScale * mat.specularRoughnessScale
		)
	);
	const f32 mfVisibility = (
		utilMicrofacetSmithGgxVisibility(
			dot(nor, wi),
			dot(nor, wo),
			mat.specularRoughnessScale * mat.specularRoughnessScale
		)
	);
	const f32 fSpec = (
		mat.specularWeight * mfFresnel * mfDistribution * mfVisibility
	);

	// -- step 2: directional albedo of the dielectric at wo
	const f32 iorRatio = (mat.specularIor - 1.0f) / (mat.specularIor + 1.0f);
	f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f) f0 = 0.04f; // dielectric default
	const f32 dielectricEnergy = (
		mat.specularWeight * f0
		* utilMicrofacetDielectricEnergyCompensate(
			dot(nor, wo),
			mat.specularRoughnessScale
		)
	);

	// -- step 3: Lambertian subsurface below the dielectric interface
	// E_{multiscatter} = (1 - E_{spec}) * C/pi
	const f32v3 fBelow = (1.0f - dielectricEnergy) * mat.subsurfaceColor * IPI;

	return f32v3(fSpec) + fBelow;
}

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

f32v3 openPbrEvaluateF(
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo,
	OpenPbrMaterial mat
) {
	// M_opaque_base = mix(M_glossy_diffuse, S_subsurface, subsurfaceWeight)
	const f32v3 glossyDiffuse = openPbrGlossyDiffuseEvaluateF(mat, nor, wi, wo);
	// no SSS for now
	// const f32v3 subsurface = openPbrSubsurfaceEvaluateF(mat, nor, wi, wo);
	const f32v3 subsurface = f32v3(0.0f);
	const f32v3 opaqueBase = mix(glossyDiffuse, subsurface, mat.subsurfaceWeight);

	// no transmission for now
	const f32v3 transmission = f32v3(0.0f);
	const f32v3 dielectricBase = (
		mix(opaqueBase, transmission, mat.transmissionWeight)
	);

	const f32v3 metal = openPbrFresnelMetallicEvaluateF(mat, nor, wi, wo);
	const f32v3 baseSubstrate = mix(dielectricBase, metal, mat.baseMetallic);

	const f32v3 coatedBase = (
		openPbrCoatEvaluateF(mat, nor, wi, wo, baseSubstrate)
	);

	// TODO fuzz, thin film, emission, ambient medium

	return coatedBase;
}

f32 openPbrEvaluatePdf(
	const f32v3 nor,
	const f32v3 wi,
	const f32v3 wo,
	const OpenPbrMaterial mat
) {
	return utilMicrofacetPdf(mat, nor, wi, wo);
}

f32v3 openPbrSampleWo(
	const f32v3 nor,
	const f32v3 wi,
	const OpenPbrMaterial mat,
	out f32 pdf,
	inout f32 seed,
	inout f32v2 seed2
) {
	// walk through the layers of materials picking sample
	const f32 iorRatio = (mat.specularIor - 1.0f) / (mat.specularIor + 1.0f);
	f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f) f0 = 0.04f; // dielectric default
	const f32 dielectricProbability = (
		mat.specularWeight * f0
		* utilMicrofacetDielectricEnergyCompensate(
			dot(nor, wi),
			mat.specularRoughnessScale
		)
	);

	const f32 coatIorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	f32 coatF0 = coatIorRatio * coatIorRatio;
	const f32 probabilityCoat = (
		mat.coatWeight * coatF0
		* utilMicrofacetDielectricEnergyCompensate(
			dot(nor, wi),
			mat.coatRoughness
		)
	);
	const f32 probabilitySpecular = max(dielectricProbability, mat.baseMetallic);
	const f32 probabilityDiff = (
		max(0.0f, 1.0f - probabilityCoat - probabilitySpecular)
	);
	const f32 probabilityTotal = (
		probabilityCoat + probabilitySpecular + probabilityDiff
	);
	const f32 u = fnSampleUniform(seed);

	f32v3 wo;
	if (u < probabilityCoat / probabilityTotal) {
		const f32v3 h = (
			utilMicrofacetSampleGgxVndf(
				nor, wi, max(mat.coatRoughness, 1e-5f),
				fnSampleUniform2(seed2)
			)
		);
		wo = normalize(2.0f * dot(wi, h) * h - wi);
	} else if (u < (probabilityCoat + probabilitySpecular) / probabilityTotal) {
		// sample specular microfacet
		const f32v2 xi = fnSampleUniform2(seed2);
		const f32 roughness = max(mat.specularRoughnessScale, 1e-5f);
		const f32v3 h = utilMicrofacetSampleGgxVndf(nor, wi, roughness, xi);
		wo = normalize(2.0f * dot(wi, h) * h - wi);
	} else {
		wo = utilCosineHemisphereSampleWo(nor, seed2);
	}
	pdf = utilMicrofacetPdf(mat, nor, wi, wo);
	return wo;
}

#endif // UTIL_MATERIAL_OPENPBR_GLSL
