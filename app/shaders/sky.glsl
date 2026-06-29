// Preetham 1999 analytic sky model
// output is normalized to artistic units (physical kcd/m² divided by 30)
// dir: normalized view direction (y-up world space)
// sunDir: normalized direction toward sun
// turbidity: 1.0 (very clear) to 10.0 (hazy)

float skyPerezF(
	const float cosTheta,
	const float gamma,
	const float A, const float B,
	const float C, const float D, const float E
) {
	const float cosGamma = cos(gamma);
	return (
		(1.0 + A * exp(B / max(cosTheta, 0.01)))
		* (1.0 + C * exp(D * gamma) + E * cosGamma * cosGamma)
	);
}

vec3 sampleSky(
	const vec3 dir,
	const vec3 sunDir,
	const float turbidity,
	const float skyIntensity
) {
	if (dir.y < 0.0) {
		return vec3(0.3f);
	}
	// return vec3(0.0f);
	const float T = turbidity;
	const float sunTheta = acos(clamp(sunDir.y, 0.0, 1.0));
	const float cosTheta = max(dir.y, 0.01);
	const float gamma = acos(clamp(dot(dir, sunDir), -1.0, 1.0));

	// -- distribution coefficients (Preetham 1999 Table 2)
	const float Ay =  0.1787*T - 1.4630;
	const float By = -0.3554*T + 0.4275;
	const float Cy = -0.0227*T + 5.3251;
	const float Dy =  0.1206*T - 2.5771;
	const float Ey = -0.0670*T + 0.3703;

	const float Ax = -0.0193*T - 0.2592;
	const float Bx = -0.0665*T + 0.0008;
	const float Cx =  0.0004*T + 0.2125;
	const float Dx = -0.0641*T - 0.8989;
	const float Ex = -0.0033*T + 0.0452;

	const float Ay2 = -0.0167*T - 0.2608;
	const float By2 = -0.0950*T + 0.0092;
	const float Cy2 = -0.0079*T + 0.2102;
	const float Dy2 = -0.0441*T - 1.6537;
	const float Ey2 = -0.0109*T + 0.0529;

	// -- zenith luminance kcd/m^2 (Preetham 1999 Eq. 7)
	const float chi = (4.0/9.0 - T/120.0) * (3.14159265 - 2.0*sunTheta);
	const float Yz = max(0.0, (4.0453*T - 4.9710) * tan(chi) - 0.2155*T + 2.4192);

	// -- zenith chromaticity (Preetham 1999 Table 4)
	const float st3 = sunTheta*sunTheta*sunTheta;
	const float st2 = sunTheta*sunTheta;
	const float xz = (
		T*T * ( 0.00166*st3 - 0.02903*st2 + 0.11693*sunTheta)
		+ T  * (-0.00375*st3 + 0.06377*st2 - 0.21196*sunTheta + 0.00394)
		+ 1.0 * (0.00209*st3 - 0.03202*st2 + 0.06052*sunTheta + 0.25886)
	);
	const float yz = (
		T*T * ( 0.00275*st3 - 0.04214*st2 + 0.15346*sunTheta)
		+ T  * (-0.00610*st3 + 0.08970*st2 - 0.26756*sunTheta + 0.00516)
		+ 1.0 * (0.00317*st3 - 0.04153*st2 + 0.05135*sunTheta + 0.26688)
	);

	// -- perez distribution ratio
	const float denY  = skyPerezF(1.0, sunTheta, Ay,  By,  Cy,  Dy,  Ey);
	const float denX  = skyPerezF(1.0, sunTheta, Ax,  Bx,  Cx,  Dx,  Ex);
	const float denY2 = skyPerezF(1.0, sunTheta, Ay2, By2, Cy2, Dy2, Ey2);

	const float Y = Yz * skyPerezF(cosTheta, gamma, Ay,  By,  Cy,  Dy,  Ey)  / max(denY,  0.0001);
	const float x = xz * skyPerezF(cosTheta, gamma, Ax,  Bx,  Cx,  Dx,  Ex)  / max(denX,  0.0001);
	const float y = yz * skyPerezF(cosTheta, gamma, Ay2, By2, Cy2, Dy2, Ey2) / max(denY2, 0.0001);

	// -- Yxy -> XYZ
	const float invY = Y / max(y, 0.0001);
	const float X = invY * x;
	const float Z = invY * (1.0 - x - y);

	// -- XYZ -> linear sRGB (D65)
	const vec3 rgb = vec3(
		 3.2406*X - 1.5372*Y - 0.4986*Z,
		-0.9689*X + 1.8758*Y + 0.0415*Z,
		 0.0557*X - 0.2040*Y + 1.0570*Z
	);

	// -- solar disk
	const float sunDisk = smoothstep(0.9997, 0.9999, dot(dir, sunDir));
	const vec3 sunColor = mix(vec3(0.8, 0.4, 0.1), vec3(0.8, 0.7, 0.5), clamp(sunDir.y * 4.0, 0.0, 1.0));

	return (max(rgb, vec3(0.0)) / 30.0 + sunDisk * sunColor) * skyIntensity;
}

// approximate color of the sun as a directional light
// matches the warm/cool shift visible in the sky dome
vec3 sunLightColor(const vec3 sunDir, const float turbidity) {
	const float elevation = clamp(sunDir.y, 0.0, 1.0);
	const float haze = clamp((turbidity - 1.0) / 9.0, 0.0, 1.0);
	const vec3 highSun = mix(vec3(1.0, 0.97, 0.92), vec3(1.0, 0.85, 0.65), haze);
	const vec3 lowSun = mix(vec3(1.0, 0.55, 0.10), vec3(1.0, 0.40, 0.05), haze);
	return mix(lowSun, highSun, smoothstep(0.0, 0.3, elevation));
}
