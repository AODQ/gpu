#extension GL_EXT_debug_printf : enable

bool shader_probe_is_active_pixel() {
	return (
		pc.global.probeActive != 0u
		&& ivec2(gl_FragCoord.xy) == ivec2(pc.global.probeX, pc.global.probeY)
	);
}
