#extension GL_EXT_debug_printf : enable

bool shader_probe_is_active_pixel(const ivec2 coord) {
	return (
		pc.global.probeActive != 0u
		&& coord == ivec2(pc.global.probeX, pc.global.probeY)
	);
}
