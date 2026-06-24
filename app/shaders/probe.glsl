#extension GL_EXT_debug_printf : enable

bool shader_probe_is_active_pixel(const ivec2 coord) {
	const GpuDebugPC dbg = GpuDebugPCBuffer(pc.global.debug).data;
	return (
		dbg.probeActive != 0u
		&& coord == ivec2(dbg.probeX, dbg.probeY)
	);
}
