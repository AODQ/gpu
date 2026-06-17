#pragma once

// ---------------------------------------------------------------------------
// test::helpers — lightweight test utilities built on top of the vkof API.
// ---------------------------------------------------------------------------

#include <vkof/vkof.hpp>
#include <srat/core-array.hpp>
#include <srat/core-types.hpp>
#include <vector>
#include <cstring>

namespace test {

// ---------------------------------------------------------------------------
// slice helpers — cast a POD struct to a byte slice
// ---------------------------------------------------------------------------

template <typename T>
inline srat::slice<u8 const> as_bytes(T const & value) {
	return srat::slice<u8 const>(
		reinterpret_cast<u8 const *>(&value),
		sizeof(T)
	);
}

// ---------------------------------------------------------------------------
// dispatch — submit one compute render-node and return immediately.
// The GPU work is not yet complete when this returns; call readback() (which
// uses vkQueueWaitIdle internally) to drain the queue before inspecting data.
// ---------------------------------------------------------------------------

template <typename Push>
inline void dispatch(
	vkof::Pipeline pipeline,
	Push const & push,
	u32 groupX,
	u32 groupY = 1,
	u32 groupZ = 1
) {
	vkof::RenderNode node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	srat::slice<u8 const> const pushBytes = as_bytes(push);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd		  = cmd,
				.pipeline	 = pipeline,
				.pushconstant = pushBytes,
				.groupCountX  = groupX,
				.groupCountY  = groupY,
				.groupCountZ  = groupZ,
			});
		},
	});
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	vkof::render_node_destroy(node);
}

// ---------------------------------------------------------------------------
// readback<T> — blocking GPU→CPU copy of `count` elements of type T.
// Calls vkQueueWaitIdle, so any previously submitted dispatches are also
// guaranteed to have completed when this returns.
// ---------------------------------------------------------------------------

template <typename T>
inline std::vector<T> readback(
	vkof::Buffer buf,
	u64  byteOffset,
	u32  count
) {
	std::vector<T> result(count);
	vkof::buffer_download({
		.buffer	 = buf,
		.byteOffset = byteOffset,
		.dst		= srat::slice<u8>(
			reinterpret_cast<u8 *>(result.data()),
			static_cast<u64>(count) * sizeof(T)
		),
	});
	return result;
}

// ---------------------------------------------------------------------------
// gpu_wait — force the GPU to be idle by doing a trivial buffer_download.
// Use when you need to sync without downloading useful data.
// ---------------------------------------------------------------------------

inline void gpu_wait() {
	auto scratch = vkof::buffer_create({
		.byteCount = 4,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	u8 tmp[4] {};
	vkof::buffer_download({
		.buffer	 = scratch,
		.byteOffset = 0,
		.dst		= srat::slice<u8>(tmp, 4),
	});
	vkof::buffer_destroy(scratch);
}

// ---------------------------------------------------------------------------
// make_buffer — convenience: create a DeviceOnly buffer of N u32 elements
// ---------------------------------------------------------------------------

inline vkof::Buffer make_buffer_u32(u32 count) {
	return vkof::buffer_create({
		.byteCount = static_cast<u64>(count) * sizeof(u32),
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
}

} // namespace test
