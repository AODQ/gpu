// unit-test/test_buffer.cpp — 12 test cases for the vkof buffer API

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "helpers.hpp"
#include <cstring>
#include <vector>

TEST_SUITE("[headless]") {

// 1. Create and destroy a DeviceOnly buffer — no crash
TEST_CASE("buffer: create/destroy DeviceOnly") {
	auto buf = vkof::buffer_create({
		.byteCount = 256,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	vkof::buffer_destroy(buf);
}

// 2. Create and destroy a HostWritable buffer — no crash
TEST_CASE("buffer: create/destroy HostWritable") {
	auto buf = vkof::buffer_create({
		.byteCount = 256,
		.memory	= vkof::BufferMemory::HostWritable,
	});
	CHECK(buf.id != 0);
	vkof::buffer_destroy(buf);
}

// 3. DeviceOnly buffer has a non-zero BDA
TEST_CASE("buffer: DeviceOnly BDA is non-zero") {
	auto buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

// 4. HostWritable buffer has a non-zero BDA
TEST_CASE("buffer: HostWritable BDA is non-zero") {
	auto buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::HostWritable,
	});
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

// 5. HostWritable buffer returns a non-null, correctly sized host slice
TEST_CASE("buffer: HostWritable host_address is valid") {
	constexpr u64 kSize = 128;
	auto buf = vkof::buffer_create({
		.byteCount = kSize,
		.memory	= vkof::BufferMemory::HostWritable,
	});
	auto slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() != nullptr);
	CHECK(slice.size() == kSize);
	vkof::buffer_destroy(buf);
}

// 6. DeviceOnly buffer returns an empty host slice
TEST_CASE("buffer: DeviceOnly host_address is empty") {
	auto buf = vkof::buffer_create({
		.byteCount = 128,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	auto slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() == nullptr);
	vkof::buffer_destroy(buf);
}

// 7. buffer_upload followed by buffer_download round-trips correctly
TEST_CASE("buffer: upload/download round-trip") {
	constexpr u32 kCount = 64;
	constexpr u64 kBytes = kCount * sizeof(u32);

	// Source data
	std::vector<u32> src(kCount);
	for (u32 i = 0; i < kCount; ++i) { src[i] = i * 7 + 13; }

	auto buf = vkof::buffer_create({
		.byteCount = kBytes,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});

	vkof::buffer_upload({
		.buffer	 = buf,
		.byteOffset = 0,
		.data	   = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(src.data()), kBytes
		),
	});

	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(result[i] == src[i]);
	}
	vkof::buffer_destroy(buf);
}

// 8. Two distinct buffers have distinct BDAs
TEST_CASE("buffer: distinct BDAs") {
	auto b0 = vkof::buffer_create({ .byteCount = 64, .memory = vkof::BufferMemory::DeviceOnly });
	auto b1 = vkof::buffer_create({ .byteCount = 64, .memory = vkof::BufferMemory::DeviceOnly });
	CHECK(vkof::buffer_virtual_address(b0) != vkof::buffer_virtual_address(b1));
	vkof::buffer_destroy(b0);
	vkof::buffer_destroy(b1);
}

// 9. After destroy, buffer_virtual_address returns 0
TEST_CASE("buffer: BDA is 0 after destroy") {
	auto buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_destroy(buf);
	CHECK(vkof::buffer_virtual_address(buf) == 0u);
}

// 10. After destroy, buffer_host_address returns an empty slice
TEST_CASE("buffer: host_address empty after destroy") {
	auto buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_destroy(buf);
	auto slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() == nullptr);
}

// 11. Minimum-size (4-byte) buffer works
TEST_CASE("buffer: 4-byte buffer") {
	auto buf = vkof::buffer_create({
		.byteCount = 4,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

// 12. Large (1 MB) buffer works
TEST_CASE("buffer: 1MB DeviceOnly buffer") {
	constexpr u64 kOneMB = 1024u * 1024u;
	auto buf = vkof::buffer_create({
		.byteCount = kOneMB,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

} // TEST_SUITE("[headless]")
