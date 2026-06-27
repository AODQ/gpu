# unit-test/CMakeLists.txt — vkof-test binary (doctest v2.4.11)

# ---------------------------------------------------------------------------
# doctest (FetchContent — shared with the parent project's FetchContent cache)
# ---------------------------------------------------------------------------

FetchContent_Declare(
	doctest
	GIT_REPOSITORY https://github.com/doctest/doctest.git
	GIT_TAG		v2.4.11
	GIT_SHALLOW	TRUE
)
FetchContent_MakeAvailable(doctest)

# ---------------------------------------------------------------------------
# vkof-test executable
# ---------------------------------------------------------------------------

add_executable(
	vkof-test
	main.cpp
	test_buffer.cpp
	test_image.cpp
	test_sampler.cpp
	test_pipeline.cpp
	test_compute.cpp
	test_transient.cpp
	test_rendernode.cpp
	test_mor.cpp
	test_accel.cpp
)

# Bake the absolute path to the test shaders directory into the binary
# so test cases can reference shader files via TEST_SHADER_DIR "shader.comp".
target_compile_definitions(
	vkof-test
	PRIVATE
	TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/"
)

target_include_directories(
	vkof-test
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
	vkof-test
	PRIVATE
	cull_flags
	vkof
	mor
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
	doctest::doctest
)

target_link_options(
	vkof-test
	PRIVATE
	-fuse-ld=mold
	${san_link}
)

install(
	TARGETS vkof-test
	RUNTIME DESTINATION bin
)

set_target_properties(vkof-test PROPERTIES INSTALL_RPATH "$ORIGIN")
