cmake_minimum_required(VERSION 3.20)
project(cull)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# CMake 4+ removed support for cmake_minimum_required versions < 3.5.
# Allow FetchContent'd projects that still declare VERSION 3.x to configure.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# ------------------------------------------------------------------------------
# -- ccache
# ------------------------------------------------------------------------------

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK	ccache)
endif()

# ------------------------------------------------------------------------------
# -- sanitizers
# ------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set(san_flags -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
	set(san_link  -fsanitize=address -fsanitize=undefined)
else()
	set(san_flags "")
	set(san_link  "")
endif()

# ------------------------------------------------------------------------------
# -- dependencies (FetchContent)
# ------------------------------------------------------------------------------

include(FetchContent)

# check for latest tags at:
#   https://github.com/charles-lunarg/vk-bootstrap/releases
#   https://github.com/glfw/glfw/releases
#   https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/releases

FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
	GIT_TAG		main
	GIT_SHALLOW	TRUE
)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG		3.4
	GIT_SHALLOW	TRUE
)

FetchContent_Declare(
	vma
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
	GIT_TAG		v3.1.0
	GIT_SHALLOW	TRUE
)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.90.9
)
FetchContent_MakeAvailable(imgui)

set(GLFW_BUILD_DOCS	 OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS	OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(vk-bootstrap glfw vma)

# ------------------------------------------------------------------------------
# -- Vulkan
# ------------------------------------------------------------------------------

find_package(Vulkan REQUIRED)

# ------------------------------------------------------------------------------
# -- shader compilation
# ------------------------------------------------------------------------------

find_program(GLSLANG glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
if(NOT GLSLANG)
	message(FATAL_ERROR "glslangValidator not found")
endif()

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/app/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)

## compile_shader(<output_var> <stage> <source_path>)
## stage is the glslangValidator -S argument: vert frag mesh task comp etc.
#function(compile_shader out_var stage src)
#	get_filename_component(name ${src} NAME)
#	set(spv ${SHADER_BIN_DIR}/${name}.spv)
#	add_custom_command(
#		OUTPUT  ${spv}
#		COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_DIR}
#		COMMAND ${GLSLANG} -V --target-env vulkan1.3 -S ${stage} ${src} -o ${spv}
#		DEPENDS ${src}
#		VERBATIM
#	)
#	set(${out_var} ${spv} PARENT_SCOPE)
#endfunction()
#
#compile_shader(mesh_spv mesh ${SHADER_SRC_DIR}/triangle.mesh.glsl)
#compile_shader(frag_spv frag ${SHADER_SRC_DIR}/triangle.frag.glsl)
#
#add_custom_target(shaders ALL DEPENDS ${mesh_spv} ${frag_spv})

# ------------------------------------------------------------------------------
# -- shared compile flags (interface library consumed by all three targets)
# ------------------------------------------------------------------------------

add_library(cull_flags INTERFACE)

target_compile_options(
	cull_flags
	INTERFACE
	-Wextra
	-Wpedantic
	$<$<CONFIG:Debug>:-O0 -g3 -ggdb>
	$<$<CONFIG:RelWithDebInfo>:-O2 -g3 -ggdb>
	$<$<CONFIG:Release>:-O3>
	${san_flags}
)

# ------------------------------------------------------------------------------
# -- srat (non-graphics utility library)
# ------------------------------------------------------------------------------

add_library(
	srat
	STATIC
	lib/srat/alloc-virtual-range.cpp
)

target_include_directories(
	srat
	PUBLIC
	lib/
)

target_link_libraries(
	srat
	PRIVATE
	cull_flags
)

# ------------------------------------------------------------------------------
# -- vkof (Vulkan object framework; depends on srat)
# ------------------------------------------------------------------------------

add_library(
	vkof
	STATIC
	lib/vkof/vkof.cpp
	lib/vkof/vma.cpp
)

target_link_libraries(
	vkof
	PUBLIC
	srat
	PRIVATE
	cull_flags
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_include_directories(
	vkof
	SYSTEM PRIVATE
	${vma_SOURCE_DIR}/include
	${imgui_SOURCE_DIR}
)

target_precompile_headers(
	vkof
	PRIVATE
	<vulkan/vulkan.hpp>
	<VkBootstrap.h>
	<vk_mem_alloc.h>
	<GLFW/glfw3.h>
)

# ------------------------------------------------------------------------------
# -- cull executable
# ------------------------------------------------------------------------------

add_executable(
	cull
	app/main.cpp
	app/gfx.cpp
	app/cull.cpp
)
#add_dependencies(cull shaders)

target_sources(
	cull
	PRIVATE
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_link_libraries(
	cull
	PRIVATE
	cull_flags
	srat
	vkof
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_include_directories(
	cull
	SYSTEM PRIVATE
	${vma_SOURCE_DIR}/include
	${imgui_SOURCE_DIR}
)

target_precompile_headers(
	cull
	REUSE_FROM
	vkof
)

target_link_options(
	cull
	PRIVATE
	-fuse-ld=mold
	${san_link}
)

# copy compiled shaders next to the binary so relative paths work
#add_custom_command(
#	TARGET
#	cull
#	POST_BUILD COMMAND
#		${CMAKE_COMMAND} -E copy_directory
#		${SHADER_BIN_DIR} $<TARGET_FILE_DIR:cull>/shaders
#)

install(
	TARGETS cull
	RUNTIME DESTINATION bin
)

#install(
#	DIRECTORY ${SHADER_BIN_DIR}/
#	DESTINATION bin/shaders
#)

# ------------------------------------------------------------------------------
# -- unit tests
# ------------------------------------------------------------------------------

add_subdirectory(unit-test)
