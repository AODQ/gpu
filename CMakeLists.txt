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

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(vk-bootstrap glfw vma)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.90.9
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
	imguizmo
	GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
	GIT_TAG master
	GIT_SHALLOW TRUE
)
FetchContent_GetProperties(imguizmo)
if(NOT imguizmo_POPULATED)
	FetchContent_Populate(imguizmo)
endif()

FetchContent_Declare(
	cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.14
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cgltf)

FetchContent_Declare(
	meshoptimizer
	GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
	GIT_TAG        v0.21
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        master
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
	rapidjson
	GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
	GIT_TAG        master
	GIT_SHALLOW    TRUE
)
set(RAPIDJSON_BUILD_DOC      OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rapidjson)

# ------------------------------------------------------------------------------
# -- Vulkan
# ------------------------------------------------------------------------------

find_package(Vulkan REQUIRED)

# ------------------------------------------------------------------------------
# -- NVIDIA Aftermath (optional GPU crash dump support)
# ------------------------------------------------------------------------------

option(VKOF_AFTERMATH "Enable NVIDIA Aftermath GPU crash dumps" OFF)
if(VKOF_AFTERMATH)
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
	find_package(Aftermath REQUIRED)
endif()

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
	lib/srat/camera.cpp
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
# -- scene (scene descriptor; depends on srat + rapidjson)
# ------------------------------------------------------------------------------

add_library(scene STATIC lib/scene/scene.cpp)

target_include_directories(scene PUBLIC lib/)

target_include_directories(
	scene
	SYSTEM PRIVATE
	${rapidjson_SOURCE_DIR}/include
)

target_link_libraries(
	scene
	PUBLIC  srat
	PRIVATE cull_flags
)

# ------------------------------------------------------------------------------
# -- vkof (Vulkan object framework; depends on srat)
# ------------------------------------------------------------------------------

add_library(
	vkof
	STATIC
	lib/vkof/vkof.cpp
	lib/vkof/vma.cpp
	lib/vkof/stb_image_write.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

if(VKOF_AFTERMATH)
	target_sources(vkof PRIVATE lib/vkof/aftermath.cpp)
	target_compile_definitions(vkof PRIVATE VKOF_AFTERMATH)
	target_link_libraries(vkof PRIVATE Aftermath::Aftermath)
	install(FILES "${Aftermath_LIBRARY}" DESTINATION bin)
endif()

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
	SYSTEM PUBLIC
	${imgui_SOURCE_DIR}
	SYSTEM PRIVATE
	${vma_SOURCE_DIR}/include
	${stb_SOURCE_DIR}
)

target_precompile_headers(
	vkof
	PRIVATE
	<VkBootstrap.h>
	<GLFW/glfw3.h>
)

set_source_files_properties(
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

# ------------------------------------------------------------------------------
# -- mor (glTF scene loading + meshlet building; depends on srat + vkof)
# ------------------------------------------------------------------------------

add_library(mor STATIC lib/mor/mor.cpp lib/mor/stb_image.cpp)

target_include_directories(mor PUBLIC lib/)
target_include_directories(mor SYSTEM PRIVATE ${cgltf_SOURCE_DIR} ${stb_SOURCE_DIR})

target_link_libraries(
	mor
	PUBLIC  srat vkof
	PRIVATE cull_flags meshoptimizer
)

# ------------------------------------------------------------------------------
# -- cull executable
# ------------------------------------------------------------------------------

add_executable(
	cull
	app/main.cpp
	app/gfx.cpp
	app/cull.cpp
	app/asset_library.cpp
	app/material-tables.cpp
	${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp
)

set_source_files_properties(
	${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp
	PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)
#add_dependencies(cull shaders)


target_link_libraries(
	cull
	PRIVATE
	cull_flags
	srat
	scene
	vkof
	mor
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_compile_definitions(
	cull
	PRIVATE
	ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/"
	$<$<BOOL:${VKOF_AFTERMATH}>:VKOF_AFTERMATH>
)

target_include_directories(
	cull
	SYSTEM PRIVATE
	${vma_SOURCE_DIR}/include
	${imguizmo_SOURCE_DIR}/src
	${rapidjson_SOURCE_DIR}/include
	${stb_SOURCE_DIR}
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

set_target_properties(cull PROPERTIES INSTALL_RPATH "$ORIGIN")

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
