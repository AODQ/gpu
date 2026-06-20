# cull

A Vulkan 1.3 renderer focused on GPU-driven visibility-buffer rendering with
mesh shaders, bindless descriptors, and hierarchical-Z occlusion culling.

## Architecture

### Library layers

**`lib/srat/`** — non-graphics utility library (no Vulkan dependency).
Primitive type aliases, arena allocator, virtual-range allocator, slice/array
types.

**`lib/vkof/`** — Vulkan abstraction. Opaque handle-based API for pipelines,
buffers, images, samplers, and a render graph. All shader compilation happens
at runtime via `glslangValidator`.

**`lib/mor/`** — glTF scene loading and meshlet generation. Builds meshlet
buffers and uploads them to the GPU.

### Application

**`app/`** — entry point and rendering. The render graph has two nodes:

- **draw node** (mesh shader) — emits packed visibility IDs into a `r32ui`
  render target. Each pixel stores `modelId[8] | meshletId[17] | triangleId[7]`.
- **resolve node** (compute) — unpacks the visibility buffer and writes debug
  visualizations (meshlet index, material index, instance index) to the color
  target.

GPU-driven frustum and HiZ occlusion culling emit `VkDrawMeshTasksIndirectEXT`
commands via compute shaders.

## Building

Requires: Vulkan SDK (`glslangValidator` on `PATH`), ninja, mold linker, C++23
compiler.

```sh
# RelWithDebInfo (most common dev build, -O2 -g3)
make release

# Debug (-O0 -g3, ASan + UBSan)
make debug

# Fully optimized (-O3)
make release-optimize
```

Binaries land in `install/<build-type>/bin/`.

```sh
./install/release/bin/cull          # run the app
./install/release/bin/vkof-test     # run unit tests
```

Shaders are compiled on demand from GLSL source at runtime. No `.spv` files
are pre-built or expected.

## CMake options

| Option | Default | Description |
|---|---|---|
| `VKOF_AFTERMATH` | `OFF` | Enable NVIDIA Aftermath GPU crash dump support. Requires the Aftermath SDK — see `third-party/aftermath/README.md` for setup. On device loss, writes `aftermath-crash.nv-gpudmp` to the working directory; open with Nsight Graphics for full shader-level decoding. |

Pass options at configure time:

```sh
cmake -G Ninja -B build/release \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=install/release \
    -DVKOF_AFTERMATH=ON
cmake --build build/release
cmake --install build/release
```
