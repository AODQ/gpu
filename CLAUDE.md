# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

BEFORE PROCEEDING WITH ANY CHANGES, on any prompt then provide a proposal
or a to-do list and ask for feedback. Do not begin writing code until the
proposal is approved by user.

## Build commands

cd into the build directory `build`/ to build. make sure everything installs
into the local `install/` directory. Use ninja

```sh
# Debug build (ASan + UBSan enabled, -O0 -g3)
ninja debug

# RelWithDebInfo build (most common dev build, -O2 -g3)
ninja release

# Fully optimized build (-O3)
ninja release-optimize

# Run the app
./install/release/bin/cull

# Run the unit tests
./install/release/bin/vkof-test

# Run a single test suite or test case (doctest filter flags)
./build/release/unit-test/vkof-test --test-suite="[headless]"
./build/release/unit-test/vkof-test --test-case="buffer: upload/download round-trip"
```

Shaders are not pre-compiled. Do not compile .spv. Do not expect .spv files
to be present. All the GLSL files are compiled on-the-fly during run-time.

## Architecture

### Library layers (`lib/`)

**`lib/srat/`** — non-graphics utility library, no Vulkan dependency.
- `core-types.hpp` — primitive type aliases (`u8`/`u32`/`f32` etc.), `SRAT_ASSERT`, and the `Let` macro (`auto const &`).
- `core-array.hpp` — `srat::slice<T, N>` (non-owning span, supports fixed and dynamic extent) and `srat::array<T, N>` (stack array). These are used pervasively throughout the API instead of raw pointers.
- `core-config.hpp` — `SRAT_DEBUG()` macro, `Let` macro, tracy integration stubs.
- `alloc-virtual-range.*` — virtual memory range allocator.
- `alloc-arena.hpp` — arena allocator.

**`lib/vkof/`** — Vk Of; the main abstraction. Depends on srat, vk-bootstrap, VMA, GLFW.
- All public types are opaque handles: `struct Pipeline { u64 id; }`, `struct Buffer { u64 id; }`, etc. A zero `id` means invalid/destroyed.
- Full API is in `lib/vkof/vkof.hpp`. Implementation in `vkof.cpp` / `vma.cpp`.
- Key subsystems exposed:
  - **Pipeline** — `pipeline_graphics_create` / `pipeline_compute_create` / `pipeline_destroy`
  - **Buffer** — create, destroy, VA (`buffer_virtual_address`), host mapping, upload/download
  - **Image + Sampler** — create/destroy, bindless handles via `image_sampler_handle` / `image_storage_handle`
  - **Render graph** — `RenderNode` with declared image/buffer reads-writes, color/depth attachments, and a callback; executed via `render_graph_execute`. `TransientImage` / `TransientBuffer` are frame-lifetime resources that can be double-buffered.
  - **Draw/Dispatch** — `cmd_draw`, `cmd_dispatch`, `cmd_copy_image` (called inside render node callbacks)
  - **Lifetime** — `vkof::init()` (windowed) / `vkof::init_headless()` (no surface, for tests) / `vkof::shutdown()`

### Application (`app/`)

- `main.cpp` — entry point, currently calls `vkof::init()` / `vkof::shutdown()` with most rendering code commented out.
- `gfx.cpp/hpp` — device/swapchain setup helpers.
- `cull.cpp/hpp` — GPU-driven culling: frustum + HiZ occlusion culling implemented as compute shaders, emitting indirect draw commands.
- `shaders/` — GLSL source + pre-compiled `.spv` files. `visibility.comp` is the main culling shader (frustum cull → HiZ sample → emit `VkDrawIndexedIndirectCommand`). `hiz_mip.comp` builds the hierarchical depth pyramid.

### Shared CPU/GPU header pattern

`app/shaders/shared/shared.h` (also at `app/shared/shared.h`) is included by both GLSL shaders and C++ code. The macros `VKOFF_GLSL_BUFFER` / `VKOFF_GLSL_PTR` expand differently in each context, allowing struct layouts and buffer-reference pointers to be defined once and shared.

### Unit tests (`unit-test/`)

Tests use **doctest v2.4.11** and the `vkof::init_headless()` path — no window or display required. `helpers.hpp` provides `test::dispatch<Push>()`, `test::readback<T>()`, and `test::gpu_wait()` for compute-heavy test patterns. Tests are grouped under `TEST_SUITE("[headless]")`. The shader directory is baked in at build time as `TEST_SHADER_DIR`.

## Key conventions

- `srat::slice<T>` replaces raw pointer + size everywhere. Cast to bytes with `.cast<u8>()` or `reinterpret_cast` + construct a slice.
- Designated initializers are used consistently for all create-info structs.
- `SRAT_ASSERT` fires only in Debug builds; `SRAT_ASSERT_ALWAYS` is unconditional.
- Warnings-as-errors (`-Wextra -Wpedantic`) are enforced for all targets.

## Code style

### Naming

| Category | Convention | Examples |
|---|---|---|
| Types / structs | `PascalCase` | `HandlePool`, `BufferCreateInfo`, `AllocVirtualRange` |
| Free functions | `snake_case` | `buffer_create`, `render_node_add_image`, `handle_make` |
| Member functions | `camelCase` | `isIndexAlive`, `allocatedCount`, `printAllocationStats` |
| Struct fields / locals | `camelCase` | `byteCount`, `elementCount`, `debugName` |
| Static constexpr | `sk` prefix + `PascalCase` | `skTileSize`, `skSliceDynamicExtent`, `skEpsilon` |
| Enum values | `snake_case` or `camelCase` | `write_on_test_off`, `readWrite`, `DeviceOnly` |
| File names | `kebab-case` with dashes, never underscores | `util-ggx-sample.glsl`, `core-types.hpp`, `main.cpp` |
| Shader util headers | `util-` prefix + `kebab-case` | `util-ggx-sample.glsl`, `util-taa-filter.glsl` |

### Formatting

- **Indentation:** tabs.
- **`#pragma once`**
- **Brace placement:** opening brace on a new line for `struct`/`class`/`namespaec` bodies. For functions, if parameters break into a new line, the brace can also go on a new line. Otherwise braces always go on same line.
- **`const` placement:** east const — `u32 const maxHandles`, `char const * debugName`, `Handle const & handle`.
- **`const`** ALWAYS use const. Add const to everything including function parameters. Treat code as SSA, immutable, etc - except in cases where it makes sense to mutate
- **Pointer spacing:** `T * ptr`, `T * const ptr` — space before and after `*` or `&`. Treat them as keywords that require spaces.
- **`[[nodiscard]]`** on all query/getter functions that return a value the caller must not silently discard.
- **`inline`** for small single-expression functions defined in headers.
- **`class` vs `struct`:** `struct` for plain-old-data types and small value types (e.g. `HandlePool`), `class` for larger types with invariants and non-trivial member functions (e.g. `VirtualRangeAllocator`).
- **`using`** for type aliases, not `typedef`.
- **`enum struct`** for strongly-typed enums, not unscoped `enum`.
- **`object oriented design`** avoid completely. Use free functions and POD with handles to hide implementation details.
- **`alignment`** do not ever align parameters or struct fields, except in outstanding cases such as matrices which often have a natural column alignment. This means NO aligning `=` signs, NO aligning `:` separators, NO aligning `->`, NO padding any tokens to form columns — in C++, GLSL, or any other language. A single space before and after the token is all that is ever used.
- **`comments`** never place a comment on the same line as code. comments always go on their own line, above the code they describe.
- **`indent parameter blocks`** if a list of items -- for a function call, function signature, if statement, etc -- breaks into multiple lines, then break into a new line and indent the block. Do not ever align them.
- **`multi-line expression wrapping`** when breaking a long expression across lines, the outer parens must wrap the *entire* expression including any trailing operator, division, or comparison. Nothing is left outside the closing paren. Example — wrong: `return vec3(...\n) / 255.0;` — correct: `return (\n    vec3(...) / 255.0\n);`

### C++ idioms

- Use auto and type inference only if the type is obvious from the right-hand side, such as iterators or `auto it = static_cast<Foo *>(ptr)`. Otherwise, prefer explicit types for readability.
- Small structs used as named-parameter bundles passed by `const &` (the `*CreateInfo` / `*Info` pattern) instead of long argument lists.
- `static` factory methods (`AllocVirtualRange::create`, `HandlePool::create`) rather than constructors when setup can fail or requires non-trivial allocation.
