# RT / DDGI Plan

## Status

Steps 1–6 are complete. Steps 7+ are the active work.

### Completed
- **Step 1** — RT extensions enabled in vkof (`VK_KHR_acceleration_structure`,
  `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations`); function pointers loaded.
- **Step 2** — `AccelerationStructureBlas` / `AccelerationStructureTlas` handle
  types in vkof.hpp; separate impl pools in vkof.cpp.
- **Step 3** — Flat u32 index buffer built in `mor::scene_gpu_upload`; exposed
  via `mor::Buffers.flatIndices` / `.vertexCount` / `.triangleCount`.
- **Step 4** — `vkof::blas_create` / `blas_destroy` implemented; synchronous
  one-shot build at model load time.
- **Step 5** — `vkof::tlas_create` / `tlas_build` / `tlas_destroy` /
  `acceleration_structure_device_address` implemented.
- **Step 6** — App wired: BLAS built per model in `fnVerifyModelLoaded`; TLAS
  rebuilt each frame in a `tlasNode` render node before resolve; `debugPcBuffer`
  extracts debug fields from `GpuGlobalPC` into a separate VA (`GpuDebugPC`).

---

## Step 7 — Expose TLAS VA to shaders (active)

Replace `_reserved[0..1]` in `GpuGlobalPC` with `u64 tlasVa`. Set it each frame
from `vkof::acceleration_structure_device_address(tlas)`.

In resolve.comp add `GL_EXT_ray_query` and cast the VA to
`accelerationStructureEXT` via the `GL_EXT_ray_tracing_position_fetch` /
`GL_EXT_buffer_reference` path:

```glsl
accelerationStructureEXT tlas = accelerationStructureEXT(pc.global.tlasVa);
rayQueryEXT rq;
rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFF,
    worldPos + worldNormal*0.01, 0.001, toLight, dist);
rayQueryProceedEXT(rq);
bool inShadow = (
    rayQueryGetIntersectionTypeEXT(rq, true)
    != gl_RayQueryCommittedIntersectionNoneEXT
);
```

---

## Step 8 — DDGI probe grid

### Data structures (shared header `app/shared/ddgi-shared.h`)

```c
struct GpuDdgiGrid {
    f32v3 origin;
    u32 probeCountX;
    f32v3 spacing;
    u32 probeCountY;
    u32 probeCountZ;
    u32 raysPerProbe;     // e.g. 128
    u32 irradianceRes;    // texels per probe in irradiance atlas (e.g. 6)
    u32 depthRes;         // texels per probe in depth atlas (e.g. 14)
};
```

Two persistent (non-transient) images:
- Irradiance atlas: `(irradianceRes+2) * probeCountX*probeCountY*probeCountZ`
  pixels wide, R16G16B16A16_SFLOAT
- Depth atlas: same tiling, R16G16_SFLOAT (mean/mean² for Chebyshev)

`VA(GpuDdgiGrid) ddgiGrid` added to `GpuGlobalPC` (another reserved slot).

---

## Step 9 — Probe ray dispatch

Compute shader `ddgi_trace.comp`:
- One workgroup per probe; `raysPerProbe` invocations per workgroup.
- Each invocation: generate a ray direction (spherical Fibonacci or random
  rotated), fire a `rayQueryEXT` against the TLAS.
- On hit: fetch surface data the same way resolve.comp does (model indirect
  buffer → material → base color texture sample). Add emissive.
- Accumulate per-probe radiance into a temporary buffer (one vec4 per
  ray, or reduce in shared memory).
- Write blended result into irradiance atlas and depth atlas.

---

## Step 10 — Probe update (temporal blend)

Separate compute pass or done in the same shader:
- Blend new radiance with previous frame's irradiance using hysteresis
  factor (~0.97 stable, lower on camera cut).
- Update depth atlas with new mean depth / mean² depth per probe.

---

## Step 11 — Probe sample in resolve.comp

Replace ambient term `bsdfMaterial.albedo * 0.02f` with proper DDGI lookup:
- Find 8 surrounding probes for `worldPos`.
- For each: sample irradiance atlas at oct-encoded `worldNormal` direction.
- Weight by trilinear + visibility (Chebyshev test against depth atlas).
- Normalize and multiply by `albedo / PI`.
