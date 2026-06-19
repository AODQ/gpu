# mor

glTF scene loader built on meshoptimizer. Produces GPU-ready meshlet buffers accessible via BDA.

Meshlets are built with `meshopt_buildMeshlets`: up to 64 vertices and 124 triangles each. Each meshlet stores its own instance index so a single flat draw covers the whole scene.

---

## CPU scene

```cpp
mor::Scene * scene = mor::scene_create();
mor::scene_load_gltf(scene, "/path/to/scene.gltf");

u32 instances = mor::scene_instance_count(scene);
u32 meshlets = mor::scene_meshlet_count(scene);
u32 vertices = mor::scene_vertex_count(scene);
```

---

## GPU upload

```cpp
mor::GpuScene gpu = mor::scene_gpu_upload(scene);
mor::scene_destroy(scene);

u32 meshletCount = mor::scene_gpu_meshlet_count(gpu);
mor::Buffers bufs = mor::scene_gpu_buffers(gpu);

// later:
mor::scene_gpu_destroy(gpu);
```

`mor::Buffers` contains six `u64` BDAs:

| field | content |
|---|---|
| `meshlets` | `Meshlet[]` — vertex/triangle offsets + instance index |
| `instances` | `Instance[]` — world transform + meshlet range |
| `positions` | `f32v3[]` — vertex positions |
| `attributes` | `VertexAttr[]` — normals, UVs |
| `meshletVerts` | `u32[]` — global vertex indices per meshlet |
| `meshletTris` | `u8[]` — local triangle indices (3 per triangle) |

---

## Shared CPU/GPU types (`mor_shared.h`)

Included by both C++ and GLSL. In GLSL, include after extension declarations:

```glsl
#extension GL_EXT_buffer_reference    : require
#extension GL_EXT_scalar_block_layout : require
#include "mor/mor_shared.h"
```

```glsl
MeshletBuf meshletBuf = MeshletBuf(pc.draw.meshlets);
Meshlet m = meshletBuf.data[gl_WorkGroupID.x];
PositionBuf posBuf = PositionBuf(pc.draw.positions);
```

---

## Mesh shader pattern

One workgroup per meshlet, `local_size_x = 64`. Each invocation handles one vertex; triangles need a second pass since meshlets can have up to 124:

```glsl
SetMeshOutputsEXT(m.vertexCount, m.triangleCount);
uint i = gl_LocalInvocationID.x;

if (i < m.vertexCount) {
	uint globalIdx = vertBuf.data[m.vertexOffset + i];
	gl_MeshVerticesEXT[i].gl_Position = viewProj * vec4(posBuf.data[globalIdx], 1.0);
}
if (i < m.triangleCount) {
	uint base = m.triangleOffset + i * 3;
	gl_PrimitiveTriangleIndicesEXT[i] = uvec3(
		triBuf.data[base + 0], triBuf.data[base + 1], triBuf.data[base + 2]
	);
}
// second pass — covers triangles 64-123
uint j = i + 64u;
if (j < m.triangleCount) {
	uint base = m.triangleOffset + j * 3;
	gl_PrimitiveTriangleIndicesEXT[j] = uvec3(
		triBuf.data[base + 0], triBuf.data[base + 1], triBuf.data[base + 2]
	);
}
```

Draw the whole scene in one call:

```cpp
vkof::cmd_draw({
	.cmd = cmd,
	.pipeline = pipe,
	.pushconstant = { (u8 const *)&drawPC, sizeof(drawPC) },
	.vertexCount = meshletCount,
	.instanceCount = 1,
});
```
