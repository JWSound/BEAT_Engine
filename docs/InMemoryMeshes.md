# In-memory mesh transport

A compiled mesh supplies exactly one source: a nonempty `file`, or `file: ""`
with `mesh_data`. File requests remain compatible. Clients negotiate
`contracts.mesh_data: [1]` and `request_transports: ["file", "inline_json"]`.

`EngineWorker.submit(request_dict)` sends `{request_inline: request_dict, ...}`
over the existing stdin connection. `submit(Path(...))` retains request-file
behavior. Inline commands support solve. Compatibility is checked before sending;
status/result/completion events are unchanged. Cancellation without a marker
file requires terminating the worker.

## Packed mesh schema 1

```json
{
  "schema_version": 1,
  "points": {"dtype": "<f8", "shape": [3, 3], "data": "<base64>"},
  "cells": [{
    "type": "triangle",
    "connectivity": {"dtype": "<i8", "shape": [1, 3], "data": "<base64>"},
    "physical_tags": {"dtype": "<i8", "shape": [1], "data": "<base64>"}
  }],
  "physical_names": {"wall": [1, 2]}
}
```

Arrays are row-major little-endian buffers, base64 encoded. Indices are zero-based,
tags are positive, and every used tag must have a name in the matching dimension.
Blocks must be nonempty, preserve order, and have matching tag counts. Coordinates
must be finite; indices must be in bounds without repeated vertices per element.
Units and transforms use resource `scale_to_m` and `translation_m`. Transport
performs no coordinate/topology deduplication.

Supported cells: triangle (3), triangle6 (6), tetra (4), tetra10 (10).
BEM accepts only triangle; FEM accepts consistent P1 or P2 volume/surface order.
Public quadratic ordering is meshio/VTK; tetra10 nodes 9 and 10 are exchanged to
obtain the kernel's Gmsh ordering. Assignments, ports, outputs and solver options
remain in the existing compiled-system/request contract.

Python validation uses only the standard library. Julia builds the same
BoundaryMesh/VolumeMesh as the file loaders. Memory provenance records source
and a hash of the received payload. This is serialized transport, not shared
memory; no latency gain is claimed without benchmarks.

## Checks

```sh
python -m pytest
python -m ruff check src/beat_engine/*.py src/beat_engine/beat_contract tests
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/memory_mesh_tests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```
