"""Packed mesh-buffer validation; standard library only, no application models."""

import base64
import binascii
import math
import struct

MESH_DATA_VERSION = 1
CELL_WIDTHS = {"triangle": 3, "triangle6": 6, "tetra": 4, "tetra10": 10}


def decode_array(raw, dtype, tail):
    if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data"} or raw["dtype"] != dtype:
        raise ValueError("Invalid packed mesh array descriptor.")
    shape = raw["shape"]
    if (not isinstance(shape, list) or len(shape) != len(tail) + 1
            or shape[1:] != list(tail) or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("Invalid packed mesh array shape.")
    expected = math.prod(shape) * 8
    data = raw["data"]
    if not isinstance(data, str) or len(data) != 4 * ((expected + 2) // 3):
        raise ValueError("Packed mesh array byte count does not match shape.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid mesh base64 data.") from exc
    if len(decoded) != expected:
        raise ValueError("Packed mesh array byte count does not match shape.")
    return decoded


def validate_mesh_data(raw):
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "points", "cells", "physical_names"}:
        raise ValueError("Invalid mesh_data fields.")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != MESH_DATA_VERSION:
        raise ValueError("Unsupported mesh_data schema version.")
    points = decode_array(raw["points"], "<f8", (3,))
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<d", points)):
        raise ValueError("Mesh coordinates must be finite.")
    count = raw["points"]["shape"][0]
    names = raw["physical_names"]
    if not isinstance(names, dict):
        raise ValueError("physical_names must be an object.")
    groups = set()
    for name, value in names.items():
        if (not isinstance(name, str) or not name or not isinstance(value, list) or len(value) != 2
                or any(type(n) is not int for n in value) or value[0] <= 0 or value[1] not in (2, 3)):
            raise ValueError("Invalid physical group.")
        groups.add(tuple(value))
    cells = raw["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError("Mesh requires cell blocks.")
    for block in cells:
        if not isinstance(block, dict) or set(block) != {"type", "connectivity", "physical_tags"}:
            raise ValueError("Invalid cell block.")
        kind = block["type"]
        if not isinstance(kind, str) or kind not in CELL_WIDTHS:
            raise ValueError("Unsupported mesh cell type.")
        width = CELL_WIDTHS[kind]
        indices = decode_array(block["connectivity"], "<i8", (width,))
        tags = decode_array(block["physical_tags"], "<i8", ())
        if block["connectivity"]["shape"][0] != block["physical_tags"]["shape"][0]:
            raise ValueError("Mesh connectivity and tag counts differ.")
        for row in struct.iter_unpack("<" + "q" * width, indices):
            if any(index < 0 or index >= count for index in row) or len(set(row)) != width:
                raise ValueError("Invalid mesh vertex indices.")
        dimension = 3 if kind.startswith("tetra") else 2
        if any((tag, dimension) not in groups for (tag,) in struct.iter_unpack("<q", tags)):
            raise ValueError("Mesh cell tag has no physical name.")


def validate_mesh_sources(system):
    for mesh in system["meshes"]:
        if "mesh_data" in mesh:
            if mesh.get("file"):
                raise ValueError("A mesh must supply either file or mesh_data, not both.")
            validate_mesh_data(mesh["mesh_data"])
        elif not mesh.get("file"):
            raise ValueError("A file-backed mesh requires a filename.")
