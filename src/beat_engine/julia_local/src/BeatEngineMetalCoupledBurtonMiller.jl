# Combined coupled assembly. Reuse the exterior pair arithmetic and gather
# tables, retaining the flux coefficient as a matrix instead of applying a
# known q. No exterior launch or solver path is changed.
# Pair buffers hold A and -C (24 floats); final row scaling changes -C to C.

function _metal_coupled_flux_gather_kernel!(
    rhs_operator,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_columns,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    # The retained C column of this trial face: its DP0 dof, or its position among the active
    # faces when C is compacted (0: the face carries no flux, nothing to write).
    @inbounds dp0_column = Int32(element_columns[trial_index])
    dp0_column == Int32(0) && return nothing
    column_base = element_count * (trial_local - Int32(1))
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds value_re += blocks[pair + (local_row + Int32(17)) * pair_stride]
        @inbounds value_im += blocks[pair + (local_row + Int32(20)) * pair_stride]
        incident_position += Int32(1)
    end
    coefficient = Complex(value_re, value_im)
    @inbounds rhs_operator[Int64(row) + Int64(dp0_column - Int32(1)) * Int64(p1_count)] += coefficient
    return nothing
end

function _launch_metal_coupled_pair_kernels!(
    lhs,
    rhs_operator,
    blocks,
    cache::MetalRegularAssemblyCache,
    k,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z;
    element_columns=cache.element_dp0_dofs,
    lhs_panels=nothing,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Fused Metal assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_fused_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    packed = _metal_packed_pair_tables_for(cache)
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        # One transform per launch and gather, so each pair block is overwritten (no accumulation).
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_fused_pair_blocks_kernel!(
            blocks, packed.points4, packed.normals4, cache.areas, packed.curls4, cache.faces, tables.elements,
            cache.rule_points, cache.rule_weights,
            Int32(element_count), Int32(chunk_start), Int32(chunk_count), pair_stride,
            k, inv(k), Int32(cache.face_count), Val(packed.rule), Val(rule_count),
            pair_offsets, singular_trial_indices, skip_mode,
            trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
            Val(false),
        )
        stamp = _metal_gather_stage!("fused_pairs", timed, stamp)
        _metal_launch(
            _metal_coupled_flux_gather_kernel!,
            cache.p1_dof_count * chunk_count,
            rhs_operator,
            blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            element_columns,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_rhs", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        if lhs_panels !== nothing
            # Tiled A: the same gather once per column panel, each writing only its own columns.
            for (panel, column_first, column_last) in lhs_panels
                _metal_launch(
                    _metal_fused_lhs_gather_panel_kernel!,
                    cache.p1_dof_count * node_count,
                    panel, blocks, tables.element_positions, cache.vertex_offsets, cache.incident_elements,
                    cache.incident_local_indices, tables.chunk_nodes, tables.inc_offsets, tables.inc_packed,
                    Int32(node_start), Int32(node_count), Int32(element_count), pair_stride, p1_count,
                    Int32(column_first), Int32(column_last);
                    groupsize=groupsize,
                )
            end
            stamp = _metal_gather_stage!("fused_lhs", timed, stamp)
            continue
        end
        _metal_launch(
            _metal_fused_lhs_gather_kernel!,
            cache.p1_dof_count * node_count,
            lhs,
            blocks,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            tables.chunk_nodes,
            tables.inc_offsets,
            tables.inc_packed,
            Int32(node_start),
            Int32(node_count),
            Int32(element_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_lhs", timed, stamp)
    end
    return nothing
end

function _launch_metal_coupled_singular_kernels!(
    lhs,
    rhs_operator,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1);
    compact=nothing,
    lhs_panels=nothing,
)
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    T = typeof(k)
    sx = T(transform.signs[1])
    sy = T(transform.signs[2])
    sz = T(transform.signs[3])
    csx = T(transform.determinant * transform.signs[1])
    csy = T(transform.determinant * transform.signs[2])
    csz = T(transform.determinant * transform.signs[3])
    part_count = _metal_singular_part_count()
    # Same maps the four-operator path uses: the fused left-hand side lands on
    # the same P1-row/P1-column cells as the double layer and hypersingular.
    gather_tables = lock(() -> _metal_singular_gather_tables(regular_cache, singular_cache, part_count), _metal_packed_cache_lock)
    value_count = pair_count * part_count
    lhs_values = rhs_values = nothing
    try
        lhs_values = Metal.zeros(eltype(rhs_operator), value_count, 9)
        rhs_values = Metal.zeros(eltype(rhs_operator), value_count, 3)
        # The exterior fused path's packed singular blocks, grouped by rule point count.
        tables = _metal_fused_singular_tables_for(regular_cache, singular_cache)
        packed = _metal_packed_pair_tables_for(regular_cache)
        for (point_count, positions) in tables.groups
            group_count = length(positions)
            _metal_launch(
                _metal_fused_singular_packed_kernel!, group_count * part_count,
                lhs_values, rhs_values, positions,
                singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
                singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
                tables.rule_points4, singular_cache.rule_weights, tables.vertices4, packed.normals4, packed.curls4,
                k, inv(k), Int32(group_count), Int32(pair_count),
                sx, sy, sz, csx, csy, csz,
                Val(point_count), Val(part_count),
            )
        end
        rhs_map = if compact === nothing
            gather_tables.p1_dp0
        else
            compact.part_count == part_count || error(
                "Compact coupled C maps were built for $(compact.part_count) singular parts; now $(part_count).",
            )
            compact.singular[objectid(singular_cache)]
        end
        if lhs_panels !== nothing
            # Tiled A: each panel takes the entries of its own column range.
            block_map = gather_tables.p1_p1
            rows = Int64(regular_cache.p1_dof_count)
            for (panel, column_first, column_last) in lhs_panels
                _metal_launch(
                    _metal_singular_entry_gather_panel_kernel!,
                    block_map.entry_count,
                    panel, lhs_values,
                    block_map.entry_indices, block_map.contrib_offsets, block_map.contrib_values,
                    block_map.entry_count, pair_count, part_count,
                    (Int64(column_first) - 1) * rows + 1, Int64(column_last) * rows,
                )
            end
        end
        for (destination, values, block_map) in (
            (lhs, lhs_values, gather_tables.p1_p1),
            (rhs_operator, rhs_values, rhs_map),
        )
            destination === nothing && continue
            _metal_launch(
                _metal_singular_entry_gather_kernel!,
                block_map.entry_count,
                destination, values,
                block_map.entry_indices, block_map.contrib_offsets, block_map.contrib_values,
                block_map.entry_count, pair_count, part_count,
            )
        end
        Metal.synchronize()
    finally
        Metal.synchronize()
        lhs_values === nothing || Metal.unsafe_free!(lhs_values)
        rhs_values === nothing || Metal.unsafe_free!(rhs_values)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Column-panel ("tiled") assembly of A.
#
# One Metal buffer may hold at most `maxBufferLength` bytes (20.1 GB on a 32 GB M1 Max), and the
# stock combined path keeps A (8 N^2 bytes) in one buffer. When A is larger, A is allocated as one
# page-aligned host matrix and split into column panels, each wrapped without a copy as its own
# shared Metal buffer. The kernels below are the stock gather/write-back kernels with a column
# window: each launch writes only the columns of its panel, at panel-local offsets. Pair blocks are
# computed once per chunk and gathered into every panel, so the arithmetic, the summation order
# and therefore every entry are the same as in the single-buffer path. C stays one buffer (it is
# compact). Memory is the same 8 N^2 as before, plus at most one alignment group of columns.
# ---------------------------------------------------------------------------

function _metal_fused_lhs_gather_panel_kernel!(
    lhs,
    blocks,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    element_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
    column_first::Int32,
    column_last::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * node_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    node_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    node_position = node_start + node_local - Int32(1)
    @inbounds column = Int32(chunk_nodes[node_position])
    (column < column_first || column > column_last) && return nothing
    @inbounds chunk_first = Int32(inc_offsets[node_position])
    @inbounds chunk_stop = Int32(inc_offsets[node_position + Int32(1)]) - Int32(1)
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            pair = test_position + element_count * (trial_local - Int32(1))
            component = local_row + Int32(3) * (local_column - Int32(1))   # 1..9
            @inbounds value_re += blocks[pair + (component - Int32(1)) * pair_stride]
            @inbounds value_im += blocks[pair + (component + Int32(8)) * pair_stride]
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    @inbounds lhs[Int64(row) + Int64(column - column_first) * Int64(p1_count)] += Complex(value_re, value_im)
    return nothing
end


# The singular write-back of A restricted to one panel: entries are linear column-major indices
# into the full N x N matrix; those inside `entry_first:entry_last` are written at panel offsets.
function _metal_singular_entry_gather_panel_kernel!(
    operator,
    values,
    entry_indices,
    contrib_offsets,
    contrib_values,
    entry_count,
    pair_count,
    part_count,
    entry_first::Int64,
    entry_last::Int64,
)
    index = _metal_global_linear_index()
    index > entry_count && return nothing
    @inbounds begin
        entry = Int64(entry_indices[index])
        (entry < entry_first || entry > entry_last) && return nothing
        position = Int(contrib_offsets[index])
        position_stop = Int(contrib_offsets[index + 1]) - 1
        total = zero(eltype(values))
        while position <= position_stop
            value_index = Int(contrib_values[position])
            part = 1
            while part <= part_count
                total += values[value_index]
                value_index += pair_count
                part += 1
            end
            position += 1
        end
        operator[entry - entry_first + 1] += total
    end
    return nothing
end

# The sparse identity add restricted to one panel (same arithmetic as the stock scatter kernel).
function _metal_sparse_scatter_panel_kernel!(destination, rows, columns, values, alpha, column_first::Int, column_last::Int)
    index = _metal_global_linear_index()
    if index <= length(values)
        @inbounds column = Int(columns[index])
        if column_first <= column <= column_last
            @inbounds row = Int(rows[index])
            @inbounds value = alpha * values[index]
            @inbounds destination[row, column - column_first + 1] += value
        end
    end
    return nothing
end

"""
`BLAB_METAL_COUPLED_PANEL_COLUMNS`: `auto` (default) assembles A in column panels only when it does
not fit one Metal buffer; an integer forces panels of about that many columns (rounded to page
alignment; for validation against the single-buffer path); `off` never splits (A larger than
`maxBufferLength` then fails as in the stock engine).
"""
function _metal_coupled_panel_plan(n::Int, ::Type{E}) where {E}
    setting = lowercase(strip(get(ENV, "BLAB_METAL_COUPLED_PANEL_COLUMNS", "auto")))
    setting == "off" && return nothing
    page = Int(ccall(:getpagesize, Cint, ()))
    column_bytes = n * sizeof(E)
    # Panel starts and lengths must be whole pages for a no-copy wrap: columns come in groups of `step`.
    step = page ÷ gcd(column_bytes, page)
    limit = Int(Metal.device().maxBufferLength) - page
    requested = if setting == "auto"
        column_bytes * n <= limit && return nothing
        limit ÷ column_bytes
    else
        value = tryparse(Int, setting)
        value === nothing && error("BLAB_METAL_COUPLED_PANEL_COLUMNS must be auto, off, or a column count; got $(repr(setting)).")
        min(max(value, 1), limit ÷ column_bytes)
    end
    step * column_bytes <= limit ||
        error("A Metal buffer cannot hold even $(step) columns of $(n) rows; panel assembly is impossible.")
    # A forced width below one alignment group still runs the panel path (one group per panel).
    width = max(step, (requested ÷ step) * step)
    width >= n && setting == "auto" && return nothing
    firsts = collect(1:width:n)
    panels = [(first, min(first + width - 1, n)) for first in firsts]
    # The last panel is wrapped up to a whole page: pad the host allocation by < step columns.
    padded_columns = panels[end][1] - 1 + cld(panels[end][2] - panels[end][1] + 1, step) * step
    return (panels=panels, padded_columns=padded_columns, page=page, step=step)
end

# A page-aligned, zeroed host matrix of n x padded_columns, freed by the GC (posix_memalign + free).
function _metal_aligned_host_matrix(::Type{E}, n::Int, padded_columns::Int, page::Int) where {E}
    bytes = n * padded_columns * sizeof(E)
    pointer_ref = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(:posix_memalign, Cint, (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t), pointer_ref, page, bytes)
    status == 0 || throw(OutOfMemoryError())
    storage = unsafe_wrap(Array, Ptr{E}(pointer_ref[]), (n, padded_columns); own=true)
    Threads.@threads for column in 1:padded_columns
        @inbounds fill!(view(storage, :, column), zero(E))
    end
    return storage
end

# Wrap columns `first:last` (padded to whole pages for the last panel) of `storage` as a shared Metal buffer.
function _metal_wrap_panel(storage::Matrix{E}, first::Int, last::Int, step::Int) where {E}
    n = size(storage, 1)
    columns = cld(last - first + 1, step) * step
    first + columns - 1 <= size(storage, 2) || error("Panel wrap exceeds the padded host matrix.")
    slice = unsafe_wrap(Array, pointer(storage, (first - 1) * n + 1), (n, columns))
    # Shared storage is the default of a no-copy wrap; `storage` itself must keep the memory alive
    # for as long as the panel is in use (the caller holds it in the result tuple).
    return unsafe_wrap(MtlArray{E,2}, slice, (n, columns))
end

# Unsupported diagnostic configurations keep their original operator kernels.
function metal_coupled_combined_support_reason(prepared, ::Type{T}) where {T}
    T === Float32 || return "combined Metal assembly requires Float32"
    _normalized_metal_assembly_mode(nothing) == :native || return "host-staged assembly requested"
    _normalized_metal_regular_kernel_mode() == :pair_gather || return "reference regular kernel mode requested"
    _normalized_metal_singular_mode() == :native || return "host singular mode requested"
    _normalized_metal_singular_writeback() == :gather || return "singular scatter write-back requested"
    cache = prepared.device_cache
    cache isa MetalRegularAssemblyCache || return "no native Metal regular cache"
    cache.rule_count in (1, 3, 6) || return "combined kernels support only 1-, 3-, or 6-point rules"
    prepared.device_singular_cache isa MetalSingularCorrectionCache ||
        return "no native Metal singular cache"
    prepared.symmetry_mode in (:off, :x, :xy) || return "unsupported coupled symmetry mode"
    cache.symmetry_mode == prepared.symmetry_mode ||
        return "regular cache symmetry does not match the requested symmetry"
    return nothing
end

"""
    build_metal_coupled_identity_cache(prepared, T; active_columns=nothing)

The identity scatter maps of the combined assembly. With `active_columns` (sorted DP0 dofs that
carry interface, motion or prescribed flux) C is compacted to those columns: the pair gather skips
every other trial face, the singular write-back maps and the `p1_dp0` identity are restricted to
them, and C costs `N × length(active_columns)` instead of `N × F`.
"""
function build_metal_coupled_identity_cache(prepared, ::Type{T}; active_columns=nothing) where {T}
    pp = build_metal_sparse_scatter_cache(sparse(Complex{T}.(prepared.identity_p1_p1)))
    pq = nothing
    compact = nothing
    try
        identity_pq = prepared.identity_p1_dp0
        if active_columns !== nothing
            compact = _metal_coupled_compact_columns(prepared, active_columns)
            identity_pq = identity_pq[:, compact.active]
        end
        pq = build_metal_sparse_scatter_cache(sparse(Complex{T}.(identity_pq)))
        return (p1_p1=pp, p1_dp0=pq, compact=compact)
    catch
        release_metal_sparse_scatter_cache!(pp)
        pq === nothing || release_metal_sparse_scatter_cache!(pq)
        compact === nothing || _release_metal_coupled_compact!(compact)
        rethrow()
    end
end

function _metal_coupled_compact_columns(prepared, active_columns)
    cache = prepared.device_cache
    dp0_count = prepared.dp0.global_dof_count
    p1_count = prepared.p1.global_dof_count
    active = sort!(unique!(collect(Int, active_columns)))
    all(column -> 1 <= column <= dp0_count, active) || error("Active C columns outside 1:$(dp0_count).")
    column_map = zeros(Int32, dp0_count)
    column_map[active] .= Int32.(1:length(active))
    element_columns = Int32[dof >= 1 ? column_map[dof] : Int32(0) for dof in Array(cache.element_dp0_dofs)]
    part_count = _metal_singular_part_count()
    singular = Dict{UInt,MetalSingularGatherMap}()
    device_element_columns = MtlArray(element_columns)
    try
        for singular_cache in (prepared.device_singular_cache, cache.image_singular_caches...)
            singular_cache.pair_count == 0 && continue
            haskey(singular, objectid(singular_cache)) && continue
            tables = lock(() -> _metal_singular_gather_tables(cache, singular_cache, part_count), _metal_packed_cache_lock)
            singular[objectid(singular_cache)] = _metal_compact_gather_map(tables.p1_dp0, column_map, p1_count)
        end
    catch
        Metal.unsafe_free!(device_element_columns)
        foreach(_metal_release_singular_gather_map!, values(singular))
        rethrow()
    end
    return (active=active, column_map=column_map, element_columns=device_element_columns,
            singular=singular, part_count=part_count, dp0_count=dp0_count)
end

# Restrict a P1 x DP0 singular write-back map to the active columns, renumbering the destination
# cells for the compacted N x |active| matrix. Contributions keep their value indices.
function _metal_compact_gather_map(map::MetalSingularGatherMap, column_map::Vector{Int32}, p1_count::Int)
    entries = Array(map.entry_indices)
    offsets = Array(map.contrib_offsets)
    contributions = Array(map.contrib_values)
    kept_entries = Int64[]
    kept_offsets = Int32[1]
    kept_values = eltype(contributions)[]
    for entry in 1:map.entry_count
        linear = Int(entries[entry]) - 1
        row = linear % p1_count + 1
        column = Int(column_map[linear ÷ p1_count + 1])
        column == 0 && continue
        push!(kept_entries, Int64(row + (column - 1) * p1_count))
        append!(kept_values, view(contributions, Int(offsets[entry]):(Int(offsets[entry + 1]) - 1)))
        push!(kept_offsets, Int32(length(kept_values) + 1))
    end
    isempty(kept_values) && push!(kept_values, zero(eltype(contributions)))   # no empty device buffers
    return MetalSingularGatherMap(MtlArray(isempty(kept_entries) ? Int64[0] : kept_entries),
                                  MtlArray(kept_offsets), MtlArray(kept_values), nothing, length(kept_entries))
end

function _release_metal_coupled_compact!(compact)
    Metal.unsafe_free!(compact.element_columns)
    foreach(_metal_release_singular_gather_map!, values(compact.singular))
    return nothing
end

function release_metal_coupled_identity_cache!(cache)
    release_metal_sparse_scatter_cache!(cache.p1_p1)
    release_metal_sparse_scatter_cache!(cache.p1_dp0)
    compact = get(cache, :compact, nothing)
    compact === nothing || _release_metal_coupled_compact!(compact)
    return nothing
end

"""
    assemble_coupled_burton_miller_metal(mesh, prepared, k; identity_cache=nothing)

Retain A (P1 x P1) and C (P1 x DP0) using the exterior fused pair arithmetic.
The signed wavenumber, image normals/curls, direct and image Duffy rules, and
symmetry weights follow the existing exterior path. Images use separate launches.
Returned device storage belongs to the caller, including on private storage.
"""
function assemble_coupled_burton_miller_metal(
    mesh::BoundaryMesh{T}, prepared, k::T; identity_cache=nothing,
) where {T<:AbstractFloat}
    _require_metal!()
    reason = metal_coupled_combined_support_reason(prepared, T)
    reason === nothing || error("Combined Metal coupled assembly unavailable: $reason")
    cache = prepared.device_cache
    cache.symmetry_mode == prepared.symmetry_mode || error("Metal coupled cache symmetry mismatch.")
    signed_k = outgoing_wavenumber(k)
    isfinite(signed_k) && !iszero(signed_k) || error("Combined Metal assembly needs finite nonzero k.")
    owns_identity = identity_cache === nothing
    owns_identity && (identity_cache = build_metal_coupled_identity_cache(prepared, T))
    a = c = blocks = nothing
    panels = a_storage = nothing
    succeeded = false
    try
        storage = metal_operator_storage_mode()
        # Pair buffer for one gather chunk (24 floats per pair); the gather tables no longer own it.
        if !isempty(cache.element_indices)
            tables = _metal_fused_gather_tables(cache)
            blocks = MtlArray{Float32}(undef, _METAL_FUSED_COMPONENTS * length(cache.element_indices) * tables.chunk_size)
        end
        compact = get(identity_cache, :compact, nothing)
        element_columns = compact === nothing ? cache.element_dp0_dofs : compact.element_columns
        column_count = compact === nothing ? prepared.dp0.global_dof_count : length(compact.active)
        n = prepared.p1.global_dof_count
        plan = storage === Metal.SharedStorage ? _metal_coupled_panel_plan(n, Complex{T}) : nothing
        if plan === nothing
            a = Metal.zeros(Complex{T}, n, n; storage=storage)
        else
            # A larger than one Metal buffer: one page-aligned host matrix, wrapped panel by panel.
            a_storage = _metal_aligned_host_matrix(Complex{T}, n, plan.padded_columns, plan.page)
            panels = [(_metal_wrap_panel(a_storage, first, last, plan.step), first, last) for (first, last) in plan.panels]
        end
        c = Metal.zeros(Complex{T}, n, max(column_count, 1); storage=storage)
        _launch_metal_coupled_pair_kernels!(
            a, c, blocks, cache, signed_k, cache.vertex_offsets, cache.incident_elements, Int32(0),
            one(T), one(T), one(T), one(T), one(T), one(T); element_columns=element_columns, lhs_panels=panels,
        )
        for (transform, image_cache) in zip(cache.image_transforms, cache.image_singular_caches)
            _launch_metal_coupled_pair_kernels!(
                a, c, blocks, cache, signed_k, image_cache.pair_offsets, image_cache.trial_indices, Int32(1),
                T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                T(transform.determinant * transform.signs[1]),
                T(transform.determinant * transform.signs[2]),
                T(transform.determinant * transform.signs[3]);
                element_columns=element_columns, lhs_panels=panels,
            )
        end
        # Singular compact blocks carry A and -C too; reuse the P1/DP0 map,
        # with one owner per cell instead of the exterior known-flux row map.
        _launch_metal_coupled_singular_kernels!(a, c, cache, prepared.device_singular_cache, signed_k;
                                                compact=compact, lhs_panels=panels)
        for (transform, image_cache) in zip(cache.image_transforms, cache.image_singular_caches)
            _launch_metal_coupled_singular_kernels!(a, c, cache, image_cache, signed_k, transform;
                                                    compact=compact, lhs_panels=panels)
        end
        # Only the integral part is row weighted. The cached mass matrices
        # already include the orbit weights. Convert the retained -C to C here.
        if prepared.symmetry_mode == :off
            c .*= -one(T)
        else
            weights = MtlArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, prepared.symmetry_mode)))
            try
                if panels === nothing
                    a .*= reshape(weights, :, 1)
                else
                    foreach(panel -> (panel[1] .*= reshape(weights, :, 1)), panels)
                end
                c .= .-c .* reshape(weights, :, 1)
                Metal.synchronize()
            finally
                Metal.unsafe_free!(weights)
            end
        end
        if panels === nothing
            scatter_metal_sparse_to_dense!(a, identity_cache.p1_p1; alpha=Complex{T}(0.5), add=true)
        else
            identity = identity_cache.p1_p1
            for (panel, column_first, column_last) in panels
                isempty(identity.values) && break
                _metal_launch(_metal_sparse_scatter_panel_kernel!, length(identity.values),
                              panel, identity.rows, identity.columns, identity.values,
                              convert(Complex{T}, Complex{T}(0.5)), column_first, column_last)
            end
        end
        scatter_metal_sparse_to_dense!(c, identity_cache.p1_dp0;
            alpha=T(0.5) * burton_miller_coupling(k), add=true)
        Metal.synchronize()
        succeeded = true
        if panels !== nothing
            # The panels write straight into host memory: A is that memory, no copy.
            a = unsafe_wrap(Array, pointer(a_storage), (n, n))
            return (a=a, c=c, on_gpu=true, gpu_backend=:metal, assembly_mode=:metal_coupled_combined,
                    active_columns=compact === nothing ? nothing : compact.active,
                    dp0_count=prepared.dp0.global_dof_count,
                    a_panels=[panel[1] for panel in panels], a_storage=a_storage)
        end
        return (a=a, c=c, on_gpu=true, gpu_backend=:metal, assembly_mode=:metal_coupled_combined,
                active_columns=compact === nothing ? nothing : compact.active,
                dp0_count=prepared.dp0.global_dof_count)
    finally
        Metal.synchronize()
        owns_identity && release_metal_coupled_identity_cache!(identity_cache)
        blocks === nothing || Metal.unsafe_free!(blocks)
        if !succeeded
            a === nothing || Metal.unsafe_free!(a)
            c === nothing || Metal.unsafe_free!(c)
            panels === nothing || foreach(panel -> Metal.unsafe_free!(panel[1]), panels)
        end
    end
end

# Always transfer ownership to the host tuple, matching metal_host_operators.
function metal_host_coupled_burton_miller(combined)
    get(combined, :on_gpu, false) || return combined
    Metal.synchronize()
    if hasproperty(combined, :a_storage)
        # Tiled A already lives in host memory; the panels and the storage are its backing.
        c = Metal.is_shared(combined.c) ? unsafe_wrap(Array, combined.c) : Array(combined.c)
        return (a=combined.a, c=c, on_gpu=false,
                metal_backing=(a=combined.a_panels, c=combined.c, a_storage=combined.a_storage),
                active_columns=get(combined, :active_columns, nothing),
                dp0_count=get(combined, :dp0_count, size(c, 2)))
    end
    a = Metal.is_shared(combined.a) ? unsafe_wrap(Array, combined.a) : Array(combined.a)
    c = Metal.is_shared(combined.c) ? unsafe_wrap(Array, combined.c) : Array(combined.c)
    return (a=a, c=c, on_gpu=false, metal_backing=(a=combined.a, c=combined.c),
            active_columns=get(combined, :active_columns, nothing),
            dp0_count=get(combined, :dp0_count, size(c, 2)))
end

function release_metal_coupled_burton_miller!(combined)
    backing = get(combined, :metal_backing, combined)
    if hasproperty(combined, :a_panels) && !hasproperty(combined, :metal_backing)
        foreach(Metal.unsafe_free!, combined.a_panels)   # tiled A on the device tuple
    elseif backing.a isa AbstractVector
        foreach(Metal.unsafe_free!, backing.a)           # tiled A on the host tuple
    elseif backing.a isa MtlArray
        Metal.unsafe_free!(backing.a)
    end
    Metal.unsafe_free!(backing.c)
    return nothing
end
