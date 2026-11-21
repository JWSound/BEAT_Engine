# Opt-in CPU sampling of Galerkin Burton-Miller blocks. No production solve
# calls this adapter. In particular, it does not assemble a dense global A or C.

struct CpuGalerkinNearBlockData{T<:AbstractFloat}
    pairs::Dict{Tuple{Int,Int},Int}
    test_quadratures::Vector{Vector{BeatCpuRegularQuadratureData{T}}}
    trial_quadratures::Vector{Vector{BeatCpuRegularQuadratureData{T}}}
end

struct CpuGalerkinImageBlockData{T<:AbstractFloat}
    transform::SymmetryTransform
    elements::Vector{BeatCpuElementData{T}}
    quadrature::Vector{BeatCpuRegularQuadratureData{T}}
    singular_pairs::Dict{Tuple{Int,Int},SingularCorrectionPair{T}}
    singular_rules::Vector{DuffyRule{T}}
    near::Vector{CpuGalerkinNearBlockData{T}}
end

"""
    CpuGalerkinBlockSampler

An immutable description of one frequency's CPU Galerkin operators. The stored
geometry, incidence lists, singular/near rules, and sparse mass matrices contain
no dense global operator. Sampling may reorder sums relative to dense assembly;
agreement is numerical, not bitwise. This is a block-entry adapter, not an
H-matrix builder, iterative solver, or GPU implementation.

Use `build_cpu_galerkin_block_sampler` to construct it. The phasor convention is
captured at construction, so later changes to the worker convention do not alter
an existing sampler. Input meshes, spaces, rules, and correction caches must not
be mutated during construction or use.
"""
struct CpuGalerkinBlockSampler{T<:AbstractFloat}
    pressure_count::Int
    neumann_count::Int
    wavenumber::T
    coupling::Complex{T}
    phasor::String
    elements::Vector{BeatCpuElementData{T}}
    quadrature::Vector{BeatCpuRegularQuadratureData{T}}
    images::Vector{CpuGalerkinImageBlockData{T}}
    p1_incidence::Vector{Vector{Int}}
    dp0_elements::Vector{Int}
    row_weights::Vector{T}
    mass_pp::SparseMatrixCSC{T,Int}
    mass_pq::SparseMatrixCSC{T,Int}
end

function _cpu_block_validate_rule(rule::TriangleRule)
    isempty(rule.weights) && throw(ArgumentError("Block sampling needs a nonempty triangle rule."))
    length(rule.points) == length(rule.weights) ||
        throw(ArgumentError("Triangle-rule points and weights have different lengths."))
    all(isfinite, rule.weights) && all(point -> all(isfinite, point), rule.points) ||
        throw(ArgumentError("Triangle-rule points and weights must be finite."))
    return nothing
end

function _cpu_block_sparse_mass(mesh::BoundaryMesh{T}, p1, dp0, rule, row_weights) where {T}
    pp_rows, pp_cols, pq_rows, pq_cols = Int[], Int[], Int[], Int[]
    pp_values, pq_values = T[], T[]
    sizehint!(pp_rows, 9 * length(mesh.faces))
    sizehint!(pp_cols, 9 * length(mesh.faces))
    sizehint!(pp_values, 9 * length(mesh.faces))
    sizehint!(pq_rows, 3 * length(mesh.faces))
    sizehint!(pq_cols, 3 * length(mesh.faces))
    sizehint!(pq_values, 3 * length(mesh.faces))
    for face in eachindex(mesh.faces)
        pp = l2_identity_element_matrix(mesh.areas[face], :p1, :p1, rule)
        pq = l2_identity_element_matrix(mesh.areas[face], :p1, :dp0, rule)
        dofs = p1.local_to_global[face]
        for local_row in 1:3
            row = dofs[local_row]
            for local_col in 1:3
                push!(pp_rows, row)
                push!(pp_cols, dofs[local_col])
                push!(pp_values, row_weights[row] * pp[local_row, local_col])
            end
            push!(pq_rows, row)
            push!(pq_cols, dp0.local_to_global[face])
            push!(pq_values, row_weights[row] * pq[local_row, 1])
        end
    end
    return (
        sparse(pp_rows, pp_cols, pp_values, p1.global_dof_count, p1.global_dof_count),
        sparse(pq_rows, pq_cols, pq_values, p1.global_dof_count, dp0.global_dof_count),
    )
end

function _cpu_block_near_data(mesh::BoundaryMesh{T}, cache, image) where {T}
    cache isa NearCorrectionCache{T} ||
        throw(ArgumentError("CPU block sampling needs a host NearCorrectionCache with the mesh precision."))
    length(cache.pairs_by_test) == length(mesh.faces) ||
        throw(ArgumentError("Near-correction cache has a different face count."))
    cache.pair_count == length(cache.pairs) ||
        throw(ArgumentError("Near-correction cache has an inconsistent pair count."))
    length(cache.correction_rules) == length(cache.correction_orders) ||
        throw(ArgumentError("Near-correction rule metadata has inconsistent lengths."))
    foreach(_cpu_block_validate_rule, cache.correction_rules)
    pair_rules = Dict{Tuple{Int,Int},Int}()
    for pair in cache.pairs
        checkbounds(Bool, mesh.faces, pair.test_index) &&
            checkbounds(Bool, mesh.faces, pair.trial_index) ||
            throw(ArgumentError("Near-correction pair is outside the sampler mesh."))
        checkbounds(Bool, cache.correction_rules, pair.rule_index) ||
            throw(ArgumentError("Near-correction pair refers to a missing quadrature rule."))
        key = (pair.test_index, pair.trial_index)
        haskey(pair_rules, key) &&
            throw(ArgumentError("Near-correction cache contains a duplicate pair."))
        haskey(image.singular_pairs, key) &&
            throw(ArgumentError("A near-correction cache cannot replace a singular Duffy pair."))
        pair_rules[key] = pair.rule_index
    end
    test_quadratures = [_beat_cpu_regular_quadrature_data(mesh, rule) for rule in cache.correction_rules]
    trial_quadratures = image.transform.signs == SVector{3,Int}(1, 1, 1) ? test_quadratures :
        [_beat_cpu_reflect_regular_quadrature_data(data, image.transform) for data in test_quadratures]
    return CpuGalerkinNearBlockData(pair_rules, test_quadratures, trial_quadratures)
end

"""
    build_cpu_galerkin_block_sampler(mesh, p1, dp0, k, rule;
        singular_order=2, symmetry_mode=:off, identity_rule=rule,
        near_correction_cache=nothing, image_near_correction_cache=nothing)

Prepare lazy CPU sampling of `A = 0.5Mpp - D + alpha*H` and
`C = S + alpha*(adjD + 0.5Mpq)`, using BEAT's current phasor convention and
`alpha = burton_miller_coupling(k)`. All singular pairs are included.

Supports Float32/Float64, the existing vertex-based continuous P1 space, a
permuted face-based DP0 space, and off/X/XY symmetry. Near caches use the current
host API, including tuple/vector collections and different orders per pair.
Correction caches must have been built for this mesh with
`build_near_correction_cache`; validation does not establish mesh identity.
Every correction transform must occur in the requested symmetry mode; unsupported
configurations fail instead of silently dropping contributions. The separate
`identity_rule` permits the coupled solver's independent mass quadrature.

No dense global mass or BEM matrix is constructed. Preprocessing retains local
geometry/quadrature data and sparse mass terms. A requested large block can still
be large; the caller is responsible for its block-size budget.
"""
function build_cpu_galerkin_block_sampler(
    mesh::BoundaryMesh{T}, p1::P1Space, dp0::DP0Space, k::T, rule::TriangleRule{T};
    singular_order::Int=2,
    symmetry_mode::Symbol=:off,
    identity_rule::TriangleRule{T}=rule,
    near_correction_cache=nothing,
    image_near_correction_cache=nothing,
) where {T<:AbstractFloat}
    T in (Float32, Float64) || throw(ArgumentError("CPU block sampling supports Float32 and Float64."))
    isfinite(k) && !iszero(k) || throw(ArgumentError("Burton-Miller sampling needs finite nonzero k."))
    singular_order > 0 || throw(ArgumentError("Singular quadrature order must be positive."))
    symmetry_mode in (:off, :x, :xy) ||
        throw(ArgumentError("CPU block sampling currently supports only off, x, and xy symmetry."))
    p1.global_dof_count == length(mesh.vertices) && p1.local_to_global == mesh.faces ||
        throw(ArgumentError("CPU block sampling requires BEAT's vertex-based P1 space for this mesh."))
    face_count = length(mesh.faces)
    length(dp0.local_to_global) == face_count && dp0.global_dof_count == face_count &&
        sort(dp0.local_to_global) == collect(1:face_count) ||
        throw(ArgumentError("DP0 mapping must be a permutation of this mesh's faces."))
    all(area -> isfinite(area) && area > 0, mesh.areas) ||
        throw(ArgumentError("CPU block sampling requires finite positive triangle areas."))
    _cpu_block_validate_rule(rule)
    _cpu_block_validate_rule(identity_rule)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh, p1, dp0, rule; singular_order, symmetry_mode, threaded=false,
    )
    direct_singular = build_singular_correction_cache(mesh, singular_order)
    identity_transform = first(symmetry_transforms(:off))
    images = CpuGalerkinImageBlockData{T}[
        CpuGalerkinImageBlockData(
            identity_transform, cpu_cache.elements, cpu_cache.regular_quadrature,
            Dict((pair.test_index, pair.trial_index) => pair for pair in direct_singular.pairs),
            direct_singular.rules, CpuGalerkinNearBlockData{T}[],
        ),
    ]
    for index in eachindex(cpu_cache.image_transforms)
        singular = cpu_cache.image_singular_caches[index]
        pairs = Dict{Tuple{Int,Int},SingularCorrectionPair{T}}()
        for test_pairs in singular.pairs_by_test, pair in test_pairs
            pairs[(pair.test_index, pair.trial_index)] = pair
        end
        push!(images, CpuGalerkinImageBlockData(
            cpu_cache.image_transforms[index], cpu_cache.image_elements[index],
            cpu_cache.image_quadrature[index], pairs, singular.rules, CpuGalerkinNearBlockData{T}[],
        ))
    end
    for cache in (_near_correction_cache_tuple(near_correction_cache)...,
                  _near_correction_cache_tuple(image_near_correction_cache)...)
        cache isa NearCorrectionCache{T} ||
            throw(ArgumentError("CPU block sampling accepts only host near caches of the mesh precision."))
        transform = cache.trial_transform
        index = findfirst(image -> image.transform.signs == transform.signs &&
            image.transform.determinant == transform.determinant, images)
        isnothing(index) && throw(ArgumentError("Near-correction transform is absent from the active symmetry mode."))
        push!(images[index].near, _cpu_block_near_data(mesh, cache, images[index]))
    end

    incidence = [Int[] for _ in 1:p1.global_dof_count]
    dp0_elements = Vector{Int}(undef, face_count)
    for element in eachindex(mesh.faces)
        for dof in p1.local_to_global[element]
            push!(incidence[dof], element)
        end
        dp0_elements[dp0.local_to_global[element]] = element
    end
    weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
    mass_pp, mass_pq = _cpu_block_sparse_mass(mesh, p1, dp0, identity_rule, weights)
    return CpuGalerkinBlockSampler(
        p1.global_dof_count, dp0.global_dof_count, outgoing_wavenumber(k),
        burton_miller_coupling(k), String(phasor_convention()), cpu_cache.elements,
        cpu_cache.regular_quadrature, images, incidence, dp0_elements, weights, mass_pp, mass_pq,
    )
end

function _cpu_sample_pair(sampler::CpuGalerkinBlockSampler{T}, image, test_index, trial_index) where {T}
    test_data = sampler.elements[test_index]
    trial_data = image.elements[trial_index]
    test_quad = sampler.quadrature[test_index]
    trial_quad = image.quadrature[trial_index]
    key = (test_index, trial_index)
    pair = get(image.singular_pairs, key, nothing)
    if isnothing(pair)
        lhs, rhs = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, test_quad, trial_quad,
            dot(test_data.normal, trial_data.normal), T(4) * test_data.area * trial_data.area,
            sampler.wavenumber, sampler.coupling,
        )
    else
        duffy = image.singular_rules[pair.rule_index]
        lhs, rhs = _beat_cpu_bm_pair_blocks(
            test_data.vertices, trial_data.vertices, test_data.normal, trial_data.normal,
            test_data.curls, trial_data.curls, pair.normal_product, pair.jac_scale,
            sampler.wavenumber, sampler.coupling,
            duffy.test_points, duffy.trial_points, duffy.weights,
        )
    end
    for near in image.near
        rule_index = get(near.pairs, key, 0)
        rule_index == 0 && continue
        normal_product = dot(test_data.normal, trial_data.normal)
        jac_scale = T(4) * test_data.area * trial_data.area
        corrected_lhs, corrected_rhs = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, near.test_quadratures[rule_index][test_index],
            near.trial_quadratures[rule_index][trial_index], normal_product, jac_scale,
            sampler.wavenumber, sampler.coupling,
        )
        base_lhs, base_rhs = _beat_cpu_bm_regular_pair_blocks(
            test_data, trial_data, test_quad, trial_quad, normal_product, jac_scale,
            sampler.wavenumber, sampler.coupling,
        )
        lhs .+= corrected_lhs .- base_lhs
        rhs .+= corrected_rhs .- base_rhs
    end
    return lhs, rhs
end

function _cpu_block_positions(indices, count)
    positions = Dict{Int,Vector{Int}}()
    for (position, index) in enumerate(indices)
        1 <= index <= count || throw(BoundsError(1:count, index))
        push!(get!(positions, Int(index), Int[]), position)
    end
    return positions
end

function _cpu_block_faces(positions, incidence)
    faces = Int[]
    for dof in keys(positions)
        append!(faces, incidence[dof])
    end
    return sort!(unique!(faces))
end

"""
    sample_burton_miller_block!(destination, sampler, rows, columns; operator=:A)

Overwrite one selected A or C block. Indices are global P1/DP0 indices;
unsorted selections, repeated indices, and empty blocks follow ordinary Julia
matrix-indexing semantics. The output element type must match the sampler's
complex precision. No full operator is assembled or cached as a side effect.
"""
function sample_burton_miller_block!(
    destination::AbstractMatrix, sampler::CpuGalerkinBlockSampler{T},
    rows::AbstractVector{<:Integer}, columns::AbstractVector{<:Integer};
    operator::Symbol=:A,
) where {T}
    operator in (:A, :C) || throw(ArgumentError("Sampled operator must be :A or :C."))
    eltype(destination) === Complex{T} ||
        throw(ArgumentError("Block output precision must match the sampler's complex precision."))
    size(destination) == (length(rows), length(columns)) ||
        throw(DimensionMismatch("Block output size does not match the requested rows and columns."))
    Base.require_one_based_indexing(destination, rows, columns)
    row_positions = _cpu_block_positions(rows, sampler.pressure_count)
    column_positions = _cpu_block_positions(columns, operator === :A ? sampler.pressure_count : sampler.neumann_count)
    fill!(destination, zero(Complex{T}))
    isempty(rows) || isempty(columns) || begin
        test_faces = _cpu_block_faces(row_positions, sampler.p1_incidence)
        trial_faces = operator === :A ? _cpu_block_faces(column_positions, sampler.p1_incidence) :
            sort!(unique!([sampler.dp0_elements[dof] for dof in keys(column_positions)]))
        for image in sampler.images, test_index in test_faces, trial_index in trial_faces
            lhs, rhs = _cpu_sample_pair(sampler, image, test_index, trial_index)
            test_dofs = sampler.elements[test_index].p1_dofs
            trial = image.elements[trial_index]
            for local_row in 1:3
                output_rows = get(row_positions, test_dofs[local_row], nothing)
                isnothing(output_rows) && continue
                if operator === :A
                    for local_col in 1:3
                        output_cols = get(column_positions, trial.p1_dofs[local_col], nothing)
                        isnothing(output_cols) && continue
                        for column in output_cols, row in output_rows
                            destination[row, column] += lhs[local_row, local_col]
                        end
                    end
                else
                    output_cols = get(column_positions, trial.dp0_dof, nothing)
                    isnothing(output_cols) && continue
                    for column in output_cols, row in output_rows
                        # The local fused CPU routine returns the RHS operator -C.
                        destination[row, column] -= rhs[local_row]
                    end
                end
            end
        end
    end
    mass = operator === :A ? sampler.mass_pp : sampler.mass_pq
    mass_scale = operator === :A ? Complex{T}(T(0.5)) : T(0.5) * sampler.coupling
    for column in eachindex(columns), row in eachindex(rows)
        destination[row, column] = destination[row, column] * sampler.row_weights[rows[row]] +
            mass_scale * mass[rows[row], columns[column]]
    end
    return destination
end

"""Allocate only the requested block; see [`sample_burton_miller_block!`](@ref)."""
function sample_burton_miller_block(
    sampler::CpuGalerkinBlockSampler{T}, rows::AbstractVector{<:Integer},
    columns::AbstractVector{<:Integer}; operator::Symbol=:A,
) where {T}
    result = Matrix{Complex{T}}(undef, length(rows), length(columns))
    return sample_burton_miller_block!(result, sampler, rows, columns; operator)
end
