"""
    BeatEngineCoupledCondensed

Standalone CPU coupled FEM-BEM solver that eliminates the FEM interior onto the retained
interface before the dense solve.

`BeatEngineCoupled` assembles the entire coupled system densely, FEM volume block included.
That block is a P1 tetrahedral Helmholtz operator with roughly fifteen nonzeros per row, so
the monolithic formulation materializes a matrix that is about 0.06% dense and then takes a
dense LU of it. This solver instead forms the Schur complement

    S = A_ΓΓ - A_ΓI * A_II⁻¹ * A_IΓ

on the retained set `Γ = interface ∪ transducer surfaces`, and solves a system whose order is
`|Γ| + |BEM| + |interface| + 2 * |transducers|` rather than one that carries every interior
FEM vertex.

This is a deliberate fork of the monolithic solver rather than a branch inside it: the two are
expected to diverge further. It duplicates the coupled assembly rather than sharing it, and in
exchange carries no CUDA paths, no validation-diagnostic paths, and no monolithic formulation.
It reuses `BeatEngineCoupled` only for mesh types, cache preparation, and operator assembly —
the physics inputs, not the solve.

Precision is mixed on purpose. `A_II` is factored in `ComplexF64` whatever `T` is, and only the
assembled `S` is demoted to `Complex{T}`. `S` has poles at the eigenvalues of `A_II` — the
cavity modes with a pressure-release interface — and near one of those the loss of digits scales
like `1/η` in the bulk loss factor. `η = 0` is both the default and a shipped configuration, so
the interior solve carries the double-precision margin unconditionally. The interior is sparse,
which makes that margin nearly free.
"""
module BeatEngineCoupledCondensed

using LinearAlgebra, SparseArrays, StaticArrays, Statistics
using ..BeatEngineCore
using ..BeatEngineCoupled

include(joinpath(@__DIR__, "BeatEngineCondensedAssembly.jl"))

export assemble_condensed_regular_operators,
    wavelength_quadrature_order,
    prepare_condensed_coupled_cache,
    release_condensed_coupled_cache!,
    build_condensed_coupled_system,
    release_condensed_coupled_system!,
    solve_condensed_coupled_excitations,
    solve_condensed_coupled_system,
    solve_condensed_coupled_systems

"""
    _interior_partition(fem_system, interface_operators, retained_vertices)

Split FEM vertices into the eliminated interior and the retained set `Γ`, enforcing the
structural precondition that makes condensation valid: interface loads must have no support on
interior vertices. If they did, eliminating the interior would have to carry a flux contribution
into the retained/flux block, which this assembly does not form.
"""
function _interior_partition(
    fem_system::SparseMatrixCSC,
    interface_operators::InterfaceOperators,
    retained_vertices,
)
    fem_count = size(fem_system, 1)
    retained = Int.(collect(retained_vertices))
    retained_set = Set(retained)
    interior = [vertex for vertex in 1:fem_count if !(vertex in retained_set)]
    nnz(interface_operators.fem_load[interior, :]) == 0 || error(
        "FEM static condensation currently requires interface loads to have support only on interface nodes.",
    )
    return interior, retained
end

"""
    _densify_sparse_columns!(dense, source, columns)

Scatter `source[:, columns]` into `dense`, which must be at least as wide as `columns`.

`source[:, columns]` would build a whole intermediate `SparseMatrixCSC` before densifying it;
this writes the nonzeros straight into a buffer the caller reuses across blocks, which keeps the
Schur sweep from allocating two large arrays per block per task.
"""
function _densify_sparse_columns!(
    dense::AbstractMatrix{ComplexF64},
    source::SparseMatrixCSC{ComplexF64},
    columns,
)
    return BeatEngineCoupled._densify_sparse_columns!(dense, source, columns)
end

"""
    _schur_block_width(requested, retained_count) -> Int

Narrow `requested` until `Γ` splits into at least one block per thread, so the column sweep never
leaves threads idle. Never widens past `requested`, and never returns less than one.

`fld` rather than `cld`: the guarantee is on the block *count*, `cld(retained_count, width)`. Ten
columns across eight threads need a width of one to reach eight blocks -- `cld(10, 8)` is 2, which
yields five.
"""
function _schur_block_width(requested::Int, retained_count::Int)
    return BeatEngineCoupled._resolved_schur_block_size(requested, retained_count)
end

"""
    _coupled_mode(name, bem_backend=:cpu; metal_default=:auto) -> :off | :on | :auto

Resolve a condensed-coupled precision/optimization switch.

- **Unset:** `metal_default` on the Metal backend, `:off` everywhere else, so CPU requests keep
  the 0.1.4 behaviour unless a switch is set explicitly.
- **`auto`:** use it where the model allows; otherwise the established path.
- **`1`/`on`/`true`/`yes`:** require it and raise when the structure does not allow it.
- **`0`/`off`/`false`/`no`:** don't use it.

Anything else is an error rather than a silent default. The precision switches have no structural
precondition, so for them `auto` and `on` behave the same.
"""
function _coupled_mode(name::AbstractString, bem_backend::Symbol=:cpu; metal_default::Symbol=:auto)
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return bem_backend == :metal ? metal_default : :off
    value in ("1", "on", "true", "yes") && return :on
    value in ("0", "off", "false", "no") && return :off
    value == "auto" && return :auto
    error("Unsupported $name value: $value. Expected 1/on, 0/off or auto.")
end

_coupled_switch(name::AbstractString, bem_backend::Symbol=:cpu; metal_default::Symbol=:auto) =
    _coupled_mode(name, bem_backend; metal_default=metal_default) != :off

"""
`BLAB_COUPLED_DENSE_FLOAT64=1`: assemble and factor the dense coupled system in `ComplexF64`
whatever `T` is, keeping the Schur block in double precision. Off by default on every backend
(on Metal, `BLAB_COUPLED_DENSE_REFINEMENT` gives the same accuracy for less time).
"""
_dense_float64_enabled(bem_backend::Symbol=:cpu) =
    _coupled_switch("BLAB_COUPLED_DENSE_FLOAT64", bem_backend; metal_default=:off)

"""
`BLAB_COUPLED_DENSE_REFINEMENT=1`: assemble the dense coupled system in `ComplexF64` as
`BLAB_COUPLED_DENSE_FLOAT64` does, but factor a `ComplexF32` copy and recover the double-precision
solution by iterative refinement against the `ComplexF64` matrix (`RefinedDenseLU`). A solve that
stalls or does not reach the Float64 backward error within `DENSE_REFINEMENT_MAX_ITERATIONS` steps is redone with a
`ComplexF64` LU and says why. Takes precedence over `BLAB_COUPLED_DENSE_FLOAT64` for the factorization.
"""
_dense_refinement_enabled(bem_backend::Symbol=:cpu) = _coupled_switch("BLAB_COUPLED_DENSE_REFINEMENT", bem_backend)
_dense_double_assembly(bem_backend::Symbol=:cpu) =
    _dense_float64_enabled(bem_backend) || _dense_refinement_enabled(bem_backend)

const DENSE_REFINEMENT_MAX_ITERATIONS = 10

"""
    RefinedDenseLU(matrix)

Single-precision LU of a double-precision dense system, solved by iterative refinement:
`x ← x + F32⁻¹ (b - A x)` with the residual in `ComplexF64`. Each step contracts the error by
about `κ(A) eps(Float32)`; the Multi_region_SAWMOD systems (κ ≈ 5e6) converge in two steps.
The stopping test is LAPACK `zcgesv`'s backward error, `‖r‖∞ ≤ ‖x‖∞ ‖A‖∞ eps(Float64) √n` per
column (`‖A‖∞` the operator norm, `opnorm`), which is what a `ComplexF64` LU attains, whatever `κ`.
Convergence is tested before the stall rule, so a solve already at that level is accepted.

The `ComplexF64` LU is used instead, with the reason in `fallback_reason`, when
- an entry is outside the `Float32` range (narrowing would overflow),
- the `Float32` factorization is singular or not finite, or
- a solve stalls (the backward-error ratio fails to halve), produces a non-finite residual, or has
  not converged after `DENSE_REFINEMENT_MAX_ITERATIONS` steps.
The fallback factor is built once and serves every later solve. A matrix that is singular in
`Float64` too throws `SingularException`; non-finite matrices or right-hand sides throw
`ArgumentError`.

`matrix` is kept, not copied: the caller hands over ownership and must not modify it while the
factorization is in use (the coupled builder allocates it per system and never writes it again).
`iterations` is the most refinement steps any solve needed; `backward_error` is the worst accepted
column's ratio to the Float64 attainable backward error in the last solve (≤ 1 unless it came
from the fallback, whose ratio is recorded as computed).
"""
mutable struct RefinedDenseLU
    matrix::Matrix{ComplexF64}
    matrix_norm::Float64
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    fallback::Union{Nothing,LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{LinearAlgebra.BlasInt}}}
    iterations::Int
    fallback_reason::Union{Nothing,String}
    backward_error::Float64
end

function RefinedDenseLU(matrix::Matrix{ComplexF64})
    all(isfinite, matrix) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    refined = RefinedDenseLU(matrix, opnorm(matrix, Inf), nothing, nothing, 0, nothing, NaN)
    if maximum(abs, matrix; init=0.0) > floatmax(Float32)
        _dense_fall_back!(refined, "an entry is outside the Float32 range")
    else
        candidate = lu!(ComplexF32.(matrix); check=false)
        if issuccess(candidate) && all(isfinite, candidate.factors)
            refined.factor = candidate
        else
            _dense_fall_back!(refined, "the Float32 factorization is singular or not finite")
        end
    end
    return refined
end

function _dense_fall_back!(factorization::RefinedDenseLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason
    factorization.factor = nothing
    factorization.fallback = lu(factorization.matrix)
    return factorization
end

Base.size(factorization::RefinedDenseLU, dims...) = size(factorization.matrix, dims...)

# Worst column's backward error, in units of the double-precision attainable one.
function _dense_backward_error_ratio(factorization::RefinedDenseLU, residual, solution)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(size(factorization.matrix, 1))
    return maximum(
        column -> norm(view(residual, :, column), Inf) /
                  max(norm(view(solution, :, column), Inf) * threshold, floatmin(Float64)),
        axes(solution, 2);
        init=0.0,
    )
end

function _dense_residual!(residual, factorization::RefinedDenseLU, target, solution)
    copyto!(residual, target)
    mul!(residual, factorization.matrix, solution, -one(ComplexF64), one(ComplexF64))
    return residual
end

function Base.:\(factorization::RefinedDenseLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = ComplexF64.(rhs)
    residual = similar(target)
    if isnothing(factorization.fallback)
        solution = ComplexF64.(factorization.factor \ ComplexF32.(target))
        previous = Inf
        reason = nothing
        for iteration in 0:DENSE_REFINEMENT_MAX_ITERATIONS
            ratio = _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
            if !isfinite(ratio)
                reason = "non-finite residual after $iteration refinement steps"
                break
            end
            if ratio <= 1
                factorization.backward_error = ratio
                return solution
            end
            iteration == DENSE_REFINEMENT_MAX_ITERATIONS && break
            if ratio > previous / 2
                reason = "refinement stalled at $(ratio)x the Float64 backward error after $iteration steps"
                break
            end
            previous = ratio
            solution .+= ComplexF64.(factorization.factor \ ComplexF32.(residual))
            factorization.iterations = max(factorization.iterations, iteration + 1)
        end
        isnothing(reason) &&
            (reason = "refinement did not reach the Float64 backward error in $(DENSE_REFINEMENT_MAX_ITERATIONS) steps")
        _dense_fall_back!(factorization, reason)
    end
    solution = factorization.fallback \ target
    factorization.backward_error =
        _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
    return solution
end

"""
    dense_solver_diagnostics(system) -> Dict{String,Any}

The dense coupled factorization that ran: `dense_solver` (`lu_float32`, `lu_float64`,
`lu_float32_refined`, or `lu_float64_fallback` after a refinement fallback),
`dense_refinement_iterations`, `dense_refinement_fallback_reason` and `dense_refinement_backward_error`
(the last solve's worst column, in units of the Float64 attainable backward error).
"""
function dense_solver_diagnostics(system)
    factorization = hasproperty(system, :factorization) ? system.factorization : nothing
    if factorization isa RefinedDenseLU
        return Dict{String,Any}(
            "dense_solver" => isnothing(factorization.fallback) ? "lu_float32_refined" : "lu_float64_fallback",
            "dense_refinement_iterations" => factorization.iterations,
            "dense_refinement_fallback_reason" => factorization.fallback_reason,
            "dense_refinement_backward_error" => isnan(factorization.backward_error) ? nothing : factorization.backward_error,
        )
    end
    kind = factorization isa LinearAlgebra.LU ? "lu_" * lowercase(string(real(eltype(factorization)))) : nothing
    return Dict{String,Any}(
        "dense_solver" => kind,
        "dense_refinement_iterations" => 0,
        "dense_refinement_fallback_reason" => nothing,
        "dense_refinement_backward_error" => nothing,
    )
end

"""
`BLAB_COUPLED_FEM_FLOAT64=1`: under `precision=float32`, assemble the FEM stiffness, mass and
bulk-loss matrices in `Float64` (once per mesh) and hand the condensation a `ComplexF64` dynamic
stiffness. No effect under `precision=float64`.

`A_II = K - k² M` has a near-constant pressure mode whose eigenvalue scales like `k²`: the air
spring of an enclosed volume. `K` annihilates constants only up to its round-off, and a `Float32`
`K` leaves row sums near `eps(Float32) ‖K‖`, which that mode amplifies by `~1/(k h)²`. On
Multi_region_SAWMOD at 20 Hz this puts a 5e-5 relative error in the transducer mechanical block
and 1.2e-4 in every output, while the `A_II` factorization is already `Float64`. The matrices are
sparse and cached, so the double-precision assembly costs little.
"""
_fem_float64_enabled(bem_backend::Symbol=:cpu) = _coupled_switch("BLAB_COUPLED_FEM_FLOAT64", bem_backend)

"""
    _fem_system_float64(store, fem_mesh, prepared, frequency_hz, sound_speed, density)

The `ComplexF64` FEM dynamic stiffness for `BLAB_COUPLED_FEM_FLOAT64`: the same terms as the
`Complex{T}` one in `build_condensed_coupled_system`, from `Float64` matrices assembled on a
widened copy of `fem_mesh`.

- **Geometry:** Float64 arithmetic on the mesh's own (possibly Float32-rounded) coordinates, not
  higher-precision geometry. What matters for the constant mode is that `K` is assembled
  consistently in Float64, so its row sums vanish to Float64 round-off on that geometry.
- **Elements:** P1, exactly as `prepare_coupled_cache` assembles the cached system; the Float64
  stiffness and mass must have the cached stiffness's sparsity pattern, or this refuses.
- **Walls:** wall-impedance boundary masses are widened, not reassembled: they are positive
  boundary terms with no cancellation to protect.
- **Cache:** `store` (the cache's) keeps the matrices across frequencies, keyed by value on what
  they depend on (vertex coordinates, tetrahedra, per-vertex bulk-loss factors, wall matrices), so
  a mesh or loss change made in place is picked up.
"""
function _fem_system_float64(store, fem_mesh::VolumeMesh, prepared, frequency_hz, sound_speed, density)
    walls = [operator.matrix for operator in prepared.wall_impedance_operators]
    cached = !isnothing(store) && haskey(store, :matrices) &&
             store[:vertices] == fem_mesh.vertices && store[:tetrahedra] == fem_mesh.tetrahedra &&
             store[:bulk_loss] == prepared.bulk_loss_factor_by_vertex && store[:walls] == walls
    matrices = cached ? store[:matrices] : nothing
    if isnothing(matrices)
        mesh = VolumeMesh{Float64}(
            SVector{3,Float64}.(fem_mesh.vertices),
            fem_mesh.tetrahedra,
            fem_mesh.tetra_physical_tags,
            fem_mesh.boundary_faces,
            fem_mesh.boundary_physical_tags,
            fem_mesh.physical_names,
            fem_mesh.quadratic_tetrahedra,
            fem_mesh.quadratic_boundary_faces,
        )
        stiffness, mass = assemble_p1_fem_matrices(mesh)
        size(stiffness) == size(prepared.stiffness) &&
            stiffness.colptr == prepared.stiffness.colptr && stiffness.rowval == prepared.stiffness.rowval &&
            mass.colptr == prepared.stiffness.colptr && mass.rowval == prepared.stiffness.rowval ||
            error("Double-precision FEM matrices do not have the cached (P1) FEM system's structure.")
        matrices = (
            stiffness=stiffness,
            mass=mass,
            bulk_loss_mass=spdiagm(0 => Float64.(prepared.bulk_loss_factor_by_vertex)) * mass,
            walls=[SparseMatrixCSC{Float64,Int}(operator.matrix) for operator in prepared.wall_impedance_operators],
        )
        if !isnothing(store)
            store[:vertices] = copy(fem_mesh.vertices)
            store[:tetrahedra] = copy(fem_mesh.tetrahedra)
            store[:bulk_loss] = copy(prepared.bulk_loss_factor_by_vertex)
            store[:walls] = [copy(matrix) for matrix in walls]
            store[:matrices] = matrices
        end
    end
    omega = 2pi * Float64(frequency_hz)
    system = assemble_fem_dynamic_stiffness(
        matrices.stiffness,
        matrices.mass,
        omega / Float64(sound_speed);
        bulk_loss_mass=matrices.bulk_loss_mass,
    )
    for (operator, matrix) in zip(prepared.wall_impedance_operators, matrices.walls)
        admittance = miki_rigid_backed_surface_admittance(
            Float64(frequency_hz),
            Float64(sound_speed),
            Float64(density),
            Float64(operator.thickness_m),
            Float64(operator.flow_resistivity_pa_s_per_m2),
        )
        system -= neumann_scale(Float64(density), omega) * admittance .* matrix
    end
    return system
end

"""
    _build_condensation(fem_system, interface_operators, retained_vertices)

Factor the FEM interior with UMFPACK and form the dense Schur complement.

`S` is returned unnegated, in the same sign convention as `fem_system`, so it fills exactly the
slot the monolithic formulation fills with the full FEM block.

`A_IΓ` is densified a block of columns at a time rather than all at once: a transducer-heavy `Γ`
makes `interior_count × retained_count` a multi-gigabyte intermediate.

The blocks are swept in parallel. UMFPACK's solve is one triangular pair per right-hand side with
no BLAS-3 step and no internal threading, so the sweep dominates condensation by well over an
order of magnitude relative to the sparse factorization it consumes, and only the task split
scales it. Each block owns a disjoint column slice of the accumulator, so the only shared mutable
state is the factorization, which is split per task rather than locked.

`schur_block_columns` is not a BLAS-3 tile size: the solve is per column whatever the block width,
so a wide block buys no arithmetic and only costs locality. Peak scratch is
`2 * interior_count * schur_block_columns` complex entries *per task*; wide blocks measure no
faster than narrow ones for several times the scratch, so the default stays narrow.

The requested width is an upper bound, not the width used: `_schur_block_width` narrows it until
`Γ` splits into at least one block per thread. Without that, occupancy depends on the size of the
interface: a 500-node interface under a 256-column block yields two blocks, and six of eight
threads idle through the phase that dominates condensation.
"""
function _build_condensation(
    fem_system::SparseMatrixCSC{<:Complex},
    interface_operators::InterfaceOperators{T},
    retained_vertices;
    schur_block_columns::Int=32,
    schur_float64::Bool=false,
) where {T<:AbstractFloat}
    schur_block_columns > 0 || error("Schur block column count must be positive.")
    interior_vertices, retained = _interior_partition(
        fem_system,
        interface_operators,
        retained_vertices,
    )
    interior_count = length(interior_vertices)
    retained_count = length(retained)

    interior_system = SparseMatrixCSC{ComplexF64,Int}(fem_system[interior_vertices, interior_vertices])
    interior_retained = SparseMatrixCSC{ComplexF64,Int}(fem_system[interior_vertices, retained])
    retained_interior = SparseMatrixCSC{ComplexF64,Int}(fem_system[retained, interior_vertices])

    # UMFPACK performs symbolic and numeric factorization in one call, so the whole cost lands
    # in `factorization_s` and the analysis slot stays zero. If every FEM vertex is retained,
    # condensation is an exact no-op and there is no interior matrix to factor.
    factorization_started = time_ns()
    factorization = interior_count == 0 ? nothing : lu(interior_system)
    factorization_s = (time_ns() - factorization_started) / 1.0e9

    schur_started = time_ns()
    retained_system = Matrix{ComplexF64}(fem_system[retained, retained])
    schur_result = interior_count == 0 ?
                   (schur=retained_system, block_size=0, thread_count=1) :
                   BeatEngineCoupled._blocked_umfpack_schur_complement(
        factorization,
        interior_retained,
        retained_interior,
        retained_system;
        block_size=schur_block_columns,
    )
    # Kept double precision for a double-precision dense system; demoted otherwise.
    schur = schur_float64 ? schur_result.schur : Complex{T}.(schur_result.schur)
    schur_extraction_s = (time_ns() - schur_started) / 1.0e9

    return (
        backend=interior_count == 0 ? :cpu_noop : :cpu_umfpack,
        factorization=factorization,
        interior_system=interior_system,
        interior_retained=interior_retained,
        retained_interior=retained_interior,
        schur=schur,
        interior_vertices=interior_vertices,
        retained_vertices=retained,
        interior_count=interior_count,
        retained_count=retained_count,
        # The width actually swept, not the one requested.
        schur_block_columns=schur_result.block_size,
        schur_block_size=schur_result.block_size,
        schur_thread_count=schur_result.thread_count,
        timings=(
            analysis_s=0.0,
            factorization_s=factorization_s,
            schur_extraction_s=schur_extraction_s,
        ),
    )
end

function _release_condensation!(condensation)
    isnothing(condensation) && return nothing
    # UMFPACK holds its factors outside the Julia heap; release them with the system rather than
    # waiting for the finalizer, so a frequency sweep does not accumulate them.
    isnothing(condensation.factorization) || finalize(condensation.factorization)
    return nothing
end

"""
    _forward_schur(condensation, fem_rhs) -> (reduced_rhs, interior_rhs)

Reduce a FEM right-hand side onto `Γ`: `g_Γ = f_Γ - A_ΓI * A_II⁻¹ * f_I`. `f_I` is returned
because the backward substitution needs it and nothing else retains it.
"""
function _forward_schur(
    condensation,
    fem_rhs::AbstractMatrix{Complex{T}};
    result_type::Type{<:AbstractFloat}=T,
) where {T<:AbstractFloat}
    R = Complex{result_type}
    interior_rhs = ComplexF64.(fem_rhs[condensation.interior_vertices, :])
    retained_rhs = ComplexF64.(fem_rhs[condensation.retained_vertices, :])
    condensation.interior_count == 0 && return R.(retained_rhs), interior_rhs
    mul!(
        retained_rhs,
        condensation.retained_interior,
        condensation.factorization \ interior_rhs,
        -one(ComplexF64),
        one(ComplexF64),
    )
    return R.(retained_rhs), interior_rhs
end

"""
    _backward_schur(condensation, interior_rhs, retained_pressure)
        -> (fem_pressure, interior_residual)

Recover the eliminated interior, `u_I = A_II⁻¹ (f_I - A_IΓ u_Γ)`, and return the full FEM
pressure in mesh vertex order.

`interior_residual` is `‖A_II u_I + A_IΓ u_Γ - f_I‖` per excitation, relative to the largest of
those three terms. It costs one sparse matvec over blocks retained for the backward solve anyway.

It is normalized against the largest term rather than against `f_I` alone because a
voltage-driven transducer forces the system entirely through the electrical rows, leaving `f_I`
identically zero; dividing by `eps` there turns a converged solve into an O(1) reading.

Note what it does *not* measure: it validates the back-substitution, not the conditioning of
`S`. Near an interior resonance the recovered pressure can be visibly wrong while this residual
stays at round-off, so it is not a resonance detector.
"""
function _backward_schur(
    condensation,
    interior_rhs,
    retained_pressure::AbstractMatrix{Complex{T}},
) where {T<:AbstractFloat}
    retained_double = ComplexF64.(retained_pressure)
    interior_pressure = condensation.interior_count == 0 ?
                        copy(interior_rhs) :
                        condensation.factorization \
                        (interior_rhs - condensation.interior_retained * retained_double)
    interior_term = condensation.interior_system * interior_pressure
    retained_term = condensation.interior_retained * retained_double
    residual = interior_term + retained_term - interior_rhs

    interior_residual = zeros(T, size(residual, 2))
    for column in axes(residual, 2)
        scale = max(
            norm(view(interior_term, :, column)),
            norm(view(retained_term, :, column)),
            norm(view(interior_rhs, :, column)),
            eps(Float64),
        )
        interior_residual[column] = T(norm(view(residual, :, column)) / scale)
    end

    fem_pressure = zeros(
        Complex{T},
        condensation.interior_count + condensation.retained_count,
        size(retained_pressure, 2),
    )
    fem_pressure[condensation.interior_vertices, :] = Complex{T}.(interior_pressure)
    fem_pressure[condensation.retained_vertices, :] = retained_pressure
    return fem_pressure, interior_residual
end

"""
    wavelength_quadrature_order(areas, frequency_hz, sound_speed, base_order; ...)

Pick this frequency's regular quadrature order from `kh` against a mesh element-size statistic,
where `h = sqrt(area_stat)` and `kh = 2*pi*f/c * h`.

Owned by this solver rather than shared with the exterior path: this is the only solver that keys
its quadrature caches per order, and the exterior loop holds its device caches at a single rule.

`q1_max` defaults to 0.0, which disables the one-point tier -- `kh <= 0.0` is false for any
positive frequency. That tier is off deliberately. The hypersingular kernel decays like 1/r^3, so
its regular-pair integrand is far less smooth than the single layer's and a centroid rule
approximates it poorly; enabling it needs its own convergence study, not a default.
"""
function wavelength_quadrature_order(
    areas,
    frequency_hz::Real,
    sound_speed::Real,
    base_order::Int;
    mesh_stat::AbstractString="p90",
    q1_max::Real=0.0,
    q2_max::Real=2.0,
)
    q1_cutoff = Float64(q1_max)
    q2_cutoff = Float64(q2_max)
    q1_cutoff >= 0.0 || error("wavelength_kh_q1_max must be non-negative.")
    q2_cutoff > q1_cutoff || error("wavelength_kh_q2_max must exceed wavelength_kh_q1_max.")
    values = collect(Float64.(areas))
    isempty(values) && error("Cannot select wavelength quadrature order from an empty mesh.")
    area = if mesh_stat == "median"
        median(values)
    elseif mesh_stat == "p75"
        quantile(values, 0.75)
    elseif mesh_stat == "p90"
        quantile(values, 0.90)
    elseif mesh_stat == "max"
        maximum(values)
    else
        error("Unsupported wavelength mesh stat: $mesh_stat. Expected median, p75, p90, or max.")
    end
    element_length = sqrt(area)
    kh = Float64(2pi * frequency_hz / sound_speed) * element_length
    order = kh <= q1_cutoff ? 1 : kh <= q2_cutoff ? 2 : base_order
    return (
        order=order,
        base_order=base_order,
        mesh_stat=mesh_stat,
        area=area,
        length=element_length,
        kh=kh,
        q1_max=q1_cutoff,
        q2_max=q2_cutoff,
    )
end

"""
    _condensed_quadrature_bundle(bem_mesh, p1, dp0, order; singular_order, symmetry_mode)

Build the order-dependent half of a coupled cache for one quadrature order.

`BeatEngineCoupled.prepare_coupled_cache` is used unmodified for everything that does not depend
on the regular rule -- the FEM matrices, interface operators, P1/DP0 spaces and singular caches.
This adds only the parts that do, from the same public constructors that cache uses, so a
frequency sweep that changes order rebuilds those and nothing else.

The bundle owns its rule. `assemble_regular_galerkin_operators_cpu` compares `cpu_cache.rule` to
the rule it is handed, and `TriangleRule` defines no `==`, so that comparison is pointer identity.
It is a strict stale-cache guard and it only holds while a rule and the assembly cache built from
it travel together.
"""
function _condensed_quadrature_bundle(
    bem_mesh::BoundaryMesh{T},
    p1::P1Space,
    dp0::DP0Space,
    order::Int;
    singular_order::Int,
    symmetry_mode::Symbol,
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    rule = triangle_rule(T, order)

    assembly_started = time_ns()
    cpu_assembly_cache = bem_backend == :cpu ? build_beat_cpu_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    ) : nothing
    device_cache = bem_backend == :metal ? build_metal_regular_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    ) : nothing
    bem_cpu_assembly_cache_s = (time_ns() - assembly_started) / 1.0e9

    identity_started = time_ns()
    # The identity rule is clamped to order 2 and never follows a q1 regular rule down.
    # `l2_identity_element_matrix` integrates a degree-2 polynomial (P1xP1) or degree-1
    # (P1xDP0), and the 3-point order-2 rule is degree-2 exact -- so every order >= 2 gives
    # identical matrices, but the 1-point centroid rule does not: it returns area/9 uniformly
    # where the exact block is area/6 on the diagonal and area/12 off it. That would silently
    # degrade the Burton-Miller 0.5*I term rather than fail.
    identity_rule = order >= 2 ? rule : triangle_rule(T, 2)
    identity_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, p1, dp0, identity_rule, :p1, :p1; symmetry_mode=symmetry_mode,
    )
    identity_p1_dp0 = assemble_l2_identity_matrix(
        bem_mesh, p1, dp0, identity_rule, :p1, :dp0; symmetry_mode=symmetry_mode,
    )
    bem_identity_cache_s = (time_ns() - identity_started) / 1.0e9

    field_started = time_ns()
    cpu_field_cache = build_field_evaluation_cache(bem_mesh, rule; symmetry_mode=symmetry_mode)
    field_cache = bem_backend == :metal ?
                  build_metal_field_evaluation_cache(cpu_field_cache) : cpu_field_cache
    field_cache_s = (time_ns() - field_started) / 1.0e9

    return (
        order=order,
        rule=rule,
        cpu_assembly_cache=cpu_assembly_cache,
        device_cache=device_cache,
        identity_p1_p1=identity_p1_p1,
        identity_p1_dp0=identity_p1_dp0,
        field_cache=field_cache,
        timings=(
            bem_cpu_assembly_cache_s=bem_cpu_assembly_cache_s,
            bem_identity_cache_s=bem_identity_cache_s,
            field_cache_s=field_cache_s,
        ),
    )
end

"""
    prepare_condensed_coupled_cache(fem_mesh, bem_mesh, interface_map; regular_quadrature_orders, ...)

Wrap an unmodified `BeatEngineCoupled.prepare_coupled_cache` with one quadrature bundle per order
the sweep will select.

Every order is built eagerly. Coupled frequencies are not solved in ascending order -- the
live-plotting order emits both endpoints first and then the interior in van der Corput order -- so
a lazily grown cache would reach peak memory on the second frequency solved rather than at setup,
after the user has already seen a result and believes the run is healthy.
"""
function prepare_condensed_coupled_cache(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap;
    quadrature_order::Int=2,
    regular_quadrature_orders=nothing,
    singular_order::Int=2,
    symmetry_mode::Symbol=:off,
    retained_fem_vertices=interface_map.fem_vertex_indices,
    bulk_loss_factor_by_vertex=zeros(T, length(fem_mesh.vertices)),
    wall_impedances=NamedTuple[],
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    # The condensed solver's linear algebra is CPU-only; the BEM operators may
    # come from the CPU or from Metal, which hands them back as host matrices.
    bem_backend in (:cpu, :metal) ||
        error("Condensed coupled BEM backend must be :cpu or :metal; got $bem_backend.")
    base = prepare_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bem_backend=bem_backend,
        symmetry_mode=symmetry_mode,
        retained_fem_vertices=retained_fem_vertices,
        bulk_loss_factor_by_vertex=bulk_loss_factor_by_vertex,
        wall_impedances=wall_impedances,
    )
    # The base order is always carried, whether or not any frequency selects it: with the q1 tier
    # enabled and a base order of 4, every frequency can select 1 or 2 and leave the base out of
    # the requested set. The base bundle is free anyway -- it aliases what `prepare_coupled_cache`
    # already built -- and `build_condensed_coupled_system` validates requests against
    # `base_quadrature_order`, so the cache has to be able to answer for it regardless.
    requested = isnothing(regular_quadrature_orders) ? Int[] :
                Int.(collect(regular_quadrature_orders))
    all(order -> order >= 1, requested) ||
        error("Condensed coupled regular quadrature orders must be positive.")
    orders = sort(unique(vcat(requested, quadrature_order)))
    bundles = Dict{Int,Any}()
    # The base order's bundle is what `prepare_coupled_cache` already built; reusing those fields
    # keeps a single-order sweep byte-identical to calling the plain cache directly.
    bundles[quadrature_order] = (
        order=quadrature_order,
        rule=base.rule,
        cpu_assembly_cache=base.cpu_assembly_cache,
        device_cache=base.device_cache,
        identity_p1_p1=base.identity_p1_p1,
        identity_p1_dp0=base.identity_p1_dp0,
        field_cache=base.field_cache,
    )
    for order in orders
        order == quadrature_order && continue
        bundles[order] = _condensed_quadrature_bundle(
            bem_mesh,
            base.p1,
            base.dp0,
            order;
            singular_order=singular_order,
            symmetry_mode=base.symmetry_mode,
            bem_backend=bem_backend,
        )
    end
    # Extra bundles are real setup cost, so fold them into the fields the caller already reports
    # rather than leaving a mixed-order sweep looking as cheap as a single-order one. Field names
    # match `prepare_coupled_cache`'s timings exactly, so `cache.timings` reads the same either way.
    extra = [bundle.timings for (order, bundle) in bundles if order != quadrature_order]
    timings = merge(
        base.timings,
        (
            bem_cpu_assembly_cache_s=base.timings.bem_cpu_assembly_cache_s +
                                     sum(t -> t.bem_cpu_assembly_cache_s, extra; init=0.0),
            bem_identity_cache_s=base.timings.bem_identity_cache_s +
                                 sum(t -> t.bem_identity_cache_s, extra; init=0.0),
            field_cache_s=base.timings.field_cache_s +
                          sum(t -> t.field_cache_s, extra; init=0.0),
        ),
    )
    return (
        base=base,
        quadrature_bundles=bundles,
        base_quadrature_order=quadrature_order,
        singular_order=singular_order,
        timings=timings,
        # Double-precision FEM matrices for BLAB_COUPLED_FEM_FLOAT64.
        fem_float64_store=Dict{Symbol,Any}(),
    )
end

function release_condensed_coupled_cache!(cache)
    # Extra bundles (orders other than the base) own their own device caches
    # under Metal; host bundles are reclaimed by the collector. The base cache
    # still owns whatever `prepare_coupled_cache` allocated.
    if cache.base.bem_backend == :metal
        for (order, bundle) in cache.quadrature_bundles
            order == cache.base_quadrature_order && continue
            release_metal_regular_assembly_cache!(bundle.device_cache)
            release_metal_field_evaluation_cache!(bundle.field_cache)
        end
    end
    hasproperty(cache, :fem_float64_store) && empty!(cache.fem_float64_store)
    release_coupled_cache!(cache.base)
    return nothing
end

"""
    _stage_overlap_enabled(bem_backend) -> Bool

Whether to run the FEM static condensation concurrently with the BEM operator
assembly inside `build_condensed_coupled_system`.

The two stages are independent: the condensation reads `fem_system`, the
interface operators and the retained vertex list, none of which the BEM assembly
touches. Run in sequence, one processor idles for the other's duration — but
that only costs anything when the two stages use *different* processors. On
Metal the BEM assembly is on the GPU while the condensation is host UMFPACK, so
overlapping hides the shorter stage. On `:cpu` both are host code competing for
the same cores (the Schur complement already saturates them through
`_blocked_umfpack_schur_complement`), so overlapping buys nothing there.
Hence: Metal on, CPU off.

`BLAB_COUPLED_STAGE_OVERLAP` overrides the default: `auto`, `on`, or `off`. A
spawned condensation needs a thread of its own, so a single-threaded Julia
always runs the stages in sequence.
"""
function _stage_overlap_enabled(bem_backend::Symbol)
    requested = lowercase(strip(get(ENV, "BLAB_COUPLED_STAGE_OVERLAP", "auto")))
    requested in ("auto", "on", "off") || error(
        "Unsupported BLAB_COUPLED_STAGE_OVERLAP value: $requested. Expected auto, on, or off.",
    )
    requested == "off" && return false
    Threads.nthreads() > 1 || return false
    requested == "on" && return true
    return bem_backend == :metal
end

"""
    build_condensed_coupled_system(fem_mesh, bem_mesh, interface_map, frequency_hz, sound_speed, density; ...)

Assemble and factor the interface-condensed coupled system on the CPU.

Mirrors `BeatEngineCoupled.build_coupled_system` in inputs and in the fields callers read, but
always produces the `:fem_interface_condensed` formulation and never the monolithic one.

`cache` is a `prepare_condensed_coupled_cache` result. `regular_quadrature_order` selects which of its
bundles to assemble with, defaulting to the cache's base order.
"""
function build_condensed_coupled_system(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap,
    frequency_hz::T,
    sound_speed::T,
    density::T;
    quadrature_order::Int=2,
    regular_quadrature_order::Union{Nothing,Int}=nothing,
    singular_order::Int=2,
    cache=nothing,
    validation_diagnostics::Bool=false,
    symmetry_mode::Symbol=:off,
    bulk_loss_factor::T=zero(T),
    bulk_loss_factor_by_vertex=nothing,
    wall_impedances=NamedTuple[],
    transducers::AbstractVector{ElectrodynamicTransducer{T}}=ElectrodynamicTransducer{T}[],
    transducer_operators=nothing,
    prescribed_bem_normal_velocity=nothing,
    schur_block_columns::Int=32,
) where {T<:AbstractFloat}
    # `relative_residual` needs the monolithic coupled matrix, which this formulation never
    # forms. `fem_interior_residual` on each solution is the condensed-appropriate check.
    validation_diagnostics && error(
        "FEM static condensation cannot be combined with full-matrix validation diagnostics.",
    )

    fem_stage_started = time_ns()
    resolved_transducer_operators = isnothing(transducer_operators) ?
                                    assemble_transducer_operators(fem_mesh, bem_mesh, transducers) :
                                    transducer_operators
    transducer_fem_vertices = isempty(transducers) ?
                              Int[] :
                              unique(findnz(resolved_transducer_operators.fem_surface)[1])
    retained_fem_vertices = sort(
        unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)),
    )
    # Without a cache, build one for exactly the order this frequency selected, so the uncached
    # path honours the selection instead of silently falling back to the base order.
    selected_quadrature_order = isnothing(regular_quadrature_order) ? quadrature_order :
                                regular_quadrature_order
    condensed_cache = isnothing(cache) ? prepare_condensed_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=selected_quadrature_order,
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
        retained_fem_vertices=retained_fem_vertices,
        bulk_loss_factor_by_vertex=(
            isnothing(bulk_loss_factor_by_vertex) ?
            fill(bulk_loss_factor, length(fem_mesh.vertices)) :
            bulk_loss_factor_by_vertex
        ),
        wall_impedances=wall_impedances,
    ) : cache
    if !isnothing(cache)
        # Orders are compared as integers. `TriangleRule` defines no `==`, so comparing rules here
        # would fall back to `===` on their vectors and never hold for a freshly built rule. A
        # cache we built above is correct by construction, so only a supplied one is checked.
        condensed_cache.singular_order == singular_order ||
            error("Condensed coupled cache singular order does not match the requested singular order.")
        condensed_cache.base_quadrature_order == quadrature_order ||
            error("Condensed coupled cache base quadrature order does not match the requested quadrature order.")
    end
    bundle = get(condensed_cache.quadrature_bundles, selected_quadrature_order, nothing)
    isnothing(bundle) && error(
        "Condensed coupled cache holds no quadrature bundle for order $selected_quadrature_order; " *
        "it was built for $(sort(collect(keys(condensed_cache.quadrature_bundles)))).",
    )
    # The selected bundle's fields shadow the base cache's, so everything below reads the cache
    # exactly as it did before per-order quadrature existed.
    prepared = merge(condensed_cache.base, bundle)
    prepared.bem_backend in (:cpu, :metal) ||
        error("Condensed coupled cache must be built for the CPU or Metal BEM backend.")
    prepared.symmetry_mode == BeatEngineCore.normalized_symmetry_mode(symmetry_mode) ||
        error("Coupled cache symmetry mode does not match requested symmetry.")
    prepared.retained_fem_vertices == retained_fem_vertices ||
        error("Coupled cache retained FEM vertices do not match the current moving surfaces.")

    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    fem_system = assemble_fem_dynamic_stiffness(
        prepared.stiffness,
        prepared.mass,
        wavenumber;
        bulk_loss_mass=prepared.bulk_loss_mass,
    )
    wall_admittances = Complex{T}[
        miki_rigid_backed_surface_admittance(
            frequency_hz,
            sound_speed,
            density,
            operator.thickness_m,
            operator.flow_resistivity_pa_s_per_m2,
        )
        for operator in prepared.wall_impedance_operators
    ]
    for (operator, admittance) in zip(prepared.wall_impedance_operators, wall_admittances)
        fem_system -= neumann_scale(density, omega) * admittance .* operator.matrix
    end
    interface_operators = prepared.interface_operators
    transducer_count = length(transducers)
    size(resolved_transducer_operators.fem_surface, 2) == transducer_count ||
        error("FEM transducer operator count does not match the transducer list.")
    size(resolved_transducer_operators.bem_surface, 2) == transducer_count ||
        error("BEM transducer operator count does not match the transducer list.")
    normal_derivative_scale = neumann_scale(density, omega)
    bem_motion_flux = normal_derivative_scale .* Complex{T}.(
        resolved_transducer_operators.bem_normal_velocity
    )
    resolved_prescribed_bem_velocity = isnothing(prescribed_bem_normal_velocity) ?
                                       spzeros(T, length(bem_mesh.faces), 0) :
                                       T.(prescribed_bem_normal_velocity)
    size(resolved_prescribed_bem_velocity, 1) == length(bem_mesh.faces) ||
        error("Prescribed BEM normal velocity must contain one row per BEM face.")
    bem_prescribed_neumann = normal_derivative_scale .* Complex{T}.(resolved_prescribed_bem_velocity)
    prescribed_bem_count = size(bem_prescribed_neumann, 2)
    # Precision switches resolve against the cache's backend (unset: on for Metal, off elsewhere).
    bem_backend = prepared.bem_backend
    if T !== Float64 && _fem_float64_enabled(bem_backend)
        fem_system = _fem_system_float64(
            condensed_cache.fem_float64_store, fem_mesh, prepared, frequency_hz, sound_speed, density,
        )
    end
    # Scalar type of the dense coupled system. Under double assembly the Schur block and the condensed
    # right-hand side stay Float64 up to the dense matrix; inputs that are single precision to begin
    # with (BEM operators, traces, transducer parameters) are unchanged.
    dense_type = _dense_double_assembly(bem_backend) ? Float64 : T
    fem_system_s = (time_ns() - fem_stage_started) / 1.0e9

    # Everything the condensation reads is final here and nothing below writes
    # to it, so on Metal it runs on the host while the BEM operators assemble
    # on the GPU. Started before the BEM stage rather than at its own marker
    # below because the overlap is the whole point; `fem_condensation_s` then
    # spans the concurrent region, and `stage_overlap` in the timings says so.
    stage_overlap = _stage_overlap_enabled(prepared.bem_backend)
    condensation_started = time_ns()
    condensation_task = stage_overlap ? Threads.@spawn(_build_condensation(
        fem_system,
        interface_operators,
        retained_fem_vertices;
        schur_block_columns=schur_block_columns,
        schur_float64=dense_type === Float64,
    )) : nothing

    bem_operator_started = time_ns()
    # This solver's own fork of the CPU regular assembly, so it can be optimised without
    # touching the shared path every other backend runs through. Behaviourally identical to
    # `assemble_regular_galerkin_operators(...; backend=:cpu)`, pinned by an equivalence test.
    operators = if prepared.bem_backend == :metal
        # Metal assembles the four operators on the GPU; the condensed algebra
        # below is CPU-only, so bring them down and free the device copies.
        device_operators = assemble_regular_galerkin_operators(
            bem_mesh,
            prepared.p1,
            prepared.dp0,
            wavenumber,
            prepared.rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:metal,
            device_cache=prepared.device_cache,
            singular_cache=prepared.singular_cache,
            device_singular_cache=prepared.device_singular_cache,
            symmetry_mode=prepared.symmetry_mode,
        )
        # Wraps shared device storage in place (copies it when the storage mode
        # is private); either way the host tuple owns the device buffers, so
        # `device_operators` must not be released separately.
        metal_host_operators(device_operators)
    else
        assemble_condensed_regular_operators(
            bem_mesh,
            prepared.p1,
            prepared.dp0,
            wavenumber,
            prepared.rule;
            skip_singular=false,
            singular_order=singular_order,
            singular_cache=prepared.singular_cache,
            cpu_cache=prepared.cpu_assembly_cache,
            symmetry_mode=prepared.symmetry_mode,
        )
    end
    bem_operator_s = (time_ns() - bem_operator_started) / 1.0e9

    bem_matrix_started = time_ns()
    bem_lhs, bem_rhs_operator = burton_miller_neumann_matrices(
        operators,
        prepared.identity_p1_p1,
        prepared.identity_p1_dp0,
        wavenumber,
    )
    # `operators` is dead from here on and the matrices above are freshly
    # allocated host arrays, so free the Metal buffers now rather than leaking
    # one operator set per condensed frequency.
    prepared.bem_backend == :metal && release_operator_storage!(operators)
    bem_interface_block = -(bem_rhs_operator * Complex{T}.(interface_operators.bem_flux))
    bem_motion_block = transducer_count == 0 ? nothing : -(bem_rhs_operator * bem_motion_flux)
    bem_prescribed_rhs = prescribed_bem_count == 0 ?
                         zeros(Complex{T}, length(bem_mesh.vertices), 0) :
                         Complex{T}.(bem_rhs_operator * bem_prescribed_neumann)
    bem_matrix_s = (time_ns() - bem_matrix_started) / 1.0e9

    stage_overlap || (condensation_started = time_ns())
    condensation = if isnothing(condensation_task)
        _build_condensation(
            fem_system,
            interface_operators,
            retained_fem_vertices;
            schur_block_columns=schur_block_columns,
            schur_float64=dense_type === Float64,
        )
    else
        # `fetch` wraps a task failure in a TaskFailedException, which would
        # make the error a caller sees depend on whether the stage happened to
        # be overlapped. Rethrow the original instead.
        try
            fetch(condensation_task)
        catch exception
            exception isa TaskFailedException || rethrow()
            rethrow(exception.task.result)
        end
    end
    fem_condensation_s = (time_ns() - condensation_started) / 1.0e9

    block_assembly_started = time_ns()
    fem_count = length(fem_mesh.vertices)
    bem_count = length(bem_mesh.vertices)
    interface_count = length(interface_map.fem_vertex_indices)
    retained_fem_count = length(retained_fem_vertices)
    gamma_range = 1:retained_fem_count
    bem_range = (retained_fem_count + 1):(retained_fem_count + bem_count)
    flux_range = (retained_fem_count + bem_count + 1):(retained_fem_count + bem_count + interface_count)
    acoustic_system_count = retained_fem_count + bem_count + interface_count
    mechanical_range = transducer_count == 0 ?
                       (1:0) :
                       ((acoustic_system_count + 1):(acoustic_system_count + transducer_count))
    electrical_range = transducer_count == 0 ?
                       (1:0) :
                       (
        (acoustic_system_count + transducer_count + 1):
        (acoustic_system_count + 2 * transducer_count)
    )
    system_count = acoustic_system_count + 2 * transducer_count
    mechanical_impedance = Complex{T}[
        BeatEngineCoupled.mechanical_impedance(transducer, omega, density, sound_speed)
        for transducer in transducers
    ]
    electrical_impedance = Complex{T}[
        BeatEngineCoupled.electrical_impedance(transducer, omega)
        for transducer in transducers
    ]
    force_factor = T[transducer.bl_n_per_a for transducer in transducers]

    coupled = zeros(Complex{dense_type}, system_count, system_count)
    # The Schur complement takes the slot the full FEM block occupies in the monolithic
    # formulation, and the interface coupling is restricted to Γ.
    coupled[gamma_range, gamma_range] = condensation.schur
    coupled[gamma_range, flux_range] =
        -Complex{T}.(Matrix(interface_operators.fem_load[retained_fem_vertices, :]))
    coupled[flux_range, gamma_range] =
        Complex{T}.(Matrix(interface_operators.fem_trace[:, retained_fem_vertices]))
    coupled[bem_range, bem_range] = bem_lhs
    coupled[bem_range, flux_range] = bem_interface_block
    coupled[flux_range, bem_range] = -Complex{T}.(Matrix(interface_operators.bem_trace))
    if transducer_count > 0
        coupled[gamma_range, mechanical_range] =
            -normal_derivative_scale .* Complex{T}.(
                Matrix(resolved_transducer_operators.fem_surface[retained_fem_vertices, :])
            )
        coupled[bem_range, mechanical_range] = bem_motion_block
        coupled[mechanical_range, gamma_range] =
            -Complex{T}.(
                transpose(
                    Matrix(resolved_transducer_operators.fem_force[retained_fem_vertices, :]),
                )
            )
        coupled[mechanical_range, bem_range] =
            Complex{T}.(transpose(Matrix(resolved_transducer_operators.bem_force)))
        coupled[mechanical_range, mechanical_range] = Matrix(Diagonal(mechanical_impedance))
        coupled[mechanical_range, electrical_range] =
            -Matrix(Diagonal(Complex{T}.(force_factor)))
        coupled[electrical_range, mechanical_range] = Matrix(Diagonal(Complex{T}.(force_factor)))
        coupled[electrical_range, electrical_range] = Matrix(Diagonal(electrical_impedance))
    end
    block_assembly_s = (time_ns() - block_assembly_started) / 1.0e9

    coupled_factorization_started = time_ns()
    factorization = dense_type === Float64 && T !== Float64 && _dense_refinement_enabled(bem_backend) ?
                    RefinedDenseLU(coupled) : lu!(coupled)
    coupled_factorization_s = (time_ns() - coupled_factorization_started) / 1.0e9

    return (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        interface_operators=interface_operators,
        transducers=transducers,
        transducer_operators=resolved_transducer_operators,
        density=density,
        bulk_loss_factor=maximum(prepared.bulk_loss_factor_by_vertex; init=zero(T)),
        bulk_loss_factor_by_vertex=prepared.bulk_loss_factor_by_vertex,
        wall_admittances=wall_admittances,
        omega=omega,
        wavenumber=wavenumber,
        field_cache=prepared.field_cache,
        coupled=nothing,
        factorization=factorization,
        dense_scalar_type=dense_type,
        # Element type of the FEM dynamic stiffness the condensation read (BLAB_COUPLED_FEM_FLOAT64).
        fem_scalar_type=real(eltype(fem_system)),
        formulation=:fem_interface_condensed,
        # The order the assembly actually used, so diagnostics report what ran, not what was asked.
        regular_quadrature_order=selected_quadrature_order,
        condensation=condensation,
        fem_range=1:fem_count,
        gamma_range=gamma_range,
        retained_fem_vertices=retained_fem_vertices,
        bem_range=bem_range,
        flux_range=flux_range,
        mechanical_range=mechanical_range,
        electrical_range=electrical_range,
        bem_lhs=nothing,
        bem_factorization=nothing,
        bem_rhs_operator=nothing,
        prescribed_bem_rhs=bem_prescribed_rhs,
        prescribed_bem_neumann=bem_prescribed_neumann,
        bem_backend=prepared.bem_backend,
        linear_backend=:cpu,
        symmetry_mode=prepared.symmetry_mode,
        cache=condensed_cache,
        owns_cache=isnothing(cache),
        validation_diagnostics=false,
        scalar_type=T,
        full_system_order=fem_count + bem_count + interface_count + 2 * transducer_count,
        solved_system_order=system_count,
        timings=(
            fem_system_s=fem_system_s,
            bem_operator_s=bem_operator_s,
            bem_matrix_s=bem_matrix_s,
            fem_condensation_s=fem_condensation_s,
            # True when `fem_condensation_s` and `bem_operator_s` cover the same
            # wall-clock span and must not be added together.
            stage_overlap=stage_overlap,
            block_assembly_s=block_assembly_s,
            coupled_factorization_s=coupled_factorization_s,
            replay_factorization_s=0.0,
        ),
    )
end

function release_condensed_coupled_system!(system)
    _release_condensation!(system.condensation)
    system.owns_cache && release_condensed_coupled_cache!(system.cache)
    return nothing
end

function _solution_from_parts(
    system,
    fem_pressure,
    bem_pressure,
    interface_flux,
    fem_interior_residual;
    diaphragm_velocity=zeros(Complex{system.scalar_type}, length(system.transducers)),
    voice_coil_current=zeros(Complex{system.scalar_type}, length(system.transducers)),
    prescribed_bem_neumann=zeros(Complex{system.scalar_type}, length(system.bem_mesh.faces)),
)
    T = system.scalar_type
    bem_neumann = (
        Complex{T}.(system.interface_operators.bem_flux) * interface_flux +
        neumann_scale(system.density, system.omega) .*
        (Complex{T}.(system.transducer_operators.bem_normal_velocity) * diaphragm_velocity) +
        prescribed_bem_neumann
    )

    pressure_jump = (
        system.interface_operators.fem_trace * fem_pressure -
        system.interface_operators.bem_trace * bem_pressure
    )
    pressure_scale = max(
        norm(system.interface_operators.fem_trace * fem_pressure),
        norm(system.interface_operators.bem_trace * bem_pressure),
        eps(T),
    )
    fem_integrated_flux = zero(Complex{T})
    bem_integrated_flux_along_fem_normal = zero(Complex{T})
    interface_dof = Dict(
        vertex => index
        for (index, vertex) in enumerate(system.interface_map.fem_vertex_indices)
    )
    for local_face_index in eachindex(system.interface_map.fem_face_indices)
        fem_face = system.fem_mesh.boundary_faces[system.interface_map.fem_face_indices[local_face_index]]
        fem_flux_average = sum(interface_flux[interface_dof[vertex]] for vertex in fem_face) / T(3)
        bem_face_index = system.interface_map.bem_face_indices[local_face_index]
        fem_integrated_flux +=
            BeatEngineCoupled._triangle_area(system.fem_mesh.vertices, fem_face) * fem_flux_average
        bem_integrated_flux_along_fem_normal += (
            T(system.interface_map.normal_sign[local_face_index]) *
            system.bem_mesh.areas[bem_face_index] *
            bem_neumann[bem_face_index]
        )
    end
    flux_scale = max(abs(fem_integrated_flux), abs(bem_integrated_flux_along_fem_normal), eps(T))
    return (
        fem_pressure=fem_pressure,
        bem_pressure=bem_pressure,
        interface_flux=interface_flux,
        bem_neumann=bem_neumann,
        diaphragm_velocity=diaphragm_velocity,
        voice_coil_current=voice_coil_current,
        relative_residual=nothing,
        fem_interior_residual=fem_interior_residual,
        fem_rhs_condensation_s=nothing,
        fem_reconstruction_s=nothing,
        pressure_continuity_error=norm(pressure_jump) / pressure_scale,
        flux_conservation_error=abs(fem_integrated_flux - bem_integrated_flux_along_fem_normal) / flux_scale,
        all_bem_replay_error=nothing,
        interface_map=system.interface_map,
        interface_operators=system.interface_operators,
    )
end

function solve_condensed_coupled_excitations(system, excitations)
    T = system.scalar_type
    requested = collect(excitations)
    isempty(requested) && error("At least one coupled excitation is required.")
    excitation_count = length(requested)
    fem_rhs = zeros(Complex{T}, length(system.fem_mesh.vertices), excitation_count)
    bem_rhs = zeros(Complex{T}, length(system.bem_mesh.vertices), excitation_count)
    prescribed_bem_neumann = zeros(Complex{T}, length(system.bem_mesh.faces), excitation_count)
    electrical_rhs = zeros(Complex{T}, length(system.transducers), excitation_count)
    for (column, excitation) in enumerate(requested)
        kind = Symbol(excitation.kind)
        amplitude = Complex{T}(excitation.amplitude)
        if kind == :normal_velocity
            fem_boundary_tags = hasproperty(excitation, :fem_boundary_tags) ?
                                Int.(excitation.fem_boundary_tags) :
                                [Int(excitation.radiator_tag)]
            fem_boundary_weights = hasproperty(excitation, :fem_boundary_weights) ?
                                   T.(excitation.fem_boundary_weights) :
                                   ones(T, length(fem_boundary_tags))
            length(fem_boundary_tags) == length(fem_boundary_weights) ||
                error("Prescribed FEM boundary tags and weights must have the same length.")
            all(weight -> isfinite(weight) && weight > zero(T), fem_boundary_weights) ||
                error("Prescribed FEM boundary weights must be finite and greater than zero.")
            for (tag, weight) in zip(fem_boundary_tags, fem_boundary_weights)
                fem_rhs[:, column] .+= assemble_prescribed_velocity_load(
                    system.fem_mesh,
                    tag,
                    system.density,
                    system.omega,
                    amplitude * weight,
                )
            end
            bem_source_index = hasproperty(excitation, :bem_source_index) ?
                               Int(excitation.bem_source_index) :
                               0
            if bem_source_index > 0
                bem_source_index <= size(system.prescribed_bem_rhs, 2) ||
                    error("Prescribed BEM source index $bem_source_index is unavailable.")
                bem_rhs[:, column] .= amplitude .* view(system.prescribed_bem_rhs, :, bem_source_index)
                prescribed_bem_neumann[:, column] .=
                    amplitude .* view(system.prescribed_bem_neumann, :, bem_source_index)
            end
            isempty(fem_boundary_tags) && bem_source_index == 0 && error(
                "A prescribed-velocity excitation must own at least one FEM or BEM moving boundary.",
            )
        elseif kind == :voltage
            transducer_index = Int(excitation.transducer_index)
            1 <= transducer_index <= length(system.transducers) ||
                error("Voltage excitation references invalid transducer index $transducer_index.")
            electrical_rhs[transducer_index, column] = amplitude
        else
            error("Unsupported coupled excitation kind: $kind.")
        end
    end

    rhs = zeros(Complex{system.dense_scalar_type}, size(system.factorization, 1), excitation_count)
    rhs[system.bem_range, :] = bem_rhs
    rhs[system.electrical_range, :] = electrical_rhs
    reduced_rhs, interior_rhs = _forward_schur(system.condensation, fem_rhs; result_type=system.dense_scalar_type)
    rhs[system.gamma_range, :] = reduced_rhs
    # Results keep the request's precision whatever precision the dense solve used.
    solution = Complex{T}.(system.factorization \ rhs)
    fem_pressure, fem_interior_residual = _backward_schur(
        system.condensation,
        interior_rhs,
        solution[system.gamma_range, :],
    )
    return [
        _solution_from_parts(
            system,
            fem_pressure[:, column],
            solution[system.bem_range, column],
            solution[system.flux_range, column],
            fem_interior_residual[column];
            diaphragm_velocity=solution[system.mechanical_range, column],
            voice_coil_current=solution[system.electrical_range, column],
            prescribed_bem_neumann=prescribed_bem_neumann[:, column],
        )
        for column in axes(fem_pressure, 2)
    ]
end

function solve_condensed_coupled_systems(system, radiator_tags; radiator_velocities=nothing)
    T = system.scalar_type
    tags = Int.(collect(radiator_tags))
    isempty(tags) && error("At least one radiator tag is required.")
    velocities = isnothing(radiator_velocities) ?
                 fill(Complex{T}(1, 0), length(tags)) :
                 Complex{T}.(collect(radiator_velocities))
    length(velocities) == length(tags) ||
        error("Radiator tags and velocities must have the same length.")
    return solve_condensed_coupled_excitations(
        system,
        [
            (kind=:normal_velocity, radiator_tag=tag, transducer_index=0, amplitude=velocity)
            for (tag, velocity) in zip(tags, velocities)
        ],
    )
end

function solve_condensed_coupled_system(system, radiator_tag::Int; radiator_velocity=ComplexF64(1, 0))
    return only(
        solve_condensed_coupled_systems(
            system,
            [radiator_tag];
            radiator_velocities=[radiator_velocity],
        ),
    )
end

end # module
