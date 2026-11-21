# Split-precision storage and refinement for the dense condensed coupled system.
#
# Included inside `BeatEngineCoupled` (shared by the condensed solver and the CUDA coupled path). Under `precision=float32` the refined dense path
# (`RefinedDenseLU`) used to hold the whole system twice: once in `ComplexF64` (the refinement
# residual) and once in `ComplexF32` (the LU factor), 24 bytes per entry. Most of the system is the
# BEM block `bem_lhs`, which arrives in `ComplexF32` and so carries no information that a
# `ComplexF64` copy adds. Only the rows and columns built from double-precision FEM quantities
# (Schur couplings, interface and transducer blocks) do.
#
# `SplitDenseMatrix` stores every entry as `ComplexF32` (`hi`) plus the exact `ComplexF64`
# remainder `m - ComplexF64(hi)` on a declared border of rows and columns. Writes accumulate in
# `ComplexF64` exactly as the dense `ComplexF64` matrix did, so `hi` is bit-identical to
# `ComplexF32.(m)` and the reconstructed matrix bit-identical to `m`. `SplitRefinedLU` then
# factors `hi` (the same factor `RefinedDenseLU` computed) and refines against the reconstructed
# matrix, so the two give the same answers up to the summation order of the residual product:
#
#   memory  RefinedDenseLU 24 n²  ->  SplitRefinedLU 16 n² (`hi` kept for the residual)
#                                  ->   8 n² with `BLAB_COUPLED_DENSE_SPILL`, which keeps the
#                                       residual copy of `hi` in an unlinked scratch file and
#                                       factors `hi` in place (the residual streams it from disk).
#
# A refinement that stalls continues as GMRES-IR (right-preconditioned GMRES in `ComplexF64`
# with the `ComplexF32` LU as preconditioner) instead of a `ComplexF64` LU, which would need
# 16 n² more. The `ComplexF64` LU stays as the last resort and says why it ran.

"""
`BLAB_COUPLED_DENSE_SPLIT=auto|1|0`: hold the refined dense coupled system as `ComplexF32` plus
exact `ComplexF64` border corrections (`SplitDenseMatrix`). `auto` (default) uses it wherever the
refined path runs; `0` restores the `ComplexF64` matrix of `RefinedDenseLU`.
"""
_dense_split_enabled() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_SPLIT", "auto"))) in ("auto", "1", "on", "true")

"""
`BLAB_COUPLED_DENSE_SPILL=off|auto|on|<directory>`: keep the refinement copy of the `ComplexF32`
system in a scratch file instead of RAM (8 n² instead of 16 n² resident). `auto` (default)
spills when 16 n² would exceed `BLAB_COUPLED_DENSE_SPILL_FRACTION` (default 0.5) of physical
memory; `on` always spills; a directory spills there (default: `tempdir()`).
"""
function _dense_spill_target(n::Integer)
    setting = strip(get(ENV, "BLAB_COUPLED_DENSE_SPILL", "auto"))
    lowered = lowercase(setting)
    lowered in ("off", "0", "false", "no") && return nothing
    if lowered == "auto"
        fraction = something(tryparse(Float64, get(ENV, "BLAB_COUPLED_DENSE_SPILL_FRACTION", "0.5")), 0.5)
        16.0 * float(n)^2 > fraction * Sys.total_memory() || return nothing
        return tempdir()
    end
    lowered in ("on", "1", "true", "yes") && return tempdir()
    isdir(setting) || error("BLAB_COUPLED_DENSE_SPILL directory does not exist: $setting")
    return setting
end

"""`BLAB_COUPLED_DENSE_GMRES_IR=1|0` (default 1): continue a stalled refinement as GMRES-IR."""
_dense_gmres_ir_enabled() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_GMRES_IR", "1"))) in ("1", "on", "true", "auto")

const SPLIT_REFINEMENT_MAX_ITERATIONS = 10       # as DENSE_REFINEMENT_MAX_ITERATIONS
const DENSE_GMRES_IR_RESTART = 30
const DENSE_GMRES_IR_MAX_CYCLES = 10
const DENSE_SPLIT_PANEL_BYTES = 256 * 1024^2     # ComplexF64 panel used by the residual product

"""
    SpilledMatrix(matrix)

A read-only `ComplexF32` matrix kept in an unlinked scratch file (column-major, streamed by
column panels). The open handle keeps the data alive; closing it (finalizer) frees the space.
No Mmap: the engine bundles do not declare that standard library.
"""
mutable struct SpilledMatrix <: AbstractMatrix{ComplexF32}
    io::IOStream
    rows::Int
    columns::Int
    path::String
    function SpilledMatrix(matrix::Matrix{ComplexF32}, directory::AbstractString)
        path, io = mktemp(directory; cleanup=true)
        write(io, matrix)
        flush(io)
        rm(path; force=true)                 # unlinked: the space returns when `io` is closed
        spilled = new(io, size(matrix, 1), size(matrix, 2), path)
        finalizer(spilled -> close(spilled.io), spilled)
        return spilled
    end
end

Base.size(matrix::SpilledMatrix) = (matrix.rows, matrix.columns)

function Base.getindex(matrix::SpilledMatrix, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    seek(matrix.io, ((j - 1) * matrix.rows + (i - 1)) * sizeof(ComplexF32))
    return read(matrix.io, ComplexF32)
end

"""Copy columns `columns` of `store` into `panel` (a ComplexF32 buffer of matching size)."""
_load_panel!(panel::AbstractMatrix{ComplexF32}, store::AbstractMatrix{ComplexF32}, columns) =
    copyto!(panel, view(store, :, columns))

function _load_panel!(panel::AbstractMatrix{ComplexF32}, store::SpilledMatrix, columns::UnitRange{Int})
    size(panel) == (store.rows, length(columns)) || throw(DimensionMismatch("panel size"))
    seek(store.io, (first(columns) - 1) * store.rows * sizeof(ComplexF32))
    if panel isa Matrix{ComplexF32}
        read!(store.io, panel)
    else
        buffer = Matrix{ComplexF32}(undef, size(panel))
        read!(store.io, buffer)
        copyto!(panel, buffer)
    end
    return panel
end

_load_all(store::AbstractMatrix{ComplexF32}) = store
function _load_all(store::SpilledMatrix)
    matrix = Matrix{ComplexF32}(undef, size(store))
    return _load_panel!(matrix, store, 1:store.columns)
end

"""
    SplitDenseMatrix(n, border_rows, border_columns)

`n × n` matrix whose entry `(i, j)` is `ComplexF64(hi[i, j]) + Δ(i, j)`. `Δ` lives in `dcol`
(every row of a border column), in `drow` (border rows outside the border columns), or, for a
correction found anywhere else, in `stray` (counted; expected empty). Entries whose value does not
fit `Float32` keep `hi = 0` and the whole value in `Δ` and set `overflow`.
"""
mutable struct SplitDenseMatrix <: AbstractMatrix{ComplexF64}
    hi::Matrix{ComplexF32}
    col_slot::Vector{Int32}
    row_slot::Vector{Int32}
    border_columns::Vector{Int}
    border_rows::Vector{Int}
    dcol::Matrix{ComplexF64}
    drow::Matrix{ComplexF64}
    stray::Dict{Tuple{Int,Int},ComplexF64}
    overflow::Bool
end

function SplitDenseMatrix(n::Integer, border_rows, border_columns)
    columns = sort!(unique!(collect(Int, border_columns)))
    rows = sort!(unique!(collect(Int, border_rows)))
    all(c -> 1 <= c <= n, columns) && all(r -> 1 <= r <= n, rows) || throw(ArgumentError("border index out of range"))
    col_slot = zeros(Int32, n)
    row_slot = zeros(Int32, n)
    for (slot, column) in enumerate(columns)
        col_slot[column] = slot
    end
    for (slot, row) in enumerate(rows)
        row_slot[row] = slot
    end
    return SplitDenseMatrix(
        zeros(ComplexF32, n, n), col_slot, row_slot, columns, rows,
        zeros(ComplexF64, n, length(columns)), zeros(ComplexF64, length(rows), n),
        Dict{Tuple{Int,Int},ComplexF64}(), false,
    )
end

"""
    SplitDenseMatrix(hi, border_rows, border_columns)

Split storage around an existing `ComplexF32` square `hi` (adopted, not copied).
"""
function SplitDenseMatrix(hi::Matrix{ComplexF32}, border_rows, border_columns)
    n = size(hi, 1)
    size(hi, 2) == n || throw(ArgumentError("adopted dense storage must be square"))
    matrix = SplitDenseMatrix(0, Int[], Int[])
    columns = sort!(unique!(collect(Int, border_columns)))
    rows = sort!(unique!(collect(Int, border_rows)))
    all(c -> 1 <= c <= n, columns) && all(r -> 1 <= r <= n, rows) || throw(ArgumentError("border index out of range"))
    col_slot = zeros(Int32, n)
    row_slot = zeros(Int32, n)
    for (slot, column) in enumerate(columns)
        col_slot[column] = slot
    end
    for (slot, row) in enumerate(rows)
        row_slot[row] = slot
    end
    return SplitDenseMatrix(
        hi, col_slot, row_slot, columns, rows,
        zeros(ComplexF64, n, length(columns)), zeros(ComplexF64, length(rows), n),
        Dict{Tuple{Int,Int},ComplexF64}(), false,
    )
end

"""
    adopt_square_storage(address, n, order) -> Matrix{ComplexF32}

Turn malloc'd memory holding an n x n `ComplexF32` matrix (leading dimension n) with room for
`order^2` elements into an `order x order` matrix whose top-left n x n block is that matrix and
whose remaining rows and columns are zero. In place: the columns move back to front (memmove),
so no second n x n copy ever exists. The returned array owns the memory (`free` at GC).
"""
function adopt_square_storage(address::Ptr, n::Int, order::Int)
    order >= n || throw(ArgumentError("adopted order $order is smaller than the matrix ($n)"))
    pointer = Ptr{ComplexF32}(address)
    bytes = sizeof(ComplexF32)
    if order > n
        for column in n:-1:2
            ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                  pointer + (column - 1) * order * bytes, pointer + (column - 1) * n * bytes, n * bytes)
        end
    end
    matrix = unsafe_wrap(Array, pointer, (order, order); own=true)
    if order > n
        Threads.@threads for column in 1:order
            first = column <= n ? n + 1 : 1
            @inbounds fill!(view(matrix, first:order, column), zero(ComplexF32))
        end
    end
    return matrix
end

Base.size(matrix::SplitDenseMatrix) = size(matrix.hi)
Base.IndexStyle(::Type{SplitDenseMatrix}) = IndexCartesian()

@inline function _split_correction(matrix::SplitDenseMatrix, i::Int, j::Int)
    @inbounds column = matrix.col_slot[j]
    column > 0 && return @inbounds matrix.dcol[i, column]
    @inbounds row = matrix.row_slot[i]
    row > 0 && return @inbounds matrix.drow[row, j]
    isempty(matrix.stray) && return zero(ComplexF64)
    return get(matrix.stray, (i, j), zero(ComplexF64))
end

@inline function Base.getindex(matrix::SplitDenseMatrix, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    return ComplexF64(@inbounds matrix.hi[i, j]) + _split_correction(matrix, i, j)
end

@inline function _split_round(value::ComplexF64)
    rounded = ComplexF32(value)
    isfinite(rounded) || !isfinite(value) || return zero(ComplexF32), value, true
    # m - ComplexF64(hi) is exact (Sterbenz): hi is the rounding of m.
    return rounded, value - ComplexF64(rounded), false
end

@inline function _split_store!(matrix::SplitDenseMatrix, value::ComplexF64, i::Int, j::Int)
    rounded, correction, overflow = _split_round(value)
    overflow && (matrix.overflow = true)
    @inbounds matrix.hi[i, j] = rounded
    @inbounds column = matrix.col_slot[j]
    if column > 0
        @inbounds matrix.dcol[i, column] = correction
        return value
    end
    @inbounds row = matrix.row_slot[i]
    if row > 0
        @inbounds matrix.drow[row, j] = correction
        return value
    end
    if iszero(correction)
        isempty(matrix.stray) || delete!(matrix.stray, (i, j))
    else
        matrix.stray[(i, j)] = correction
    end
    return value
end

@inline function Base.setindex!(matrix::SplitDenseMatrix, value, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    _split_store!(matrix, ComplexF64(value), i, j)
    return matrix
end

# The BEM block arrives as ComplexF32: copy it straight into `hi`, clearing any corrections.
function Base.setindex!(matrix::SplitDenseMatrix, block::AbstractMatrix{ComplexF32},
                        rows::AbstractUnitRange{<:Integer}, columns::AbstractUnitRange{<:Integer})
    size(block) == (length(rows), length(columns)) || throw(DimensionMismatch("block size does not match the ranges"))
    copyto!(view(matrix.hi, rows, columns), block)
    for (slot, column) in enumerate(matrix.border_columns)
        column in columns && fill!(view(matrix.dcol, rows, slot), zero(ComplexF64))
    end
    for (slot, row) in enumerate(matrix.border_rows)
        row in rows && fill!(view(matrix.drow, slot, columns), zero(ComplexF64))
    end
    isempty(matrix.stray) || filter!(entry -> !(first(entry)[1] in rows && first(entry)[2] in columns), matrix.stray)
    return matrix
end

function Base.setindex!(matrix::SplitDenseMatrix, block::AbstractMatrix,
                        rows::AbstractVector{<:Integer}, columns::AbstractVector{<:Integer})
    size(block) == (length(rows), length(columns)) || throw(DimensionMismatch("block size does not match the indices"))
    @inbounds for (local_column, column) in enumerate(columns), (local_row, row) in enumerate(rows)
        _split_store!(matrix, ComplexF64(block[local_row, local_column]), Int(row), Int(column))
    end
    return matrix
end

"""Materialize the exact `ComplexF64` matrix (the last-resort fallback; 16 n²)."""
function _split_materialize(store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix)
    dense = ComplexF64.(_load_all(store))
    _split_add_corrections!(dense, matrix)
    return dense
end

function _split_add_corrections!(dense::AbstractMatrix{ComplexF64}, matrix::SplitDenseMatrix)
    for (slot, column) in enumerate(matrix.border_columns)
        @views dense[:, column] .+= matrix.dcol[:, slot]
    end
    for (slot, row) in enumerate(matrix.border_rows), column in axes(dense, 2)
        matrix.col_slot[column] == 0 && (dense[row, column] += matrix.drow[slot, column])
    end
    for ((i, j), correction) in matrix.stray
        dense[i, j] += correction
    end
    return dense
end

"""
    _split_mul!(result, store, matrix, x, alpha, beta)

`result = beta*result + alpha*M*x` in `ComplexF64`, where `M = ComplexF64.(store) + Δ(matrix)`.
`store` is the `ComplexF32` copy of `hi` (in RAM or memory-mapped); it is promoted one column panel
at a time, so the product streams `store` once.
"""
function _split_mul!(result::AbstractMatrix{ComplexF64}, store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix,
                     x::AbstractMatrix{ComplexF64}, alpha::ComplexF64, beta::ComplexF64)
    n = size(store, 1)
    width = max(1, min(size(store, 2), DENSE_SPLIT_PANEL_BYTES ÷ (16 * n)))
    panel = Matrix{ComplexF64}(undef, n, width)
    panel32 = Matrix{ComplexF32}(undef, n, width)
    iszero(beta) ? fill!(result, zero(ComplexF64)) : (isone(beta) || (result .*= beta))
    for first_column in 1:width:size(store, 2)
        last_column = min(first_column + width - 1, size(store, 2))
        columns = first_column:last_column
        loaded = length(columns) == width ? panel32 : Matrix{ComplexF32}(undef, n, length(columns))
        _load_panel!(loaded, store, columns)
        block = view(panel, :, 1:length(columns))
        block .= loaded
        mul!(result, block, view(x, columns, :), alpha, one(ComplexF64))
    end
    # Border columns: every row; border rows: outside the border columns; strays.
    isempty(matrix.border_columns) ||
        mul!(result, matrix.dcol, x[matrix.border_columns, :], alpha, one(ComplexF64))
    if !isempty(matrix.border_rows)
        row_part = matrix.drow * x                         # border-column entries of drow are zero
        for (slot, row) in enumerate(matrix.border_rows)
            @views result[row, :] .+= alpha .* row_part[slot, :]
        end
    end
    for ((i, j), correction) in matrix.stray
        @views result[i, :] .+= (alpha * correction) .* x[j, :]
    end
    return result
end

"""Row-sum norm `‖M‖∞` of the reconstructed matrix."""
function _split_opnorm_inf(store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix)
    n = size(store, 1)
    sums = zeros(Float64, n)
    border = matrix.col_slot
    @inbounds for j in axes(store, 2)
        border[j] > 0 && continue
        for i in 1:n
            sums[i] += abs(ComplexF64(store[i, j]))
        end
    end
    # Border columns: |hi + Δ| in every row (drow is zero there).
    @inbounds for (slot, j) in enumerate(matrix.border_columns), i in 1:n
        sums[i] += abs(ComplexF64(store[i, j]) + matrix.dcol[i, slot])
    end
    # Border rows outside the border columns: replace |hi| by |hi + Δ|.
    @inbounds for (slot, i) in enumerate(matrix.border_rows), j in axes(store, 2)
        border[j] > 0 && continue
        correction = matrix.drow[slot, j]
        iszero(correction) && continue
        value = ComplexF64(store[i, j])
        sums[i] += abs(value + correction) - abs(value)
    end
    for ((i, j), correction) in matrix.stray
        value = ComplexF64(store[i, j])
        sums[i] += abs(value + correction) - abs(value)
    end
    return maximum(sums; init=0.0)
end

"""
    SplitRefinedLU(matrix::SplitDenseMatrix)

The refined dense solve of `RefinedDenseLU` over a `SplitDenseMatrix`: `ComplexF32` LU of `hi`,
iterative refinement against the exact matrix, GMRES-IR when refinement stalls, and the
`ComplexF64` LU (materialized) only as the last resort. `spill_path` names the memory-mapped file
holding the residual copy of `hi` when `BLAB_COUPLED_DENSE_SPILL` applies; it is unlinked as soon
as it is written, so its space returns when the factorization is collected.
"""
mutable struct SplitRefinedLU
    matrix::SplitDenseMatrix
    store::AbstractMatrix{ComplexF32}
    matrix_norm::Float64
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    fallback::Union{Nothing,LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{LinearAlgebra.BlasInt}}}
    iterations::Int
    gmres_iterations::Int
    fallback_reason::Union{Nothing,String}
    backward_error::Float64
    spill_path::Union{Nothing,String}
    # A factor held elsewhere (the CUDA LU on the device): `device_solve(x::Matrix{ComplexF32})`
    # returns the ComplexF64 host solution, `release_device()` frees it. `factor` is then nothing.
    device_solve::Union{Nothing,Function}
    release_device::Union{Nothing,Function}
end

function _split_spill(hi::Matrix{ComplexF32}, directory::AbstractString)
    store = SpilledMatrix(hi, directory)
    return store, store.path
end

function SplitRefinedLU(matrix::SplitDenseMatrix)
    all(isfinite, matrix.hi) && all(isfinite, matrix.dcol) && all(isfinite, matrix.drow) &&
        all(isfinite, values(matrix.stray)) ||
        throw(ArgumentError("dense coupled matrix has non-finite entries"))
    n = size(matrix, 1)
    spill_directory = _dense_spill_target(n)
    matrix_norm = _split_opnorm_inf(matrix.hi, matrix)
    store, spill_path = isnothing(spill_directory) ? (matrix.hi, nothing) : _split_spill(matrix.hi, spill_directory)
    refined = SplitRefinedLU(matrix, store, matrix_norm, nothing, nothing, 0, 0, nothing, NaN, spill_path,
                             nothing, nothing)
    if matrix.overflow
        _split_fall_back!(refined, "an entry is outside the Float32 range")
        return refined
    end
    # With a spilled store `hi` is factored in place (8 n² resident); otherwise a copy is factored
    # and `hi` stays as the residual store (16 n²).
    working = isnothing(spill_path) ? copy(matrix.hi) : matrix.hi
    isnothing(spill_path) || (matrix.hi = Matrix{ComplexF32}(undef, 0, 0))
    candidate = dense_lu!(working; check=false)
    if issuccess(candidate) && all(isfinite, candidate.factors)
        refined.factor = candidate
    else
        _split_fall_back!(refined, "the Float32 factorization is singular or not finite")
    end
    return refined
end

"""
    SplitRefinedLU(matrix, device_solve, release_device)

Refinement on the host around a factorization held on a device (CUDA): `matrix.hi` is the
host copy of the matrix that was factored there (spilled to a scratch file under
`BLAB_COUPLED_DENSE_SPILL`, so the host may hold none of it). GMRES-IR needs the factors on the
host and is skipped; a stall falls back to a host `ComplexF64` LU.
"""
function SplitRefinedLU(matrix::SplitDenseMatrix, device_solve::Function, release_device::Function)
    all(isfinite, matrix.hi) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    spill_directory = _dense_spill_target(size(matrix, 1))
    matrix_norm = _split_opnorm_inf(matrix.hi, matrix)
    store, spill_path = isnothing(spill_directory) ? (matrix.hi, nothing) : _split_spill(matrix.hi, spill_directory)
    isnothing(spill_path) || (matrix.hi = Matrix{ComplexF32}(undef, 0, 0))
    return SplitRefinedLU(matrix, store, matrix_norm, nothing, nothing, 0, 0, nothing, NaN,
                          spill_path, device_solve, release_device)
end

"""Free a device-held factor (no-op for host factors)."""
function release_split_factor!(factorization::SplitRefinedLU)
    release = factorization.release_device
    factorization.release_device = nothing
    factorization.device_solve = nothing
    release === nothing || release()
    return nothing
end

Base.size(factorization::SplitRefinedLU, dims...) = size(factorization.store, dims...)

function _split_fall_back!(factorization::SplitRefinedLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason * " (materializes the Float64 system: 16 n² bytes)"
    factorization.factor = nothing
    release_split_factor!(factorization)
    GC.gc()
    factorization.fallback = lu!(_split_materialize(factorization.store, factorization.matrix))
    return factorization
end

function _split_residual!(residual, factorization::SplitRefinedLU, target, solution)
    copyto!(residual, target)
    _split_mul!(residual, factorization.store, factorization.matrix, solution, -one(ComplexF64), one(ComplexF64))
    return residual
end

function _split_backward_error_ratios(factorization::SplitRefinedLU, residual, solution)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(size(factorization.store, 1))
    return [norm(view(residual, :, column), Inf) /
            max(norm(view(solution, :, column), Inf) * threshold, floatmin(Float64)) for column in axes(solution, 2)]
end

function _split_precondition(factorization::SplitRefinedLU, x::AbstractVecOrMat)
    device_solve = factorization.device_solve
    if device_solve !== nothing
        solution = device_solve(ComplexF32.(x isa AbstractVector ? reshape(x, :, 1) : x))
        return x isa AbstractVector ? vec(solution) : solution
    end
    return ComplexF64.(factorization.factor \ ComplexF32.(x))
end

"""
    _split_precondition_double(factorization, x) -> ComplexF64 matrix

`(P L U)⁻¹ x` with the `ComplexF32` factors applied in `ComplexF64` arithmetic (blocked: each
factor panel is promoted once, so one application streams the factors once for all columns).
GMRES-IR needs the preconditioner at working precision to go beyond `κ eps(Float32) < 1`
(Carson & Higham 2018); the classical refinement keeps the `ComplexF32` solve.
"""
function _split_precondition_double(factorization::SplitRefinedLU, x::AbstractMatrix{ComplexF64})
    factor = factorization.factor
    factors = factor.factors
    n = size(factors, 1)
    y = x[factor.p, :]
    width = max(1, min(256, DENSE_SPLIT_PANEL_BYTES ÷ (16 * n)))
    for first_column in 1:width:n                                   # unit lower triangular, forward
        last_column = min(first_column + width - 1, n)
        block = first_column:last_column
        panel = ComplexF64.(view(factors, first_column:n, block))
        b = length(block)
        @views ldiv!(UnitLowerTriangular(panel[1:b, 1:b]), y[block, :])
        last_column < n && @views mul!(y[(last_column + 1):n, :], panel[(b + 1):end, :], y[block, :], -one(ComplexF64), one(ComplexF64))
    end
    for last_column in n:-width:1                                   # upper triangular, backward
        first_column = max(1, last_column - width + 1)
        block = first_column:last_column
        panel = ComplexF64.(view(factors, 1:last_column, block))
        @views ldiv!(UpperTriangular(panel[block, :]), y[block, :])
        first_column > 1 && @views mul!(y[1:(first_column - 1), :], panel[1:(first_column - 1), :], y[block, :], -one(ComplexF64), one(ComplexF64))
    end
    return y
end

function Base.:\(factorization::SplitRefinedLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = rhs isa AbstractVector ? reshape(ComplexF64.(rhs), :, 1) : ComplexF64.(rhs)
    solution = _split_solve_columns(factorization, target)
    return rhs isa AbstractVector ? vec(solution) : solution
end

function _split_solve_columns(factorization::SplitRefinedLU, target::Matrix{ComplexF64})
    residual = similar(target)
    if isnothing(factorization.fallback)
        solution = _split_precondition(factorization, target)
        previous = Inf
        reason = nothing
        for iteration in 0:SPLIT_REFINEMENT_MAX_ITERATIONS
            ratios = _split_backward_error_ratios(factorization, _split_residual!(residual, factorization, target, solution), solution)
            ratio = maximum(ratios; init=0.0)
            if !isfinite(ratio)
                reason = "non-finite residual after $iteration refinement steps"
                break
            end
            if ratio <= 1
                factorization.backward_error = ratio
                return solution
            end
            iteration == SPLIT_REFINEMENT_MAX_ITERATIONS && break
            if ratio > previous / 2
                reason = "refinement stalled at $(ratio)x the Float64 backward error after $iteration steps"
                break
            end
            previous = ratio
            solution .+= _split_precondition(factorization, residual)
            factorization.iterations = max(factorization.iterations, iteration + 1)
        end
        isnothing(reason) &&
            (reason = "refinement did not reach the Float64 backward error in $(SPLIT_REFINEMENT_MAX_ITERATIONS) steps")
        if _dense_gmres_ir_enabled() && factorization.device_solve === nothing
            accepted, gmres_reason = _split_gmres_ir!(factorization, target, solution, residual)
            accepted && return solution
            reason *= "; GMRES-IR: " * gmres_reason
        end
        _split_fall_back!(factorization, reason)
    end
    solution = factorization.fallback \ target
    _split_residual!(residual, factorization, target, solution)
    factorization.backward_error = maximum(_split_backward_error_ratios(factorization, residual, solution); init=0.0)
    return solution
end

"""
    _split_gmres_ir!(factorization, target, solution, residual) -> (accepted, reason)

Restarted, right-preconditioned GMRES on `M y = r` for all columns in lockstep (one streamed
product per Arnoldi step serves every column), `ComplexF64` Arnoldi, the `ComplexF32` LU factors
applied in `ComplexF64` as preconditioner (`_split_precondition_double`). After each cycle the true residual is recomputed and the same backward-error test
as the refinement decides. `solution` is updated in place.
"""
function _split_gmres_ir!(factorization::SplitRefinedLU, target, solution, residual)
    n, columns = size(target)
    m = DENSE_GMRES_IR_RESTART
    for cycle in 1:DENSE_GMRES_IR_MAX_CYCLES
        _split_residual!(residual, factorization, target, solution)
        ratios = _split_backward_error_ratios(factorization, residual, solution)
        # NaN does not compare greater than one: check before an empty active set
        # can be mistaken for convergence after a non-finite Float32 solve.
        all(isfinite, ratios) || return false, "non-finite residual"
        active = findall(>(1), ratios)
        isempty(active) && (factorization.backward_error = maximum(ratios; init=0.0); return true, "")
        k = length(active)
        basis = Array{ComplexF64}(undef, n, m + 1, k)
        hessenberg = zeros(ComplexF64, m + 1, m, k)
        beta = [norm(view(residual, :, column)) for column in active]
        for (local_column, column) in enumerate(active)
            @views basis[:, 1, local_column] .= residual[:, column] ./ beta[local_column]
        end
        preconditioned = Matrix{ComplexF64}(undef, n, k)
        product = Matrix{ComplexF64}(undef, n, k)
        steps = 0
        for j in 1:m
            preconditioned .= _split_precondition_double(factorization, basis[:, j, :])
            _split_mul!(product, factorization.store, factorization.matrix, preconditioned, one(ComplexF64), zero(ComplexF64))
            for local_column in 1:k
                w = view(product, :, local_column)
                for i in 1:j                                    # modified Gram-Schmidt
                    v = view(basis, :, i, local_column)
                    h = dot(v, w)
                    hessenberg[i, j, local_column] = h
                    w .-= h .* v
                end
                h = norm(w)
                hessenberg[j + 1, j, local_column] = h
                @views basis[:, j + 1, local_column] .= iszero(h) ? zero(ComplexF64) : w ./ h
            end
            steps = j
            factorization.gmres_iterations += 1
        end
        for (local_column, column) in enumerate(active)
            rhs_small = zeros(ComplexF64, steps + 1)
            rhs_small[1] = beta[local_column]
            y = view(hessenberg, 1:steps+1, 1:steps, local_column) \ rhs_small    # least squares (QR)
            correction = view(basis, :, 1:steps, local_column) * y
            @views solution[:, column] .+= vec(_split_precondition_double(factorization, reshape(correction, :, 1)))
        end
    end
    _split_residual!(residual, factorization, target, solution)
    ratios = _split_backward_error_ratios(factorization, residual, solution)
    maximum(ratios; init=0.0) <= 1 && (factorization.backward_error = maximum(ratios); return true, "")
    return false, "not converged after $(DENSE_GMRES_IR_MAX_CYCLES) restarts of $(DENSE_GMRES_IR_RESTART) " *
                  "(worst $(maximum(ratios))x the Float64 backward error)"
end

# ---------------------------------------------------------------------------------------------
# Experimental: iterative solve of the condensed dense system without any LU factorization.
#
# `BLAB_COUPLED_DENSE_SOLVER=gmres` (default `lu`). Mixed-precision iterative refinement whose
# inner solver is restarted GMRES in ComplexF32 on the stored `hi` part (one ComplexF32 matrix
# product per Arnoldi step serves every right-hand side), outer residuals in ComplexF64 against the
# exact split matrix. No factor is ever allocated: the dense system needs 8 n^2 bytes instead of
# 16 n^2, and the work per right-hand side is O(n^2) per iteration instead of O(n^3) once.
# If the iteration does not reach the tolerance within its budget, the solve falls back to the
# usual SplitRefinedLU, so results never silently degrade.
#
#   BLAB_COUPLED_DENSE_GMRES_TOL      relative residual target per column (default 1e-10)
#   BLAB_COUPLED_DENSE_GMRES_RESTART  Arnoldi steps per restart (default 100)
#   BLAB_COUPLED_DENSE_GMRES_MAXITER  total inner steps per solve before falling back (default 2000)
#   BLAB_COUPLED_DENSE_GMRES_PRECOND  jacobi (default) | none
# ---------------------------------------------------------------------------------------------

_dense_solver_mode() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_SOLVER", "lu")))
_dense_gmres_setting(name, default) = something(tryparse(typeof(default), get(ENV, name, string(default))), default)

mutable struct SplitGMRES
    matrix::SplitDenseMatrix
    store::Matrix{ComplexF32}
    preconditioner::Vector{ComplexF32}       # right Jacobi scaling (ones for :none)
    tolerance::Float64
    restart::Int
    max_inner::Int
    inner_iterations::Int                    # total Arnoldi steps over all solves
    outer_iterations::Int
    relative_residual::Float64               # worst column of the last solve (ComplexF64 residual)
    solve_seconds::Float64
    matvec_seconds::Float64
    fallback::Union{Nothing,SplitRefinedLU}
    fallback_reason::Union{Nothing,String}
end

function SplitGMRES(matrix::SplitDenseMatrix)
    all(isfinite, matrix.hi) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    n = size(matrix, 1)
    mode = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_GMRES_PRECOND", "jacobi")))
    preconditioner = ones(ComplexF32, n)
    if mode == "jacobi"
        @inbounds for i in 1:n
            d = matrix.hi[i, i]
            preconditioner[i] = iszero(d) ? one(ComplexF32) : inv(d)
        end
    elseif mode != "none"
        error("BLAB_COUPLED_DENSE_GMRES_PRECOND must be jacobi or none; got $(repr(mode)).")
    end
    return SplitGMRES(matrix, matrix.hi, preconditioner,
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_TOL", 1e-10),
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_RESTART", 100),
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_MAXITER", 2000),
                      0, 0, NaN, 0.0, 0.0, nothing, nothing)
end

Base.size(solver::SplitGMRES, dims...) = size(solver.store, dims...)
release_split_factor!(solver::SplitGMRES) = (solver.fallback === nothing || release_split_factor!(solver.fallback); nothing)

function Base.:\(solver::SplitGMRES, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    solver.fallback === nothing || return solver.fallback \ rhs
    started = time_ns()
    target = rhs isa AbstractVector ? reshape(ComplexF64.(rhs), :, 1) : ComplexF64.(rhs)
    solution, ok, reason = _gmres_ir_solve(solver, target)
    solver.solve_seconds += (time_ns() - started) / 1.0e9
    if !ok
        solver.fallback_reason = "GMRES fell back to the Float32 LU with refinement: " * reason
        @warn solver.fallback_reason
        solver.fallback = SplitRefinedLU(solver.matrix)
        solution = solver.fallback \ target
    end
    return rhs isa AbstractVector ? vec(solution) : solution
end

function _gmres_ir_solve(solver::SplitGMRES, target::Matrix{ComplexF64})
    n, columns = size(target)
    solution = zeros(ComplexF64, n, columns)
    residual = similar(target)
    norms = [max(norm(view(target, :, j)), floatmin(Float64)) for j in 1:columns]
    budget = solver.max_inner
    previous = Inf
    for outer in 0:20
        copyto!(residual, target)
        _split_mul!(residual, solver.store, solver.matrix, solution, -one(ComplexF64), one(ComplexF64))
        relative = [norm(view(residual, :, j)) / norms[j] for j in 1:columns]
        worst = maximum(relative; init=0.0)
        solver.relative_residual = worst
        isfinite(worst) || return solution, false, "non-finite residual"
        worst <= solver.tolerance && return solution, true, ""
        budget <= 0 && return solution, false, "no convergence within $(solver.max_inner) inner steps (residual $(worst))"
        outer > 0 && worst > previous / 2 && return solution, false, "outer refinement stalled at $(worst)"
        previous = worst
        active = findall(>(solver.tolerance), relative)
        # inner tolerance: the remaining reduction, but no tighter than Float32 can deliver
        inner_tolerance = max(solver.tolerance / worst, 1e-4)
        correction, steps = try
            _gmres_f32(solver, ComplexF32.(residual[:, active]), inner_tolerance, budget)
        catch err
            err isa SingularException || rethrow()
            return solution, false, "Float32 GMRES correction has a singular projected system: $(sprint(showerror, err))"
        end
        budget -= steps
        solution[:, active] .+= correction
        solver.outer_iterations += 1
    end
    return solution, false, "outer refinement limit reached"
end

"""
Restarted, right-preconditioned GMRES in ComplexF32 for all columns in lockstep, Givens-rotated
least squares per column. Returns the ComplexF64 correction and the number of Arnoldi steps.
"""
function _gmres_f32(solver::SplitGMRES, rhs::Matrix{ComplexF32}, tolerance::Float64, budget::Int)
    n, k = size(rhs)
    m = solver.restart
    P = solver.preconditioner
    x = zeros(ComplexF32, n, k)
    r = copy(rhs)
    bnorm = [max(norm(view(rhs, :, j)), floatmin(Float32)) for j in 1:k]
    V = Array{ComplexF32}(undef, n, m + 1, k)
    H = zeros(ComplexF32, m + 1, m, k)
    cs = zeros(ComplexF32, m, k); sn = zeros(ComplexF32, m, k)
    g = zeros(ComplexF32, m + 1, k)
    Z = Matrix{ComplexF32}(undef, n, k)
    W = Matrix{ComplexF32}(undef, n, k)
    steps = 0
    previous = Inf
    while steps < budget
        beta = [norm(view(r, :, j)) for j in 1:k]
        current = maximum(beta[j] / bnorm[j] for j in 1:k)
        current <= tolerance && break
        # A restart that did not halve the residual means Float32 has reached its floor
        # (about cond * eps(Float32)); the ComplexF64 outer refinement continues from here.
        current > previous / 2 && break
        previous = current
        fill!(H, 0); fill!(g, 0)
        for j in 1:k
            @views V[:, 1, j] .= r[:, j] ./ max(beta[j], floatmin(Float32))
            g[1, j] = beta[j]
        end
        done = falses(k)
        used = 0
        column_steps = zeros(Int, k)
        for i in 1:m
            @views Z .= V[:, i, :] .* P                      # right preconditioning
            t = time_ns()
            mul!(W, solver.store, Z)
            solver.matvec_seconds += (time_ns() - t) / 1.0e9
            steps += 1; used = i
            for j in 1:k
                done[j] && continue
                w = view(W, :, j)
                for l in 1:i                                  # modified Gram-Schmidt
                    v = view(V, :, l, j)
                    h = dot(v, w)
                    H[l, i, j] = h
                    BLAS.axpy!(-h, v, w)
                end
                hn = norm(w)
                H[i + 1, i, j] = hn
                @views V[:, i + 1, j] .= iszero(hn) ? zero(ComplexF32) : w ./ hn
                for l in 1:(i - 1)                            # apply previous rotations
                    a = H[l, i, j]; b = H[l + 1, i, j]
                    H[l, i, j] = conj(cs[l, j]) * a + conj(sn[l, j]) * b
                    H[l + 1, i, j] = -sn[l, j] * a + cs[l, j] * b
                end
                a = H[i, i, j]; b = H[i + 1, i, j]
                denom = sqrt(abs2(a) + abs2(b))
                c, s = iszero(denom) ? (one(ComplexF32), zero(ComplexF32)) : (a / denom, b / denom)
                cs[i, j] = c; sn[i, j] = s
                H[i, i, j] = conj(c) * a + conj(s) * b
                H[i + 1, i, j] = 0
                g[i + 1, j] = -s * g[i, j]
                g[i, j] = conj(c) * g[i, j]
                column_steps[j] = i
                abs(g[i + 1, j]) / bnorm[j] <= tolerance && (done[j] = true)
            end
            (all(done) || steps >= budget) && break
        end
        for j in 1:k                                          # update x_j with its least squares
            steps_j = column_steps[j]
            steps_j == 0 && continue
            y = UpperTriangular(view(H, 1:steps_j, 1:steps_j, j)) \ view(g, 1:steps_j, j)
            @views x[:, j] .+= (V[:, 1:steps_j, j] * y) .* P
        end
        mul!(r, solver.store, x)                             # true ComplexF32 residual for the restart
        r .= rhs .- r
    end
    solver.inner_iterations += steps
    return ComplexF64.(x), steps
end
