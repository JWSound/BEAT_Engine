# Whether a Metal sweep overlaps the next frequency's GPU assembly with this
# frequency's host solve, and how far ahead it assembles.
#
# Per frequency a sequential sweep costs A + S + F: the GPU assembly, the host
# dense solve, and the GPU field evaluation. Overlapped, the GPU assembles the
# next frequency while the host solves this one, so a frequency costs
# F + max(A + c, (1 + kappa) S), where c is the time the assembly loses to
# sharing the machine with the solve and kappa the fraction the solve slows
# down while the GPU streams memory. The field waits behind the next assembly
# either way, so F cancels, and the saving per frequency is
#
#     min(S - c, A - kappa S).
#
# It is positive when both sides have work to hide and the slowdowns do not eat
# it. Both move with the hardware -- a larger GPU shrinks A, faster BLAS shrinks
# S -- which is why the choice is not a dof threshold. S comes from the adaptive
# dense-solve model in BeatEngineDenseSolve.jl, already calibrated per machine;
# A, c and kappa from the constants below. All are environment overrides, and
# scripts/calibrate_metal_sweep_overlap.jl measures them.
#
# The defaults are calibrated on an Apple M1 Pro (16-core GPU, 4 Julia threads,
# 7 BLAS threads), with c and kappa rounded up. There, per frequency through
# the pipeline itself:
#
#   mesh              P1 dofs   A       S LU / GMRES    saving LU / GMRES   model
#   sample.msh          1,390   69 ms    22 / 20 ms      27 / 24 ms         19 / 17
#   sample_detailed     3,502  355 ms   269 / 140 ms    244 / 171 ms       266 / 137
#
# The dof threshold it replaces (1,900, fitted on an M1 Max) kept sample.msh
# sequential, which costs 0.12-0.25 s of a 1.3-1.7 s sweep on this machine.

const METAL_ASSEMBLY_DOF2_SECONDS_ENV = "BLAB_METAL_ASSEMBLY_DOF2_SECONDS"
const METAL_ASSEMBLY_FIXED_SECONDS_ENV = "BLAB_METAL_ASSEMBLY_FIXED_SECONDS"
const METAL_OVERLAP_COST_SECONDS_ENV = "BLAB_METAL_OVERLAP_COST_SECONDS"
const METAL_OVERLAP_HOST_SLOWDOWN_ENV = "BLAB_METAL_OVERLAP_HOST_SLOWDOWN"
const METAL_PIPELINE_ENV = "BLAB_METAL_PIPELINE"
const METAL_PIPELINE_DEPTH_ENV = "BLAB_METAL_PIPELINE_DEPTH"

"""Fused Metal assembly seconds per squared P1 dof, per symmetry copy (M1 Pro)."""
const METAL_ASSEMBLY_DOF2_SECONDS_DEFAULT = 2.77e-8

"""Fused Metal assembly seconds that do not scale with the mesh (M1 Pro)."""
const METAL_ASSEMBLY_FIXED_SECONDS_DEFAULT = 0.016

"""Seconds per frequency the assembly loses to running beside the solve (M1 Pro)."""
const METAL_OVERLAP_COST_SECONDS_DEFAULT = 0.003

"""Fraction the host solve slows while the GPU assembles (M1 Pro)."""
const METAL_OVERLAP_HOST_SLOWDOWN_DEFAULT = 0.1

function _overlap_env_float(name::AbstractString, default::Float64)
    text = strip(get(ENV, name, ""))
    isempty(text) && return default
    value = tryparse(Float64, text)
    value === nothing && error("$name must be a number; got $(repr(text)).")
    value >= 0 || error("$name must not be negative; got $value.")
    return value
end

"""
    metal_fused_assembly_seconds(dof_count, copies=1)

Modelled seconds for one fused Metal Burton-Miller assembly: every element
pair once per symmetry copy (`symmetry_reduction_factor`), plus a fixed part.
"""
function metal_fused_assembly_seconds(dof_count::Integer, copies::Integer=1)
    dof_count <= 0 && return 0.0
    per_dof2 = _overlap_env_float(METAL_ASSEMBLY_DOF2_SECONDS_ENV, METAL_ASSEMBLY_DOF2_SECONDS_DEFAULT)
    fixed = _overlap_env_float(METAL_ASSEMBLY_FIXED_SECONDS_ENV, METAL_ASSEMBLY_FIXED_SECONDS_DEFAULT)
    return max(1, copies) * per_dof2 * float(dof_count)^2 + fixed
end

"""
    sweep_overlap_saving_seconds(assembly_s, solve_s; overlap_cost_s, host_slowdown)

Seconds per frequency that overlapping the assembly with the solve saves, from
their sequential times: `min(solve_s - overlap_cost_s, assembly_s - host_slowdown * solve_s)`.
Negative when the overlap loses.
"""
function sweep_overlap_saving_seconds(
    assembly_s::Real,
    solve_s::Real;
    overlap_cost_s::Real=_overlap_env_float(METAL_OVERLAP_COST_SECONDS_ENV, METAL_OVERLAP_COST_SECONDS_DEFAULT),
    host_slowdown::Real=_overlap_env_float(METAL_OVERLAP_HOST_SLOWDOWN_ENV, METAL_OVERLAP_HOST_SLOWDOWN_DEFAULT),
)
    return min(solve_s - overlap_cost_s, assembly_s - host_slowdown * solve_s)
end

"""
    metal_sweep_overlap_plan(dof_count, drive_count, symmetry_mode; frequency_count, threads, setting)

Whether a Metal sweep overlaps assembly with the solve, and why. `reason` is
`:single_thread` or `:single_frequency` when it cannot help, `:override` when
`BLAB_METAL_PIPELINE` decided (`0` off, anything else on), and `:model` when the
modelled saving did. The modelled times are returned either way, for
diagnostics.
"""
function metal_sweep_overlap_plan(
    dof_count::Integer,
    drive_count::Integer,
    symmetry_mode;
    frequency_count::Integer,
    threads::Integer=Threads.nthreads(),
    setting::AbstractString=get(ENV, METAL_PIPELINE_ENV, ""),
)
    assembly_s = metal_fused_assembly_seconds(dof_count, symmetry_reduction_factor(symmetry_mode))
    solve = beat_dense_solve_plan(dof_count, max(1, drive_count))
    solve_s = solve.method === :gmres ? solve.gmres_model_seconds : solve.lu_model_seconds
    saving_s = sweep_overlap_saving_seconds(assembly_s, solve_s)
    setting = strip(setting)
    enabled, reason = if threads <= 1
        false, :single_thread
    elseif frequency_count <= 1
        false, :single_frequency
    elseif !isempty(setting)
        setting != "0", :override
    else
        saving_s > 0, :model
    end
    return (
        enabled=enabled,
        reason=reason,
        assembly_model_s=assembly_s,
        solve_model_s=solve_s,
        saving_model_s=saving_s,
    )
end

"""
    metal_sweep_assembly_lookahead(dof_count, drive_count, frequency_count, FloatType)

How many frequencies the Metal assembly producer may run ahead of the solve.

Derived from memory, never fixed: one in-flight frequency costs a dense
`dofs x dofs` system plus its right-hand sides, which is 12 MB at 1,200 dofs and
3.2 GB at 20,000, so the same constant cannot be right at both ends.
`BLAB_METAL_PIPELINE_DEPTH` overrides it for measurement.
"""
function metal_sweep_assembly_lookahead(
    dof_count::Integer,
    drive_count::Integer,
    frequency_count::Integer,
    ::Type{T},
) where {T<:AbstractFloat}
    entry_bytes = sizeof(Complex{T})
    system_bytes = entry_bytes * (Int(dof_count)^2 + Int(dof_count) * max(1, Int(drive_count)))
    override = strip(get(ENV, METAL_PIPELINE_DEPTH_ENV, ""))
    if !isempty(override)
        requested = tryparse(Int, override)
        requested === nothing &&
            error("$METAL_PIPELINE_DEPTH_ENV must be a positive integer; got $(repr(override)).")
        return clamp(requested, 1, max(1, Int(frequency_count)))
    end
    return sweep_pipeline_depth(system_bytes, metal_sweep_memory_available(), frequency_count)
end
