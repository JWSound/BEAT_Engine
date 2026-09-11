# Refit the Metal sweep-overlap model for this machine.
#
#   julia --threads=4 --project=src/beat_engine/julia_metal \
#       src/beat_engine/julia_local/scripts/calibrate_metal_sweep_overlap.jl [meshes...]
#
# BeatEngineSweepOverlap.jl decides whether a Metal sweep overlaps the next
# frequency's assembly with this one's solve from four constants: the fused
# assembly's cost per squared dof and its fixed part, the time the assembly
# loses running beside the solve, and the fraction the solve slows running
# beside the assembly. This measures them with the solver's own kernels on the
# bundled meshes and prints the export lines.
#
# Contention depends on how many threads compete, so run it with the worker's
# Julia thread count. BLAS gets one thread fewer than the process default,
# as a Metal sweep gives it, overlapped or not. The slowdowns are taken
# from the worse of LU and GMRES, and the worse mesh, so the model errs
# towards staying sequential.
using LinearAlgebra, Printf, Random, Statistics

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const MESHES = isempty(ARGS) ? ["sample.msh", "sample_detailed.msh"] : ARGS
const STEPS = parse(Int, get(ENV, "CALIBRATE_STEPS", "8"))
const FREQUENCY_HZ = 1000.0f0
const SWEEP_BLAS_THREADS = max(1, BLAS.get_num_threads() - 1)

function mesh_setup(name)
    path = isfile(name) ? name : joinpath(@__DIR__, "..", "test_meshes", name)
    mesh = load_gmsh22_with_tags(path, 0.001f0)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(Float32, 4)
    singular_cache = build_singular_correction_cache(mesh, 4)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    caches = (
        device_cache=build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=4),
        singular_cache=singular_cache,
        device_singular_cache=build_metal_singular_correction_cache(singular_cache),
        identity_cache=build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, Float32),
        singular_order=4,
    )
    Random.seed!(20260911)
    q = ComplexF32.(randn(Float32, dp0.global_dof_count, 1), randn(Float32, dp0.global_dof_count, 1))
    k = Float32(2pi) * FREQUENCY_HZ / 343.0f0
    assemble() = assemble_burton_miller_neumann_system_metal(mesh, p1, dp0, q, k, rule; caches...)
    return (name=basename(path), dofs=p1.global_dof_count, assemble=assemble)
end

timed_solve(system, method) = @elapsed solve_metal_burton_miller_system_with_report(system; method=method)

function assembled(setup)
    started = time_ns()
    system = setup.assemble()
    return (system=system, assembly=(time_ns() - started) / 1e9)
end

release(step) = release_metal_burton_miller_system!(step.system)

# Assemble then solve, in turn, as a sequential sweep does.
function sequential_steps(setup, method)
    return map(1:STEPS) do _
        step = assembled(setup)
        solve = timed_solve(step.system, method)
        release(step)
        (assembly=step.assembly, solve=solve)
    end
end

# The sweep pipeline itself, one step ahead. Timed from the first step's
# arrival, so the period is the steady-state cost of a frequency.
function overlapped_steps(setup, method)
    pipeline = start_sweep_assembly_pipeline(_ -> assembled(setup), STEPS + 1, 1, release)
    try
        first = take_sweep_assembly!(pipeline, 1)
        started = time_ns()
        steps = map(1:STEPS) do index
            step = index == 1 ? first : take_sweep_assembly!(pipeline, index)
            solve = timed_solve(step.system, method)
            release(step)
            (assembly=step.assembly, solve=solve)
        end
        return steps, (time_ns() - started) / 1e9 / STEPS
    finally
        shutdown_sweep_assembly_pipeline!(pipeline, release)
    end
end

function measure(setup)
    sequential_steps(setup, :lu)  # compile and warm both paths
    overlapped_steps(setup, :lu)
    rows = []
    assembly_alone = Float64[]
    for method in (:lu, :gmres)
        sequential = sequential_steps(setup, method)
        overlapped, period = overlapped_steps(setup, method)
        assembly = median(step.assembly for step in sequential)
        solve = median(step.solve for step in sequential)
        push!(assembly_alone, assembly)
        push!(rows, (
            method=method,
            solve_alone=solve,
            # The first overlapped assembly ran alone; the rest ran beside a solve.
            cost=max(0.0, median(step.assembly for step in overlapped[2:end]) - assembly),
            slowdown=max(0.0, median(step.solve for step in overlapped) / solve - 1),
            measured_saving=median(step.assembly + step.solve for step in sequential) - period,
        ))
    end
    return (name=setup.name, dofs=setup.dofs, assembly_alone=median(assembly_alone), rows=rows)
end

function calibrate()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    Threads.nthreads() > 1 || error("Start julia with --threads: the overlap needs a second thread.")
    BLAS.set_num_threads(SWEEP_BLAS_THREADS)
    println("device=$(metal.device().name) julia_threads=$(Threads.nthreads()) blas_threads=$(SWEEP_BLAS_THREADS)")
    results = [measure(mesh_setup(name)) for name in MESHES]

    # A = a N^2 + b, least squares over the meshes.
    design = hcat([float(r.dofs)^2 for r in results], ones(length(results)))
    a, b = design \ [r.assembly_alone for r in results]
    cost = maximum(row.cost for r in results for row in r.rows)
    slowdown = maximum(row.slowdown for r in results for row in r.rows)

    for r in results
        @printf("%-22s %6d dofs  assembly %7.1f ms (model %7.1f)\n",
                r.name, r.dofs, 1e3 * r.assembly_alone, 1e3 * (a * float(r.dofs)^2 + b))
        for row in r.rows
            predicted = sweep_overlap_saving_seconds(
                r.assembly_alone, row.solve_alone; overlap_cost_s=cost, host_slowdown=slowdown,
            )
            @printf("    %-5s solve %7.1f ms  assembly loses %5.1f ms  solve slows %5.1f%%  saving %6.1f ms (model %6.1f)\n",
                    row.method, 1e3 * row.solve_alone, 1e3 * row.cost, 100 * row.slowdown,
                    1e3 * row.measured_saving, 1e3 * predicted)
        end
    end
    println()
    @printf("export BLAB_METAL_ASSEMBLY_DOF2_SECONDS=%.3e\n", max(a, 0.0))
    @printf("export BLAB_METAL_ASSEMBLY_FIXED_SECONDS=%.4f\n", max(b, 0.0))
    @printf("export BLAB_METAL_OVERLAP_COST_SECONDS=%.4f\n", cost)
    @printf("export BLAB_METAL_OVERLAP_HOST_SLOWDOWN=%.3f\n", slowdown)
end

calibrate()
