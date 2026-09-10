# Solve-speed regression benchmark for one frequency sweep.
#
# This is step 1 of the performance plan: one repeatable command that produces
# machine-readable timing *and* correctness for the actual workload. It loops the
# production solve path the driver runs per frequency -- operator assembly,
# Burton-Miller system build + dense factorisation, solve, and field evaluation
# -- and writes a single JSON with per-frequency stage times and per-frequency
# correctness anchors (pressure and field norms), plus min/median/max over the
# sweep.
#
# The companion gate is scripts/compare_solve_speed.py: it diffs this JSON
# against a committed baseline and fails on correctness drift (always) or on a
# stage regressing past a threshold (stable hardware only -- shared CI runners
# are too noisy for a tight time gate).
#
#   julia --threads=2 --project=src/beat_engine/julia_local \
#       src/beat_engine/julia_local/scripts/benchmark_sweep.jl \
#       --json results/sweep.json
#
# To benchmark the Metal assembly instead of the CPU one, run the same script
# under the julia_metal project with --backend metal; only the operator assembly
# moves device-side, the solve and field stages are host-resident either way.
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))

using Dates
using LinearAlgebra
using Printf
using Statistics
using JSON
using .BeatEngineCore

const GIT_COMMIT_CACHE = Ref{Any}(:unset)

Base.@kwdef mutable struct SweepConfig
    mesh::String = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    backend::String = "cpu"
    min_freq::Float64 = 100.0
    max_freq::Float64 = 20000.0
    steps::Int = 8
    precision_name::String = "Float32"
    quadrature_order::Int = 2
    singular_order::Int = 2
    symmetry::String = "off"
    eval_points::Int = 0
    scale_factor::Float64 = 0.001
    sound_speed::Float64 = 343.0
    rho::Float64 = 1.21
    tag_throat::Int = 2
    distance::Float64 = 2.0
    repetitions::Int = 1
    warmups::Int = 1
    blas_threads::Int = 1
    threaded_assembly::Bool = true
    output::String = joinpath(@__DIR__, "..", "results", "sweep_sample.json")
    verbose::Bool = false
end

function print_usage()
    println("""
    Usage:
      julia --threads=2 --project=src/beat_engine/julia_local \\
          scripts/benchmark_sweep.jl [options]

    Options:
      --mesh PATH            Mesh path. Default: test_meshes/sample.msh
      --backend cpu|metal    Assembly backend. Default: cpu
      --min-freq HZ          Lowest sweep frequency. Default: 100
      --max-freq HZ          Highest sweep frequency. Default: 20000
      --steps N              Frequencies in the sweep (log-spaced). Default: 8
      --precision Float32    Numeric precision. Default: Float32
      --quadrature-order N   Regular quadrature order. Default: 2
      --singular-order N     Singular quadrature order. Default: 2
      --symmetry off|x|xy    Symmetry mode. Default: off
      --eval-points N        Field evaluation points. Default: 0 (skip field)
      --scale FACTOR         Mesh scale factor. Default: 0.001
      --repetitions N        Measured repetitions. Default: 1
      --warmups N            Warmup repetitions. Default: 1
      --blas-threads N       BLAS thread count. Default: 1
      --serial-assembly      Disable threaded CPU operator assembly.
      --json PATH            JSON output path. Default: results/sweep_sample.json
      --verbose              Print per-frequency timings to the console.
      --help                 Print this message.
    """)
end

function parse_args(args)
    config = SweepConfig()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            print_usage()
            exit()
        elseif arg == "--mesh"
            i += 1; config.mesh = args[i]
        elseif arg == "--backend"
            i += 1; config.backend = lowercase(strip(args[i]))
        elseif arg == "--min-freq"
            i += 1; config.min_freq = parse(Float64, args[i])
        elseif arg == "--max-freq"
            i += 1; config.max_freq = parse(Float64, args[i])
        elseif arg == "--steps"
            i += 1; config.steps = parse(Int, args[i])
        elseif arg == "--precision"
            i += 1; config.precision_name = args[i]
        elseif arg == "--quadrature-order"
            i += 1; config.quadrature_order = parse(Int, args[i])
        elseif arg == "--singular-order"
            i += 1; config.singular_order = parse(Int, args[i])
        elseif arg == "--symmetry"
            i += 1; config.symmetry = lowercase(strip(args[i]))
        elseif arg == "--eval-points"
            i += 1; config.eval_points = parse(Int, args[i])
        elseif arg == "--scale"
            i += 1; config.scale_factor = parse(Float64, args[i])
        elseif arg == "--repetitions"
            i += 1; config.repetitions = parse(Int, args[i])
        elseif arg == "--warmups"
            i += 1; config.warmups = parse(Int, args[i])
        elseif arg == "--blas-threads"
            i += 1; config.blas_threads = parse(Int, args[i])
        elseif arg == "--serial-assembly"
            config.threaded_assembly = false
        elseif arg == "--json"
            i += 1; config.output = args[i]
        elseif arg == "--verbose"
            config.verbose = true
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    config.backend in ("cpu", "metal") || error("--backend must be cpu or metal; got $(config.backend).")
    config.min_freq > 0 || error("--min-freq must be positive.")
    config.max_freq >= config.min_freq || error("--max-freq must be >= --min-freq.")
    config.steps >= 1 || error("--steps must be at least 1.")
    config.repetitions >= 1 || error("--repetitions must be at least 1.")
    0 <= config.warmups <= 3 || error("--warmups must be between 0 and 3.")
    return config
end

precision_type(name::String) = name == "Float32" ? Float32 : name == "Float64" ? Float64 :
    error("Unsupported precision: $name")

function git_commit()
    GIT_COMMIT_CACHE[] !== :unset && return GIT_COMMIT_CACHE[]
    try
        # Walk up from scripts/ to the repository root.
        dir = @__DIR__
        for _ in 1:6
            parent = dirname(dir)
            isempty(parent) || parent == dir ? break : (dir = parent)
            isdir(joinpath(dir, ".git")) || isfile(joinpath(dir, ".git")) || isfile(joinpath(dir, "pyproject.toml")) || continue
            commit = strip(read(`git -C $dir rev-parse --short HEAD`, String))
            GIT_COMMIT_CACHE[] = isempty(commit) ? nothing : commit
            return GIT_COMMIT_CACHE[]
        end
        GIT_COMMIT_CACHE[] = nothing
    catch
        GIT_COMMIT_CACHE[] = nothing
    end
    return GIT_COMMIT_CACHE[]
end

timed_stage!(timings::Dict{String,Float64}, name::String, thunk) = begin
    value = nothing
    elapsed = @elapsed value = thunk()
    timings[name] = elapsed
    value
end

# do-block form: `timed_stage!(timings, name) do ... end` desugars to
# `timed_stage!(thunk, timings, name)`, so accept that order too.
timed_stage!(thunk, timings::Dict{String,Float64}, name::String) = timed_stage!(timings, name, thunk)

function build_sweep_setup(config::SweepConfig, ::Type{T}) where {T<:AbstractFloat}
    timings = Dict{String,Float64}()
    mesh = timed_stage!(timings, "mesh_load") do
        load_gmsh22_with_tags(config.mesh, T(config.scale_factor))
    end
    symmetry_mode = Symbol(config.symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)

    p1_space = nothing
    dp0_space = nothing
    timed_stage!(timings, "space_build") do
        p1_space = build_p1_space(mesh)
        dp0_space = build_dp0_space(mesh)
        nothing
    end

    rule = timed_stage!(timings, "quadrature_rule_build") do
        triangle_rule(T, config.quadrature_order)
    end

    singular_cache = timed_stage!(timings, "singular_correction_cache_build") do
        build_singular_correction_cache(mesh, config.singular_order)
    end

    identity_p1_p1 = timed_stage!(timings, "identity_assembly_p1_p1") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    end
    identity_p1_dp0 = timed_stage!(timings, "identity_assembly_p1_dp0") do
        assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    end

    field_cache = nothing
    if config.eval_points > 0
        field_cache = timed_stage!(timings, "field_cache_build") do
            build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
        end
    else
        timings["field_cache_build"] = 0.0
    end

    # Faces driven by the Neumann throat; resolved once, they do not depend on k.
    throat_faces = findall(t -> t == config.tag_throat, mesh.physical_tags)
    return (
        mesh=mesh,
        p1_space=p1_space,
        dp0_space=dp0_space,
        rule=rule,
        singular_cache=singular_cache,
        identity_p1_p1=identity_p1_p1,
        identity_p1_dp0=identity_p1_dp0,
        field_cache=field_cache,
        throat_faces=throat_faces,
        symmetry_mode=symmetry_mode,
        timings=timings,
    )
end

function run_sweep(config::SweepConfig; measured::Bool=true)
    T = precision_type(config.precision_name)
    BLAS.set_num_threads(config.blas_threads)
    setup = build_sweep_setup(config, T)

    backend = Symbol(config.backend)
    frequencies = Float64.(exp10.(range(log10(config.min_freq), log10(config.max_freq); length=config.steps)))
    wavenumbers = [T(2pi * freq / config.sound_speed) for freq in frequencies]

    eval_points = config.eval_points > 0 ?
        fibonacci_sphere(config.eval_points, T(config.distance)) : nothing

    rows = Dict{String,Any}[]
    for freq in frequencies
        k = T(2pi * freq / config.sound_speed)
        q_neumann = zeros(Complex{T}, length(setup.mesh.faces))
        omega = T(2pi * freq)
        q_neumann[setup.throat_faces] .= Complex{T}(0, T(config.rho) * omega)

        stage = Dict{String,Float64}()
        operators = timed_stage!(stage, "operator_assembly") do
            assemble_regular_galerkin_operators(
                setup.mesh, setup.p1_space, setup.dp0_space, k, setup.rule;
                skip_singular=false,
                singular_order=config.singular_order,
                backend=backend,
                threaded=config.threaded_assembly,
                singular_cache=setup.singular_cache,
                symmetry_mode=setup.symmetry_mode,
            )
        end
        if backend == :metal
            operators = metal_host_operators(operators)
        end

        pressure = nothing
        solve_total = 0.0
        system = timed_stage!(stage, "system_build") do
            build_burton_miller_neumann_cpu_system(
                operators, setup.identity_p1_p1, setup.identity_p1_dp0, k,
            )
        end
        stage["linear_solve"] = @elapsed begin
            pressure = solve_burton_miller_neumann_cpu_system(system, q_neumann, T)
        end
        solve_total = stage["system_build"] + stage["linear_solve"]
        stage["solve_total"] = solve_total

        field_norm = nothing
        if eval_points !== nothing
            stage["field_evaluation"] = @elapsed begin
                pot = evaluate_galerkin_field_cpu(eval_points, setup.mesh, pressure, q_neumann, k, setup.field_cache)
                field_norm = Float64(norm(pot))
            end
        else
            stage["field_evaluation"] = 0.0
        end

        push!(rows, Dict{String,Any}(
            "frequency_hz" => freq,
            "wavenumber" => Float64(k),
            "operator_assembly" => stage["operator_assembly"],
            "system_build" => stage["system_build"],
            "linear_solve" => stage["linear_solve"],
            "solve_total" => stage["solve_total"],
            "field_evaluation" => stage["field_evaluation"],
            "pressure_norm" => pressure === nothing ? nothing : Float64(norm(pressure)),
            "field_norm" => field_norm,
        ))
        config.verbose && println(@sprintf(
            "%8.1f Hz  assemble %.4f s  solve %.4f s  field %.4f s",
            freq, stage["operator_assembly"], stage["solve_total"], stage["field_evaluation"],
        ))
    end

    summary_keys = ["operator_assembly", "system_build", "linear_solve", "solve_total", "field_evaluation"]
    summary = Dict{String,Any}()
    for key in summary_keys
        values = [row[key] for row in rows]
        summary[key] = Dict("min" => minimum(values), "median" => median(values), "max" => maximum(values))
    end
    per_freq_totals = [row["operator_assembly"] + row["solve_total"] + row["field_evaluation"] for row in rows]
    summary["total_sweep"] = Dict(
        "min" => minimum(per_freq_totals),
        "median" => median(per_freq_totals),
        "max" => maximum(per_freq_totals),
        "wall_sum" => sum(per_freq_totals) + sum(values(setup.timings)),
    )

    return Dict{String,Any}(
        "timestamp" => string(now()),
        "git_commit" => git_commit(),
        "julia_version" => string(VERSION),
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS,
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "mesh" => basename(config.mesh),
        "mesh_faces" => length(setup.mesh.faces),
        "mesh_vertices" => length(setup.mesh.vertices),
        "p1_dofs" => setup.p1_space.global_dof_count,
        "dp0_dofs" => setup.dp0_space.global_dof_count,
        "backend" => config.backend,
        "precision" => config.precision_name,
        "symmetry" => config.symmetry,
        "quadrature_order" => config.quadrature_order,
        "singular_order" => config.singular_order,
        "min_freq" => config.min_freq,
        "max_freq" => config.max_freq,
        "steps" => config.steps,
        "eval_points" => config.eval_points,
        "measured" => measured,
        "setup_timings_seconds" => setup.timings,
        "frequencies" => rows,
        "summary_seconds" => summary,
    )
end

function main(args=ARGS)
    config = parse_args(args)

    for i in 1:config.warmups
        println(@sprintf("Warmup %d/%d", i, config.warmups))
        run_sweep(config; measured=false)
        GC.gc()
    end

    runs = Any[]
    for i in 1:config.repetitions
        println(@sprintf("Measured run %d/%d", i, config.repetitions))
        push!(runs, run_sweep(config))
        GC.gc()
    end

    payload = if length(runs) == 1
        runs[1]
    else
        # Median across repetitions of each summary stat, so a stray noisy run
        # does not dominate the recorded number. per-frequency correctness norms
        # are Float32-deterministic and are taken from the last run; per-frequency
        # stage times stay informational only (the gate reads the summary).
        last_run = runs[end]
        merged = Dict{String,Any}()
        for key in ("timestamp", "git_commit", "julia_version", "cpu_name", "cpu_threads", "threads",
                    "blas_threads", "mesh", "mesh_faces", "mesh_vertices", "p1_dofs", "dp0_dofs",
                    "backend", "precision", "symmetry", "quadrature_order", "singular_order",
                    "min_freq", "max_freq", "steps", "eval_points", "measured")
            haskey(last_run, key) && (merged[key] = last_run[key])
        end
        merged["setup_timings_seconds"] = last_run["setup_timings_seconds"]
        merged["frequencies"] = last_run["frequencies"]
        merged["summary_seconds"] = Dict(
            stage => Dict(stat => median(run["summary_seconds"][stage][stat] for run in runs)
                          for stat in ("min", "median", "max"))
            for stage in keys(last_run["summary_seconds"])
        )
        merged
    end

    mkpath(dirname(config.output))
    open(config.output, "w") do io
        println(io, JSON.json(payload))
    end

    sk = payload["summary_seconds"]
    println(@sprintf(
        "Sweep: %s | %s | %d faces | %d steps %.0f-%.0f Hz | P1 %d dofs | eval %d",
        payload["backend"], payload["precision"], payload["mesh_faces"],
        config.steps, config.min_freq, config.max_freq, payload["p1_dofs"], config.eval_points,
    ))
    for key in ["total_sweep", "operator_assembly", "solve_total", "field_evaluation"]
        haskey(sk, key) || continue
        println(@sprintf("  %-20s median %.4f s  (min %.4f  max %.4f)",
            key, sk[key]["median"], sk[key]["min"], sk[key]["max"]))
    end
    println("Wrote $(config.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end