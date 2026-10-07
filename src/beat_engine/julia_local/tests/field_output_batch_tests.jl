# Standalone bitwise gate or included by runtests.jl; reuse its loaded bundle.
module FieldOutputBatchTests

using Test, StaticArrays, JSON

if isdefined(Main, :CompiledWorkloadBundle)
    const Driver = Main.CompiledWorkloadBundle
elseif basename(dirname(Base.active_project())) == "julia_metal"
    import BeatEngineCompiledMetalBundle
    const Driver = BeatEngineCompiledMetalBundle
else
    import BeatEngineCompiledCpuBundle
    const Driver = BeatEngineCompiledCpuBundle
end
const Engine = Driver.BeatEngineCore

field_output(id, points; quantity="exterior_pressure", options=Dict{String,Any}()) = Dict(
    "id" => id, "quantity" => quantity, "target_ids" => [],
    "options" => merge(Dict{String,Any}("points_m" => points), options),
)

# Repeated points at different offsets, odd lengths, signed zero, and the real
# polar/sphere sizes exercise slicing and thread/block boundaries.
function field_outputs()
    polar = [[sin(a), -0.0, cos(a)] for a in range(0, pi; length=37)]
    sphere = [[sin(a) * cos(b), sin(a) * sin(b), cos(a)]
              for a in range(0, pi; length=37)
              for b in range(0, 2pi; length=73)[1:72]]
    return JSON.parse(JSON.json(Any[
        field_output("one", [[1.0, -0.0, 0.0]]),
        field_output("polar", polar),
        field_output("interface", [[0.0, 0.0, 2.0]]; quantity="interface_radiated_pressure"),
        Dict("id" => "boundary", "quantity" => "bem_boundary_pressure", "target_ids" => [],
             "options" => Dict{String,Any}()),
        field_output("sphere", sphere),
        field_output("repeat", reverse(polar)),
    ]))
end

# Compare every bit of both complex components (UInt32/UInt64, not == or ≈).
bits(values::AbstractVector{Complex{T}}) where {T} =
    reinterpret(T === Float32 ? UInt32 : UInt64, collect(values))

function check_fields(evaluate, outputs, parsed, batch, T)
    together = evaluate(batch.points)
    @test all(isfinite, together)
    for output in outputs
        String(output["quantity"]) == "exterior_pressure" || continue
        id = String(output["id"])
        separate = evaluate(parsed[id])
        sliced = together[batch.ranges[id]]
        @test bits(sliced) == bits(separate)
        # Check the existing complex encoding, shape, axes and metadata too.
        @test Driver.quantity_wire(output, Driver.rows([sliced], T), "Pa", ["excitation", "observation"]) ==
              Driver.quantity_wire(output, Driver.rows([separate], T), "Pa", ["excitation", "observation"])
    end
end

@testset "exterior field batch points and backend eligibility" begin
    outputs = field_outputs()
    for T in (Float32, Float64)
        parsed = Driver.parse_field_output_points(outputs, T)
        for backend in (:cpu, :cuda, :rocm)
            batch = Driver.exterior_field_output_batch(outputs, parsed, backend)
            @test batch.points isa Vector{SVector{3,T}}
            @test batch.ranges == Dict("one" => 1:1, "polar" => 2:38,
                                       "sphere" => 39:2702, "repeat" => 2703:2739)
            for id in keys(batch.ranges)
                @test isequal(batch.points[batch.ranges[id]], parsed[id])
            end
        end
        @test Driver.exterior_field_output_batch(outputs, parsed, :metal) === nothing
        nonfield = [outputs[3], outputs[4]]
        empty_batch = Driver.exterior_field_output_batch(nonfield, parsed, :cpu)
        @test isempty(empty_batch.points) && isempty(empty_batch.ranges)
        @test isempty(Driver.exterior_field_output_batch(Any[], parsed, :cpu).points)
    end
end

@testset "CPU field separate/concatenated bitwise equality" begin
    mktempdir() do directory
        path = joinpath(directory, "plate.msh")
        write(path, Driver.workload_plate_mesh())
        for T in (Float32, Float64), symmetry in (:off, :xy), order in (2, 4)
            mesh = Engine.load_gmsh22_with_tags(path, one(T))
            cache = Engine.build_field_evaluation_cache(mesh, Engine.triangle_rule(T, order); symmetry_mode=symmetry)
            outputs = field_outputs()
            # A coincident source exercises the scalar skip and SIMD zero-radius mask.
            push!(outputs, field_output("coincident", [collect(cache.source_points[1])]))
            parsed = Driver.parse_field_output_points(outputs, T)
            batch = Driver.exterior_field_output_batch(outputs, parsed, :cpu)
            for drive in 1:2, kernel in (:scalar, :simd), convention in
                (Engine.NEGATIVE_TIME_PHASOR, Engine.POSITIVE_TIME_PHASOR)
                pressure = Complex{T}[Complex{T}(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                neumann = Complex{T}[Complex{T}(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                Engine.with_phasor_convention(convention) do
                    evaluate(points) = Engine.evaluate_galerkin_field_cpu(
                        points, mesh, pressure, neumann, T(2pi * 3000 / 343), cache; kernel)
                    check_fields(evaluate, outputs, parsed, batch, T)
                end
            end
        end
    end
end

@testset "compiled exterior/coupled batch preserves output wire and order" begin
    mktempdir() do directory
        path = joinpath(directory, "plate.msh")
        write(path, Driver.workload_plate_mesh())
        settings = Driver.coupled_workload_environment(; mumps=false)
        withenv(settings..., "OPENBLAS_NUM_THREADS" => "1") do
            old_threads = Driver.BLAS.get_num_threads()
            Driver.BLAS.set_num_threads(1)
            try
                for coupled in (false, true), T in (Float32, Float64)
                    request = coupled ? Driver.coupled_workload_request(; tiny=true) : Driver.workload_request(path, "xy")
                    request["solver_options"]["precision"] = T === Float32 ? "float32" : "float64"
                    request["frequencies_hz"] = [1000.0, 2000.0]
                    second_port = deepcopy(only(request["compiled_system"]["excitation_ports"]))
                    second_port["id"] = "port:second"
                    push!(request["compiled_system"]["excitation_ports"], second_port)
                    push!(request["excitation_port_ids"], second_port["id"])
                    outputs = field_outputs()
                    # Interface decomposition is intentionally unchanged.
                    coupled || filter!(o -> o["quantity"] != "interface_radiated_pressure", outputs)
                    if coupled
                        # Synthesis stays before field evaluation, including per-frequency weights.
                        push!(outputs, field_output("weighted", [[0.0, 0.0, 1.0]];
                            options=Dict("excitation_weights" => [Dict("real" => 0.3, "imag" => -0.7),
                                                                   Dict("real" => -0.2, "imag" => 0.4)])))
                        push!(outputs, field_output("sweep", [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
                            options=Dict("excitation_weights_sweep" => [Any[], [Dict("real" => -0.2, "imag" => 0.4),
                                                                              Dict("real" => 0.5, "imag" => -0.1)]])))
                    end
                    request["outputs"] = outputs
                    @test Driver.BeatEngineContract.validate_system_request(request) === nothing
                    run = Driver.solve_coupled_workload(request)
                    @test !run.outcome.cancelled && run.outcome.solved_count == 2
                    @test length(run.results) == 2
                    for result in run.results
                        @test [q["id"] for q in result["quantities"]] == [o["id"] for o in outputs]
                        @test result["diagnostics"]["timings"]["field_s"] >= 0
                    end
                    for (index, output) in enumerate(outputs)
                        single = deepcopy(request)
                        single["outputs"] = [output]
                        separate = Driver.solve_coupled_workload(single)
                        @test length(separate.results) == 2
                        for frequency in 1:2
                            @test run.results[frequency]["quantities"][index] ==
                                  only(separate.results[frequency]["quantities"])
                        end
                    end
                end
            finally
                Driver.BLAS.set_num_threads(old_threads)
                Driver.reset_compiled_workload_state!()
            end
        end
    end
end

@testset "Metal batching blocked; fixed-chunk kernel bitwise gate" begin
    if get(ENV, "BLAB_RUN_COUPLED_METAL", "0") != "1"
        @test_skip "Set BLAB_RUN_COUPLED_METAL=1 under julia_metal to run Metal field checks."
    else
        # A requested hardware gate must not silently pass using a CPU bundle.
        @test isdefined(Engine, :evaluate_galerkin_field_metal)
        metal = Engine.metal_module()
        @test metal.functional()
        withenv("BLAB_METAL_FIELD_CHUNKS" => "auto") do
            # The real output sizes choose different source partitions. This is
            # why the driver must NOT concatenate Metal outputs in auto mode.
            @test Engine._metal_field_chunk_count(37, 10000) !=
                  Engine._metal_field_chunk_count(3 * 37 + 2664, 10000)
        end
        mktempdir() do directory
            path = joinpath(directory, "plate.msh")
            write(path, Driver.workload_plate_mesh())
            # Metal supports Float32 only. Fixed chunks test kernel independence
            # conditional on a fixed partition, not auto-mode eligibility.
            T = Float32
            mesh = Engine.load_gmsh22_with_tags(path, one(T))
            cpu_cache = Engine.build_field_evaluation_cache(mesh, Engine.triangle_rule(T, 4); symmetry_mode=:xy)
            cache = Engine.build_metal_field_evaluation_cache(cpu_cache)
            try
                outputs = field_outputs()
                parsed = Driver.parse_field_output_points(outputs, T)
                batch = Driver.exterior_field_output_batch(outputs, parsed, :cpu)
                for chunks in ("1", "4"), drive in 1:2
                    pressure = ComplexF32[ComplexF32(sin(0.3i + drive), cos(0.7i - drive)) for i in eachindex(mesh.vertices)]
                    neumann = ComplexF32[ComplexF32(0.1sin(1.1i + drive), 0.2cos(0.9i - drive)) for i in eachindex(mesh.faces)]
                    withenv("BLAB_METAL_FIELD_CHUNKS" => chunks) do
                        @test Driver.exterior_field_output_batch(outputs, parsed, :metal) === nothing
                        evaluate(points) = Engine.evaluate_galerkin_field_metal(
                            points, mesh, pressure, neumann, T(2pi * 3000 / 343), cache)
                        check_fields(evaluate, outputs, parsed, batch, T)
                    end
                end
            finally
                Engine.release_metal_field_evaluation_cache!(cache)
            end
        end
    end
end

end # module
