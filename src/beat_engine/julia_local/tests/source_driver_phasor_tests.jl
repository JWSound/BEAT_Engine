# Load the actual source driver in an isolated namespace so these tests also
# cover its DSP and Neumann data, not just the core operator conventions.
module SourceDriverPhasorTests
using Test, StaticArrays, LinearAlgebra
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))

@testset "source driver DSP and drives conjugate" begin
    T = Float32
    mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
        [(1,3,2),(1,2,4),(1,4,3),(2,3,4)], ones(Int,4))
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(T,3)
    ipp = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1)
    ipq = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0)
    crossover = Dict("type"=>"lowpass", "filter"=>"butterworth", "order"=>2, "frequency_hz"=>1000.)
    channel = Dict("level_db"=>-3., "polarity"=>1, "delay_ms"=>0.3, "lpf"=>crossover)
    radiator = Dict("tag"=>1, "mesh_id"=>1, "channel"=>"main", "velocity_offset_db"=>-2.,
        "level_db"=>-3., "polarity"=>1, "delay_ms"=>0.3, "lpf"=>crossover)
    results = map((NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)) do convention
        with_phasor_convention(convention) do
            frequency, k = T(500), T(0.7)
            omega, rho = T(2pi)*frequency, T(1.21)
            op = assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule;
                skip_singular=false,singular_order=3,backend=:cpu)
            pressure, q = pressure_for_drives(mesh,ones(Int,4),op,ipp,ipq,[radiator],
                ComplexF32[1],rho,omega,k)
            columns = channel_neumann_columns(mesh,ones(Int,4),[radiator],["main"],rho,omega,T)
            @test vec(columns) ≈ q .* T(10)^(-T(2)/T(20))
            @test q ≈ fill(neumann_scale(rho,omega),4)
            (; pressure,q,columns,filter=butterworth_response("lowpass",2,T(1000),frequency),
                channel=channel_drive(channel,frequency),
                radiator=drive_for_radiator(radiator,Dict(),frequency),
                routed=drive_for_radiator(radiator,Dict("main"=>channel),frequency))
        end
    end
    for key in keys(results[1])
        @test getproperty(results[2],key) ≈ conj.(getproperty(results[1],key)) rtol=1f-5 atol=1f-6
    end
    @test phasor_convention() == NEGATIVE_TIME_PHASOR
end

# Exercise the public source-request path through both Neumann builders. The
# unit checks above cannot catch a backend kernel that keeps the negative-time
# Green function after the driver has selected positive time. Set
# BLAB_TEST_SOURCE_BACKEND=metal under the julia_metal project to run it on
# the GPU.
function captured_source_result(request)
    mktemp() do _, output
        redirect_stdout(output) do
            solve_request(request)
        end
        flush(output)
        seekstart(output)
        events = [JSON.parse(line) for line in eachline(output)]
        @test last(events)["type"] == "completed"
        return only([event["result"] for event in events if event["type"] == "result"])
    end
end

wire_pressure(result, axis) =
    ComplexF32.(result[axis * "_pressure"]["real"][1], result[axis * "_pressure"]["imag"][1])

@testset "source requests conjugate end to end" begin
    config = Dict{String,Any}(
        "mesh_file" => joinpath(@__DIR__, "..", "test_meshes", "sample.msh"),
        "scale_factor" => 0.001, "tag_throat" => 2, "symmetry" => "off",
        "min_angle" => 0.0, "max_angle" => 10.0, "step_size" => 10.0, "distance" => 2.0,
        "quadrature_order" => 2, "singular_order" => 2, "regular_quadrature_mode" => "fixed",
        "rho" => 1.2041, "sound_speed" => 343.0,
    )
    request = Dict{String,Any}(
        "config" => config,
        "frequencies_hz" => [500.0],
        "beat_engine_backend" => get(ENV, "BLAB_TEST_SOURCE_BACKEND", "cpu"),
    )
    previous_fused = get(ENV, "BLAB_BEAT_FUSED_BM", nothing)
    try
        for fused in (true, false)
            ENV["BLAB_BEAT_FUSED_BM"] = fused ? "1" : "0"
            negative, positive = map((NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)) do convention
                request["phasor_convention"] = convention
                captured_source_result(request)
            end
            @test negative["diagnostics"]["phasor_convention"] == NEGATIVE_TIME_PHASOR
            @test positive["diagnostics"]["phasor_convention"] == POSITIVE_TIME_PHASOR
            for axis in ("horizontal", "vertical")
                p_negative = wire_pressure(negative, axis)
                @test maximum(abs.(p_negative)) > 0
                @test wire_pressure(positive, axis) ≈ conj.(p_negative) rtol=5f-4 atol=1f-4
                @test positive[axis * "_spl_db"] ≈ negative[axis * "_spl_db"] atol=1f-3
            end
            # impedance_for_radiators reports Z in one fixed convention, so the
            # wire value must not depend on the request's phasor convention.
            @test positive["impedance"][1] ≈ negative["impedance"][1] rtol=5f-4
            @test abs(negative["impedance"][1][2]) > 0
        end
    finally
        previous_fused === nothing ? delete!(ENV, "BLAB_BEAT_FUSED_BM") :
            (ENV["BLAB_BEAT_FUSED_BM"] = previous_fused)
    end
    @test phasor_convention() == NEGATIVE_TIME_PHASOR
end
end
