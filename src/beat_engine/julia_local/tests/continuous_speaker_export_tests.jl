module ContinuousSpeakerExportTests
using Test
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

@testset "Continuous Level 3 trains once on its configured band" begin
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    request["frequencies_hz"] = [110.0, 45.0]
    request["solver_options"]["phasor_convention"] = POSITIVE_TIME_PHASOR
    request["solver_options"]["continuous_speaker_rom"] = Dict(
        "band_hz" => [20.0, 250.0], "max_rank" => 64, "max_training" => 8, "validation_count" => 3)
    results = mktemp() do _, io
        redirect_stdout(io) do
            solve_request(request)
        end
        seekstart(io)
        JSON.parse.(readlines(io))
    end
    @test length(results) == 2
    @test !haskey(results[2]["diagnostics"], "continuous_speaker_rom")
    model = JSON.parse(results[1]["diagnostics"]["continuous_speaker_rom"])
    @test model["band"] == [20,250]
    info = model["metadata"]
    @test info["interior_factorization"] == "cpu_fp64"
    @test extrema(info["training"]) == (20.0,250.0)
    @test model["ports"] == request["excitation_port_ids"]
    @test model["matrices"]["K"]["shape"] == model["matrices"]["M"]["shape"]
    for kind in ("checks", "external_source_checks")
        rows = info["validation"][kind]
        @test !isempty(rows)
        @test all(r -> !(r["frequency_hz"] in info["training"]), rows)
        @test all(r -> maximum(r[m] for m in ("field_error","flux_error","velocity_error")) <= 0.01, rows)
    end
end
end
