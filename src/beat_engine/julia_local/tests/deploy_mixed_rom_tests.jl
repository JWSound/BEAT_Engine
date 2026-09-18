# Standalone algebra gate for heterogeneous Deploy ROMs; no application assets.
module DeployMixedROMTests
using Test, LinearAlgebra, StaticArrays
if isdefined(Main, :BeatEngineCore)
    const BeatEngineCore = Main.BeatEngineCore
else
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore
get_value(value, key, default) = get(value, key, default)
include(joinpath(@__DIR__, "..", "deploy_solver.jl"))

function fixture_model(directory, name, rank, nodes, faces, inputs, instances)
    shapes = Dict(
        "k" => (rank, rank), "c" => (rank, nodes), "d" => (faces, rank),
        "b" => (rank, inputs), "e" => (faces, inputs),
        "velocity" => (inputs, rank), "current" => (inputs, rank),
        "velocity_drive" => (inputs, inputs), "current_drive" => (inputs, inputs),
    )
    arrays = Dict{String,Array{ComplexF32,3}}()
    descriptors = Dict{String,Any}()
    for (key, shape) in shapes
        values = reshape(ComplexF32.(1:prod(shape)), 1, shape...) .* ComplexF32(0.03, 0.01)
        if key == "k"
            values[1, :, :] = Matrix{ComplexF32}(I, rank, rank) .* ComplexF32(2, .2)
        end
        arrays[key] = values
        path = joinpath(directory, "$name-$key.bin")
        open(path, "w") do stream
            write(stream, vec(permutedims(values, (3, 2, 1))))
        end
        descriptors[key] = Dict("file" => path, "shape" => collect(size(values)),
            "dtype" => "complex64", "order" => "C", "offset" => 0, "nbytes" => sizeof(values))
    end
    raw = Dict{String,Any}(
        "representation" => "parity_petrov_galerkin_rom", "symmetry_mode" => "off",
        "image_count" => 1, "rank_per_sector" => rank, "sector_signs" => [[1, 1]],
        "node_orbits" => [[i] for i in 0:nodes-1], "face_orbits" => [[i] for i in 0:faces-1],
        "binary_arrays" => descriptors, "instances" => instances,
    )
    return raw, arrays
end

@testset "mixed Deploy ROM matches explicit local linear systems" begin
    mktempdir() do directory
        instance(id, model, node, face, count) = Dict{String,Any}(
            "id" => id, "model_id" => model, "node_offset" => node, "face_offset" => face,
            "input_real" => collect(1.0:count), "input_imag" => fill(.2, count),
        )
        a1 = instance("a1", "a", 0, 0, 1)
        b1 = instance("b1", "b", 2, 3, 2)
        a2 = instance("a2", "a", 5, 5, 1)
        a, aa = fixture_model(directory, "a", 1, 2, 3, 1, [a1, a2])
        b, bb = fixture_model(directory, "b", 2, 3, 2, 2, [b1])
        raw = Dict("models" => Dict("a" => a, "b" => b), "instances" => [a1, b1, a2])
        model = load_deploy_speaker_rom(Dict("rom" => raw), Float64, 7, 8)
        pressure = ComplexF64.(1:7) .* (0.2 + .3im)
        for include_drive in (false, true)
            response = deploy_speaker_rom_response(model, pressure; include_drive=include_drive)
            expected_q = zeros(ComplexF64, 8)
            for (index, (item, arrays)) in enumerate(zip([a1, b1, a2], [aa, bb, aa]))
                matrix(key) = ComplexF64.(arrays[key][1, :, :])
                nodes = size(matrix("c"), 2)
                faces = size(matrix("d"), 1)
                local_pressure = pressure[item["node_offset"] .+ (1:nodes)]
                drive = include_drive ? complex.(item["input_real"], item["input_imag"]) : zeros(length(item["input_real"]))
                state = matrix("k") \ (matrix("b") * drive - matrix("c") * local_pressure)
                expected_q[item["face_offset"] .+ (1:faces)] = matrix("d") * state + matrix("e") * drive
                @test response.velocities[index] ≈ matrix("velocity") * state + matrix("velocity_drive") * drive
                @test response.currents[index] ≈ matrix("current") * state + matrix("current_drive") * drive
            end
            @test response.q ≈ expected_q
            reordered = load_deploy_speaker_rom(Dict("rom" => merge(raw, Dict("instances" => [a2, b1, a1]))), Float64, 7, 8)
            reversed_response = deploy_speaker_rom_response(reordered, pressure; include_drive=include_drive)
            @test reversed_response.q ≈ response.q
            @test reversed_response.velocities ≈ reverse(response.velocities)
        end
        # A collection with one shared model must preserve the legacy calculation.
        legacy = load_deploy_speaker_rom(Dict("rom" => a), Float64, 7, 8)
        one = load_deploy_speaker_rom(Dict("rom" => Dict("models" => Dict("a" => a), "instances" => [a1, a2])), Float64, 7, 8)
        @test deploy_speaker_rom_response(one, pressure; include_drive=true) == deploy_speaker_rom_response(legacy, pressure; include_drive=true)
        invalid = deepcopy(raw)
        invalid["instances"][2]["model_id"] = "a"
        @test_throws ErrorException load_deploy_speaker_rom(Dict("rom" => invalid), Float64, 7, 8)
    end
end
end
