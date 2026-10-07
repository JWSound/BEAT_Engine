module ExteriorTransducerPrecisionTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
include(joinpath(@__DIR__, "exterior_transducer_helpers.jl"))

@testset "public exterior Float32 and refined LU stay in the driver budget" begin
    mktemp() do path,io
        mesh_file(io,sphere(0.1,SVector(0.,0.,0.),2))
        r=driver_request(path);r["frequencies_hz"]=[50.,58.,100.]
        r["compiled_system"]["components"][1]["parameters"]["re_ohm"]=6.000000000001
        system=r["compiled_system"];region=system["regions"][1]
        domain=aggregate_bem_region(system["meshes"],region,system["boundaries"],Float32)
        data=exterior_motion_basis(system,r["excitation_port_ids"],system["boundaries"],domain,domain.mesh,region,:off)
        @test data.transducers[1].re_ohm == 6.000000000001
        @test typeof(data.transducers[1].re_ohm) === Float64
        @test eltype(data.force) === Float64
        @test exterior_neumann(domain.mesh,data.basis[1],1.21f0,Float32(2pi*54)) ==
            exterior_basis_neumann(domain.mesh,data.basis[1],1.21f0,Float32(2pi*54),data.operators)
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR), rms in (1.,0.6)
            r["solver_options"]["phasor_convention"]=convention
            r["compiled_system"]["components"][1]["parameters"]["rms_n_s_per_m"]=rms
            r["solver_options"]["precision"]="float64"
            reference=captured(r)
            for steps in (0,3)
                r["solver_options"]["precision"]="float32"
                withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>string(steps)) do
                    for (a,b) in zip(captured(r),reference)
                        actual,expected=quantities(a),quantities(b)
                        for q in ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure")
                            @test maximum(abs.(decoded(actual[q])./decoded(expected[q]).-1)) <= 1e-2
                        end
                        @test maximum(abs.((2.83./decoded(actual["voice_coil_current"]))./(2.83./decoded(expected["voice_coil_current"])).-1)) <= 1e-2
                        @test actual["diaphragm_velocity"]["values"]["dtype"] == "complex128"
                        @test actual["voice_coil_current"]["values"]["dtype"] == "complex128"
                        @test a["diagnostics"]["precision"] == "float32"
                        if steps>0
                            @test a["diagnostics"]["dense_solve_refinement_operator"] == "rounded_float32"
                        end
                    end
                end
            end
            # Omitted precision keeps the existing exterior default.
            explicit=captured(r);delete!(r["solver_options"],"precision")
            @test [a["quantities"] for a in captured(r)] == [a["quantities"] for a in explicit]
        end
    end
end
end
