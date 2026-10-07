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
                            # Measured sphere drift is <=2.13e-6; 1e-4 leaves
                            # platform margin without using the broad 1e-2 gate.
                            @test maximum(abs.(decoded(actual[q])./decoded(expected[q]).-1)) <= 1e-4
                        end
                        @test maximum(abs.((2.83./decoded(actual["voice_coil_current"]))./(2.83./decoded(expected["voice_coil_current"])).-1)) <= 1e-4
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

@testset "Float32 BEM network uses unrounded Float64 frequency and fluid values" begin
    mktemp() do path,io
        mesh_file(io,sphere(0.1,SVector(0.,0.,0.),2;refinements=1))
        r=driver_request(path);r["frequencies_hz"]=[58.123456789]
        r["solver_options"]["precision"]="float32"
        system=r["compiled_system"];region=system["regions"][1]
        region["density_kg_per_m3"]=1.210000012345
        region["sound_speed_m_per_s"]=343.123456789
        p=system["components"][1]["parameters"]
        p["lumped_sealed_rear_chamber"]=Dict("enabled"=>true,"volume_m3"=>0.001,"projected_area_m2"=>0.01)
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            r["solver_options"]["phasor_convention"]=convention
            actual=quantities(only(captured(r)))
            omega=2pi*only(r["frequencies_hz"])
            s=convention==POSITIVE_TIME_PHASOR ? im*omega : -im*omega
            stiffness=region["density_kg_per_m3"]*region["sound_speed_m_per_s"]^2*0.01^2/0.001
            zm=p["rms_n_s_per_m"]+s*p["mmd_kg"]+(1/p["cms_m_per_n"]+stiffness)/s
            ze=p["re_ohm"]+s*p["le_h"]
            z=only(decoded(actual["radiation_impedance_matrix"]))
            u=(p["bl_n_per_a"]/ze)*2.83/(zm+z+p["bl_n_per_a"]^2/ze)
            current=(2.83-p["bl_n_per_a"]*u)/ze
            @test only(decoded(actual["diaphragm_velocity"])) ≈ u rtol=1e-12
            @test only(decoded(actual["voice_coil_current"])) ≈ current rtol=1e-12
        end
    end
end
end
