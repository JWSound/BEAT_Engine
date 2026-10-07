# Mandatory hardware gate under julia_metal: unavailable Metal is a failure.
module ExteriorTransducerMetalTests
using Test, StaticArrays, LinearAlgebra, Base64
import Metal
import BeatEngineCompiledMetalBundle
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
include(joinpath(@__DIR__, "exterior_transducer_helpers.jl"))

@testset "Metal exterior transducer CPU-reference budget, images and producer" begin
    @test Metal.functional()
    mesh=sphere(0.1,SVector(0.,0.,0.),2;refinements=2)
    mktemp() do path,io
        mesh_file(io,mesh)
        r=driver_request(path);r["frequencies_hz"]=[54.,100.,600.]
        r["compiled_system"]["components"][1]["parameters"]["rms_n_s_per_m"]=0.6
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            r["solver_options"]["phasor_convention"]=convention
            r["solver_options"]["bem_backend"]="cpu";r["solver_options"]["precision"]="float64"
            reference=captured(r)
            r["solver_options"]["bem_backend"]="metal";r["solver_options"]["precision"]="float32"
            runs=[]
            # Force the existing pipeline solely to exercise its producer; the
            # production on/off selection still comes from its calibration.
            for (assembly,pipeline) in (("direct_system","0"),("direct_system","1"),("operator_matrices","0"))
                r["solver_options"]["burton_miller_assembly"]=assembly
                withenv("BLAB_METAL_PIPELINE"=>pipeline) do
                    results=captured(r);push!(runs,results)
                    for (a,b) in zip(results,reference)
                        actual,expected=quantities(a),quantities(b)
                        for q in ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure")
                            error=maximum(abs.(decoded(actual[q])./decoded(expected[q]).-1))
                            println("METAL_METRIC ",JSON.json(Dict("convention"=>convention, "assembly"=>assembly,
                                "pipeline"=>pipeline, "frequency_hz"=>a["freq_hz"], "quantity"=>q, "error"=>error)))
                            @test error<=1e-2
                        end
                        @test maximum(abs.((2.83./decoded(actual["voice_coil_current"]))./(2.83./decoded(expected["voice_coil_current"])).-1)) <= 1e-2
                        @test a["diagnostics"]["burton_miller_assembly"] == assembly
                    end
                end
            end
            for (a,b) in zip(runs[1],runs[2])
                @test a["quantities"] == b["quantities"]
            end
        end
    end
    # Exercise quarter images with the real compiled bundle, not another direct
    # include. Its CPU host workload and closure inventory are qualified too.
    bundle=BeatEngineCompiledMetalBundle
    mktemp() do path,io
        write(io,bundle.workload_transducer_mesh("xy"));flush(io)
        for mode in ("off","x","xy")
            # off needs the complete octahedron.
            seekstart(io);truncate(io,0);write(io,bundle.workload_transducer_mesh(mode));flush(io)
            r=bundle.transducer_workload_request(path,mode;precision="float32")
            function run(request)
                mktemp() do _,out
                    redirect_stdout(out) do
                        bundle.solve_request(bundle.JSON.parse(bundle.JSON.json(request));event_mode=true)
                    end
                    seekstart(out)
                    [e["result"] for e in JSON.parse.(readlines(out)) if haskey(e,"result")]
                end
            end
            reference=run(r);r["solver_options"]["bem_backend"]="metal"
            withenv("BLAB_METAL_PIPELINE"=>"1") do
                for (a,b) in zip(run(r),reference), (qa,qb) in zip(a["quantities"],b["quantities"])
                    @test maximum(abs.(decoded(qa)./decoded(qb).-1))<=1e-2
                end
            end
        end
    end
end
end
