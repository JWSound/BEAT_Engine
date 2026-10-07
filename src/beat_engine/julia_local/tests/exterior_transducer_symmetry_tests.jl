# Full surfaces and exact plane cuts use the same mirrored octahedral triangles.
module ExteriorTransducerSymmetryTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
include(joinpath(@__DIR__, "exterior_transducer_helpers.jl"))

function cut_mesh(mesh,axes)
    indices=findall(f -> all(mesh.vertices[v][a]>=0 for v in f for a in axes),mesh.faces)
    vertices=sort(unique(vcat([collect(mesh.faces[i]) for i in indices]...)))
    map=Dict(v=>i for (i,v) in enumerate(vertices))
    BoundaryMesh(mesh.vertices[vertices],[ntuple(j->map[mesh.faces[i][j]],3) for i in indices],mesh.physical_tags[indices])
end

function symmetry_request(file,mode,completion,orbit)
    r=driver_request(file);r["frequencies_hz"]=[100.,600.]
    r["solver_options"]["symmetry"]=mode
    p=r["compiled_system"]["components"][1]["parameters"]
    p["surface_completion_factor"]=completion;p["physical_driver_orbit_count"]=orbit
    p["symmetry_role"]=completion>1 ? "fractional_driver" : "complete_representative"
    p["fractional_symmetry_axes"]=completion==4 ? ["x","y"] : completion==2 ? ["x"] : []
    r["outputs"][4]["options"]["points_m"]=[[0.,0.,0.15],[0.4,0.3,2.],[-0.4,0.3,2.],[0.4,-0.3,2.],[-0.4,-0.3,2.]]
    r
end

@testset "x/xy fractional sphere: exact cuts, LEM and mirrored fields" begin
    full=sphere(0.1,SVector(0.,0.,0.),2;refinements=2)
    mktemp() do fullpath,io
        mesh_file(io,full)
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            reference=symmetry_request(fullpath,"off",1,1);reference["solver_options"]["phasor_convention"]=convention
            results=captured(reference)
            for (mode,axes,copies) in (("x",(1,),2),("xy",(1,2),4))
                reduced=cut_mesh(full,axes)
                @test validate_exterior_transducer_surface!(reduced,Symbol(mode)) === nothing
                @test length(reduced.faces)*copies == length(full.faces)
                mktemp() do path,io
                    mesh_file(io,reduced)
                    r=symmetry_request(path,mode,copies,1);r["solver_options"]["phasor_convention"]=convention
                    for (a,b) in zip(captured(r),results)
                        actual,expected=quantities(a),quantities(b)
                        for q in ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure")
                            error=maximum(abs.(decoded(actual[q])./decoded(expected[q]).-1))
                            println("SYMMETRY_METRIC ",JSON.json(Dict("convention"=>convention, "mode"=>mode,
                                "frequency_hz"=>a["freq_hz"], "quantity"=>q, "error"=>error)))
                            @test error <= 1e-6
                        end
                        @test maximum(abs.((2.83./decoded(actual["voice_coil_current"]))./(2.83./decoded(expected["voice_coil_current"])).-1)) <= 1e-6
                        meta=actual["radiation_impedance_matrix"]["metadata"]
                        @test meta["row_weights"] == [1.]
                        @test meta["surface_completion_factors"] == [Float64(copies)]
                        @test abs(only(meta["effective_volume_area_m2"])) < 1e-14
                        @test meta["effective_volume_area_zero_or_near_cancelling"] == [true]
                        field=vec(decoded(actual["exterior_pressure"]))
                        @test field[2] ≈ field[3] rtol=1e-12
                        if mode=="xy"
                            @test field[2] ≈ field[4] rtol=1e-12
                            @test field[2] ≈ field[5] rtol=1e-12
                        end
                    end
                    # Every active plane applies, including orbit-only axes.
                    p=r["compiled_system"]["components"][1]["parameters"]
                    p["motion_axis"]=[1.,0.,1.]
                    @test_throws "motion_axis must lie" solve_request(r)
                    if mode=="xy"
                        p["motion_axis"]=[0.,1.,1.]
                        @test_throws "motion_axis must lie" solve_request(r)
                    end
                    p["motion_axis"]=[0.,0.,1.];p["physical_driver_orbit_count"]=2
                    @test_throws "symmetry images" solve_request(r)
                end
            end
        end
    end
end

@testset "orbit-only and mixed completion: per-copy load, published weights and real area" begin
    # Upper hemisphere moving patch gives a nonzero signed effective area.
    a=0.1
    for (mode,centre,completion,orbit,axes) in (
        ("x",SVector(0.3,0.,0.),1,2,()),
        ("xy",SVector(0.3,0.3,0.),1,4,()),
        ("xy",SVector(0.,0.3,0.),2,2,(1,)))
        base=sphere(a,centre,2;refinements=1)
        patch=BoundaryMesh(base.vertices,base.faces,[n[3]>0 ? 2 : 9 for n in base.normals])
        representative=isempty(axes) ? patch : cut_mesh(patch,axes)
        # Reconstruct complete physical copies with distinct mesh tags; force
        # on one physical copy must not get the orbit multiplier.
        copies=[patch]
        mode=="x" && push!(copies,sphere(a,SVector(-centre[1],centre[2],0.),3;refinements=1))
        if mode=="xy"
            centres=completion==2 ? [SVector(0.,-centre[2],0.)] :
                [SVector(-centre[1],centre[2],0.),SVector(centre[1],-centre[2],0.),SVector(-centre[1],-centre[2],0.)]
            for (j,c) in enumerate(centres)
                push!(copies,sphere(a,c,j+2;refinements=1))
            end
        end
        copies=[BoundaryMesh(m.vertices,m.faces,[n[3]>0 ? j+1 : 9 for n in m.normals]) for (j,m) in enumerate(copies)]
        full=join_meshes(copies...)
        mktemp() do fullpath,io
            mesh_file(io,full)
            mktemp() do path,io
                mesh_file(io,representative)
                for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
                    r=symmetry_request(path,mode,completion,orbit);r["frequencies_hz"]=[100.]
                    r["solver_options"]["phasor_convention"]=convention
                    f=driver_request(fullpath;tags=collect(2:orbit+1));f["excitation_port_ids"]=["port:$j" for j in 1:orbit]
                    f["solver_options"]["phasor_convention"]=convention
                    f["outputs"][4]["options"]=r["outputs"][4]["options"]
                    actual=quantities(only(captured(r))); expected=quantities(only(captured(f)))
                    z=decoded(actual["radiation_impedance_matrix"]);zf=decoded(expected["radiation_impedance_matrix"])
                    @test only(z) ≈ sum(zf[1,:]) rtol=1e-6
                    # Simultaneously voltage-drive all copies: sum independent columns.
                    @test only(decoded(actual["diaphragm_velocity"])) ≈ sum(decoded(expected["diaphragm_velocity"])[:,1]) rtol=1e-6
                    @test only(decoded(actual["voice_coil_current"])) ≈ sum(decoded(expected["voice_coil_current"])[:,1]) rtol=1e-6
                    @test vec(decoded(actual["exterior_pressure"])) ≈ vec(sum(decoded(expected["exterior_pressure"]);dims=1)) rtol=1e-6
                    meta=actual["radiation_impedance_matrix"]["metadata"]
                    @test meta["row_weights"] == [Float64(orbit)]
                    @test only(meta["row_weights"])*only(z) ≈ sum(zf) rtol=1e-6
                    @test meta["passivity_min_eig"] >= 0
                    @test only(meta["effective_volume_area_m2"]) ≈ sum(expected["radiation_impedance_matrix"]["metadata"]["effective_volume_area_m2"]) rtol=1e-12
                    @test meta["effective_volume_area_zero_or_near_cancelling"] == [false]
                    # Orbit-only motion must also lie in each active plane.
                    if completion==1
                        r["compiled_system"]["components"][1]["parameters"]["motion_axis"]=[1.,0.,1.]
                        @test_throws "every active symmetry plane" solve_request(r)
                    end
                end
            end
        end
    end
end
end
