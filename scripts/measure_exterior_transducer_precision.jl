#!/usr/bin/env julia
# Measurement, not a baseline generator. The public precision rule is deliberately
# bypassed here: assemble the same CPU direct BEM and host Float64 network steps.
module ExteriorTransducerPrecision
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "BeatEngineCompiledDriver.jl"))
const TESTS = joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "tests")
include(joinpath(TESTS, "exterior_test_helpers.jl"))
include(joinpath(TESTS, "exterior_transducer_helpers.jl"))
const FREQUENCIES = Float64[20,35,40,45,48,50,52,54,56,58,60,62,65,70,80,100,600]
const POINTS = [SVector(0.,0.,0.15), SVector(0.,0.,2.), SVector(1.,0.,2.)]

function measure(file,T,steps,convention)
    request=driver_request(file); system=request["compiled_system"]; region=system["regions"][1]
    domain=aggregate_bem_region(system["meshes"],region,system["boundaries"],T)
    mesh=domain.mesh
    data=exterior_motion_basis(system,request["excitation_port_ids"],system["boundaries"],domain,mesh,region,:off)
    p1,dp0=build_p1_space(mesh),build_dp0_space(mesh); rule=triangle_rule(T,4)
    ipp=assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1)
    ipq=assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0)
    singular=build_singular_correction_cache(mesh,4)
    cache=build_beat_cpu_assembly_cache(mesh,p1,dp0,rule;singular_order=4)
    field=build_field_evaluation_cache(mesh,rule)
    rows=Dict{String,Any}[]
    with_phasor_convention(convention) do
        for f in FREQUENCIES
            omega=T(2pi)*T(f); rho=T(1.21); c=T(343.); k=omega/c
            q=hcat(exterior_basis_neumann_values(mesh,data.basis,rho,omega,data.operators)...)
            bem=assemble_burton_miller_neumann_system_cpu(mesh,p1,dp0,q,k,rule;
                identity_p1_p1=ipp,identity_p1_dp0=ipq,singular_order=4,singular_cache=singular,cpu_cache=cache,
                regular_kernel=BeatEngineCore.beat_cpu_regular_kernel(),singular_kernel=BeatEngineCore.beat_cpu_singular_kernel())
            p,report=solve_burton_miller_neumann_system_cpu_with_report(bem;method=:lu,refinement_steps=steps)
            z=transpose(data.force)*p
            for rms in (1.0,0.6)
                system["components"][1]["parameters"]["rms_n_s_per_m"]=rms
                ts,_=electrodynamic_transducers_from_wire(system["components"],system["boundaries"],Dict{String,Int}(),
                    domain.boundary_tag_by_id,nothing,mesh,Float64,:off)
                net=solve_exterior_lumped_network(z,ts,data.basis,system["excitation_ports"],request["excitation_port_ids"],
                    omega,rho,c,2.83)
                u=only(net.velocity); i=only(net.current)
                pressure=exterior_field(SVector{3,T}.(POINTS),mesh,vec(p).*u,vec(q).*u,k,field,:cpu;
                    cpu_kernel=BeatEngineCore.beat_cpu_field_kernel())
                push!(rows,Dict("frequency_hz"=>f,"rms"=>rms,"u"=>[u],"i"=>[i],"Zin"=>[2.83/i],
                    "pressure"=>pressure,"refinement_steps"=>hasproperty(report,:refinement) ? report.refinement.steps : 0))
            end
        end
    end
    rows
end

function main()
    BLAS.set_num_threads(1)
    table=Dict{String,Any}[]; detail=Dict{String,Any}[]
    mktemp() do file,io
        mesh_file(io,sphere(0.1,SVector(0.,0.,0.),2)) # identical slice-2 mesh: 512 faces
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            reference=measure(file,Float64,0,convention)
            for steps in (0,3)
                actual=measure(file,Float32,steps,convention)
                for rms in (1.0,0.6), quantity in ("u","i","Zin","pressure")
                    errors=Float64[]; db=Float64[]; degrees=Float64[]
                    for (a,b) in zip(actual,reference)
                        a["rms"] == rms || continue
                        ratio=a[quantity]./b[quantity]
                        append!(errors,abs.(ratio.-1));append!(db,abs.(20log10.(abs.(ratio))));append!(degrees,abs.(rad2deg.(angle.(ratio))))
                        push!(detail,Dict("convention"=>convention,"refinement_requested"=>steps,"rms"=>rms,
                            "quantity"=>quantity,"frequency_hz"=>a["frequency_hz"],"max_complex_relative"=>maximum(abs.(ratio.-1)),
                            "max_db"=>maximum(abs.(20log10.(abs.(ratio)))),"max_degrees"=>maximum(abs.(rad2deg.(angle.(ratio)))),
                            "refinement_steps"=>a["refinement_steps"]))
                    end
                    push!(table,Dict("convention"=>convention,"refinement_requested"=>steps,"rms"=>rms,
                        "mechanical_q"=>sqrt(0.015/0.0005)/rms,"quantity"=>quantity,
                        "max_complex_relative"=>maximum(errors),"max_db"=>maximum(db),"max_degrees"=>maximum(degrees)))
                end
            end
        end
    end
    report=Dict("faces"=>512,"vertices"=>258,"frequencies_hz"=>FREQUENCIES,"points_m"=>POINTS,
        "summary"=>table,"detail"=>detail,"float32_allowed"=>all(r["max_complex_relative"]<=1e-2 for r in table))
    println(JSON.json(report))
    @testset "precision measurement is finite" begin
        @test length(detail)==544
        @test all(isfinite(r["max_complex_relative"]) for r in table)
        @test all(isfinite(r["max_db"]) for r in table)
        @test all(isfinite(r["max_degrees"]) for r in table)
    end
end
main()
end
