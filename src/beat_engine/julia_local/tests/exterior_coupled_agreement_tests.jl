# CPU Float64 qualification; criteria in docs/ExteriorCoupledQualification.md
# preserve the original failure and the post-diagnostic revision decision.
# No numerical baselines are generated or modified.
module ExteriorCoupledAgreementTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))

const FREQUENCIES = [40., 160., 400.]
const VOLTAGE = 2.83
const BUDGETS = Dict("u"=>(0.02,2.), "i"=>(0.02,2.), "Zin"=>(0.02,2.),
                     "pressure"=>(0.03,2.), "load"=>(0.03,2.))
const COUNTS = Dict("requests"=>0, "frequency_results"=>0, "rhs_columns"=>0)
const POINTS = [[0.,0.,0.3], [0.,0.,1.], [0.7,0.,0.7], [-0.4,0.2,0.8]]

# Exact flat box; shared edge/corner vertices, consistently outward triangles.
function box(n)
    vertices = SVector{3,Float64}[]; faces = NTuple{3,Int}[]; tags = Int[]
    lookup = Dict{NTuple{3,Int},Int}()
    function vertex(key)
        get!(lookup, key) do
            push!(vertices, SVector(0.1*key[1]/n, 0.08*key[2]/n, 0.06*key[3]/n))
            length(vertices)
        end
    end
    for axis in 1:3, sign in (-1,1)
        others = filter(!=(axis), [1,2,3])
        for j in 0:n-1, i in 0:n-1
            ids = Int[]
            for (ii,jj) in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))
                key = zeros(Int,3); key[axis]=sign*n
                key[others[1]]=-n+2ii; key[others[2]]=-n+2jj
                push!(ids,vertex(Tuple(key)))
            end
            for f in ((ids[1],ids[2],ids[3]),(ids[1],ids[3],ids[4]))
                a,b,c = vertices[collect(f)]
                dot(cross(b-a,c-a),(a+b+c)/3) < 0 && (f=(f[1],f[3],f[2]))
                push!(faces,f)
                push!(tags, axis == 3 && sign == 1 ? (i < n÷2 ? 2 : 3) : 1)
            end
        end
    end
    BoundaryMesh(vertices,faces,tags)
end

# Globally ordered prism splitting makes adjacent triangles share diagonals.
# The shell has no independently meshed interface, and exactly one radial layer.
function shell(mesh, scale)
    n = length(mesh.vertices)
    vertices = vcat(mesh.vertices, scale .* mesh.vertices)
    tets = NTuple{4,Int}[]
    for face in mesh.faces
        a,b,c = sort(collect(face)); A,B,C = (a+n,b+n,c+n)
        for t in ((a,b,c,C),(a,b,B,C),(a,A,B,C))
            v = vertices[collect(t)]
            d = det(hcat(v[2]-v[1],v[3]-v[1],v[4]-v[1]))
            abs(d) > 1e-16 || error("Degenerate shell tetrahedron")
            push!(tets, d > 0 ? t : (t[2],t[1],t[3],t[4]))
        end
    end
    inner = [(a,c,b) for (a,b,c) in mesh.faces]
    outer = [f .+ n for f in mesh.faces]
    counts = Dict{NTuple{3,Int},Int}()
    for (a,b,c,d) in tets, f in ((a,b,c),(a,b,d),(a,c,d),(b,c,d))
        key = Tuple(sort(collect(f))); counts[key]=get(counts,key,0)+1
    end
    @test all(v in (1,2) for v in values(counts))
    @test Set(k for (k,v) in counts if v == 1) == Set(Tuple(sort(collect(f))) for f in vcat(inner,outer))
    @test validate_exterior_transducer_surface!(mesh) === nothing
    vertices,inner,outer,tets
end

function gmsh(path, vertices, faces, tags)
    open(path,"w") do io
        println(io,"\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n",length(vertices))
        for (i,v) in enumerate(vertices)
            println(io,i," ",join(v," "))
        end
        println(io,"\$EndNodes\n\$Elements\n",length(faces))
        for (i,f) in enumerate(faces)
            println(io,i," 2 2 ",tags[i]," ",tags[i]," ",join(f," "))
        end
        println(io,"\$EndElements")
    end
end

# Public little-endian, row-major mesh transport (also used by native loader gates).
function packed(values, dtype)
    flat=ndims(values)==1 ? vec(values) : vec(permutedims(values))
    Dict("dtype"=>dtype,"shape"=>collect(size(values)),
         "data"=>base64encode(reinterpret(UInt8,htol.(reinterpret(UInt64,flat)))))
end
rows(values)=reduce(vcat,[permutedims(collect(v)) for v in values])
function fem_payload(vertices, inner, outer, tags, tets)
    Dict("schema_version"=>1,"points"=>packed(rows(vertices),"<f8"),
        "physical_names"=>merge(Dict("boundary:$t"=>[t,2] for t in unique(vcat(tags,[4]))),Dict("air"=>[5,3])),
        "cells"=>[Dict("type"=>kind,"connectivity"=>packed(Int64.(rows(cells).-1),"<i8"),
                       "physical_tags"=>packed(Int64.(physical),"<i8"))
            for (kind,cells,physical) in (("triangle",vcat(inner,outer),vcat(tags,fill(4,length(outer)))),
                                         ("tetra",tets,fill(5,length(tets))))])
end

boundary(id,region,mesh,tag,kind) = Dict("id"=>id,"name"=>id,"region_id"=>region,"kind"=>kind,
    "group"=>Dict("mesh_id"=>mesh,"dimension"=>2,"tag"=>tag,"name"=>nothing),"parameters"=>Dict())
function request(path, tags; coupled=false, fempath="", nodes=0, faces=0, ideal=false, control=false)
    r = JSON.parsefile(joinpath(@__DIR__,"..","..","beat_contract","example-exterior-request.json"))
    s=r["compiled_system"]; s["contract_version"]=coupled ? 2 : 3
    s["meshes"][1]["file"]=path; s["meshes"][1]["scale_to_m"]=1.
    s["components"]=Any[]; s["boundaries"]=Any[]; s["excitation_ports"]=Any[]
    if coupled
        push!(s["meshes"],Dict("id"=>"mesh:shell","name"=>"Shell","file"=>fempath,
            "purpose"=>"fem_volume","scale_to_m"=>1.,"translation_m"=>[0.,0.,0.]))
        push!(s["regions"],Dict("id"=>"region:shell","name"=>"Shell","kind"=>"bounded_air",
            "mesh_ids"=>["mesh:shell"],"volume_groups"=>[Dict("mesh_id"=>"mesh:shell","dimension"=>3,"tag"=>5)],
            "sound_speed_m_per_s"=>343.,"density_kg_per_m3"=>1.21,"loss_model"=>Dict()))
        append!(s["boundaries"],[boundary("interface:fem","region:shell","mesh:shell",4,"interface"),
                               boundary("interface:bem","region:exterior","mesh:exterior",4,"interface")])
        s["interfaces"]=[Dict("id"=>"interface:shell","name"=>"Conforming shell",
            "bounded_boundary_id"=>"interface:fem","unbounded_boundary_id"=>"interface:bem",
            "topology"=>Dict("fem_vertex_indices"=>collect(nodes:2nodes-1),
                "fem_to_bem_vertex_indices"=>collect(0:nodes-1),"fem_face_indices"=>collect(faces:2faces-1),
                "bem_face_indices"=>collect(0:faces-1),"normal_sign"=>fill(1,faces),
                "max_coordinate_error"=>0.,"fem_facets_on_tetra_boundary"=>faces,"bem_boundary_edges"=>0))]
    end
    region,mesh = coupled ? ("region:shell","mesh:shell") : ("region:exterior","mesh:exterior")
    push!(s["boundaries"],boundary("rigid",region,mesh,1,"rigid"))
    for (index,tag) in enumerate(tags)
        id,bid,pid="driver:$index","moving:$index","port:$index"
        push!(s["boundaries"],boundary(bid,region,mesh,tag,"moving"))
        push!(s["components"],Dict("id"=>id,"name"=>id,"kind"=>"electrodynamic_transducer",
            "boundary_ids"=>[bid],"parameters"=>Dict("re_ohm"=>6.,"le_h"=>0.0005,"bl_n_per_a"=>7.,
                "mmd_kg"=>0.02,"cms_m_per_n"=>0.001,"rms_n_s_per_m"=>1.5,
                "motion_profile"=>"rigid_translation","motion_axis"=>[0.,0.,1.])))
        push!(s["excitation_ports"],Dict("id"=>pid,"name"=>pid,"component_id"=>id,"kind"=>"voltage"))
    end
    r["excitation_port_ids"]=["port:$i" for i in eachindex(tags)]
    r["frequencies_hz"]=FREQUENCIES
    r["solver_options"]=Dict("precision"=>"float64","bem_backend"=>"cpu","symmetry"=>"off",
        "quadrature_order"=>4,"singular_order"=>4,"regular_quadrature_mode"=>"fixed",
        "transducer_reference_voltage_v"=>VOLTAGE,"static_condensation"=>false)
    r["outputs"]=[Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=>
        q == "exterior_pressure" ? Dict("points_m"=>POINTS) : Dict())
        for q in ("diaphragm_velocity","voice_coil_current","exterior_pressure")]
    coupled || push!(r["outputs"],Dict("id"=>"load","quantity"=>"radiation_impedance_matrix","target_ids"=>[],"options"=>Dict()))
    if ideal
        for component in s["components"]
            component["kind"]="ideal_velocity_source"
            # Inner FEM normals are -z. Negating this supported normal-source
            # solution gives physical +z unit motion on the flat box patch.
            component["parameters"]=coupled ? Dict{String,Any}() :
                Dict("motion_profile"=>"rigid_translation","motion_axis"=>[0.,0.,1.])
        end
        for port in s["excitation_ports"]; port["kind"]="normal_velocity"; end
        r["outputs"]=[Dict("id"=>"exterior_pressure","quantity"=>"exterior_pressure",
            "target_ids"=>[],"options"=>Dict("points_m"=>POINTS))]
        if !coupled
            append!(r["outputs"],[Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=>Dict())
                for q in ("radiation_impedance","radiation_impedance_matrix")])
        end
    end
    if control || ideal
        push!(r["outputs"],Dict("id"=>"boundary_pressure",
            "quantity"=>coupled ? "fem_nodal_pressure" : "bem_boundary_pressure",
            "target_ids"=>[],"options"=>Dict()))
    end
    r
end

function captured(r)
    COUNTS["requests"]+=1
    COUNTS["frequency_results"]+=length(r["frequencies_hz"])
    COUNTS["rhs_columns"]+=length(r["frequencies_hz"])*length(r["excitation_port_ids"])
    mktemp() do _,io
        redirect_stdout(io) do
            solve_request(deepcopy(r);event_mode=true)
        end
        seekstart(io)
        [e["result"] for e in JSON.parse.(readlines(io)) if haskey(e,"result")]
    end
end
function decoded(item)
    values=collect(reinterpret(ComplexF64,base64decode(item["values"]["content_base64"])))
    shape=Tuple(Int.(item["values"]["shape"]))
    length(shape)==1 ? values : permutedims(reshape(values,reverse(shape)))
end
function observables(result, convention; coupled=false)
    q=Dict(item["id"]=>item for item in result["quantities"])
    U=transpose(decoded(q["diaphragm_velocity"])); I=transpose(decoded(q["voice_coil_current"]))
    w=2pi*result["freq_hz"]; zm=1.5+im*(w*0.02-1/(w*0.001)); ze=6+im*w*0.0005
    convention==NEGATIVE_TIME_PHASOR && ((zm,ze)=conj.((zm,ze)))
    # Electrical identity applies also to shorted neighbours (V=0).
    @test 7U+ze*I ≈ VOLTAGE*Matrix{ComplexF64}(LinearAlgebra.I,size(U)...) rtol=1e-11 atol=1e-12
    size(U,1)==2 && (@test abs(U[2,1]) > 1e-10)
    load=coupled ? (7I-zm*U)/U : decoded(q["load"])
    Dict("u"=>vec(U),"i"=>vec(I),"Zin"=>VOLTAGE ./ diag(I),
         "pressure"=>vec(decoded(q["exterior_pressure"])),"load"=>diag(load))
end

function compare(exterior, coupled, convention; label)
    ratios=Dict(q=>ComplexF64[] for q in keys(BUDGETS))
    @test length(exterior)==length(coupled)==length(FREQUENCIES)
    for (e,c) in zip(exterior,coupled)
        @test e["freq_hz"]==c["freq_hz"]
        eo,co=observables(e,convention),observables(c,convention;coupled=true)
        for q in keys(ratios)
            append!(ratios[q],co[q]./eo[q])
        end
        pairs(v)=[[real(x),imag(x)] for x in v]
        println("AGREEMENT_SAMPLE ",JSON.json(merge(label,Dict("frequency_hz"=>e["freq_hz"],
            "phasor"=>convention,"quantities"=>Dict(q=>Dict("exterior"=>pairs(eo[q]),
                "coupled"=>pairs(co[q])) for q in keys(ratios))))))
    end
    Dict(q=>Dict("magnitude"=>maximum(abs.(abs.(v).-1)),
        "phase_deg"=>maximum(abs.(rad2deg.(angle.(v)))),"complex"=>maximum(abs.(v.-1))) for (q,v) in ratios)
end

# Pressure at inner physical vertices; never integrate the expanded interface.
function integrated_force(mesh, pressure, tags)
    force=zeros(ComplexF64,length(tags),size(pressure,1))
    for (i,tag) in enumerate(tags), f in eachindex(mesh.faces)
        mesh.physical_tags[f]==tag || continue
        @test mesh.normals[f][3] ≈ 1.0
        force[i,:] .+= mesh.areas[f]*vec(sum(pressure[:,collect(mesh.faces[f])];dims=2))/3
    end
    force
end

relative(a,b)=norm(a-b)/norm(b)
complex_pairs(a)=[[real(z),imag(z)] for z in vec(a)]
function control_metrics(actual, ideal, mesh, tags, convention; coupled, label)
    @test length(actual)==length(ideal)==length(FREQUENCIES)
    load_errors=Float64[]; field_errors=Float64[]; direct_errors=Float64[]
    ideal_loads=ComplexF64[]; ideal_fields=ComplexF64[]
    diagnostics=Any[]
    for (a,b) in zip(actual,ideal)
        @test a["freq_hz"]==b["freq_hz"]
        aq=Dict(q["id"]=>decoded(q) for q in a["quantities"])
        bq=Dict(q["id"]=>decoded(q) for q in b["quantities"])
        U=Matrix(transpose(aq["diaphragm_velocity"]))
        I=Matrix(transpose(aq["voice_coil_current"]))
        direct=integrated_force(mesh,aq["boundary_pressure"],tags)/U
        sign=coupled ? -1 : 1
        ideal_load=sign*integrated_force(mesh,bq["boundary_pressure"],tags)
        field=Matrix(transpose(aq["exterior_pressure"]))/U
        ideal_field=sign*Matrix(transpose(bq["exterior_pressure"]))
        w=2pi*a["freq_hz"]; zm=1.5+im*(w*0.02-1/(w*0.001))
        convention==NEGATIVE_TIME_PHASOR && (zm=conj(zm))
        implied=(7I-zm*U)/U
        load=coupled ? implied : aq["load"]
        push!(load_errors,relative(load,ideal_load))
        push!(field_errors,relative(field,ideal_field))
        push!(direct_errors,relative(direct,implied))
        @test relative(direct,implied) <= 1e-9
        @test relative(direct,load) <= 1e-9
        @test relative(load,ideal_load) <= 1e-9
        @test relative(field,ideal_field) <= 1e-9
        if !coupled
            @test relative(ideal_load,bq["radiation_impedance_matrix"]) <= 1e-9
            @test relative(diag(ideal_load),vec(bq["radiation_impedance"])) <= 1e-9
        end
        append!(ideal_loads,diag(ideal_load)); append!(ideal_fields,vec(ideal_field))
        # Signed resistance/reactance are diagnostics, with no separate budget.
        d=merge(label,Dict("path"=>coupled ? "coupled" : "exterior",
            "frequency_hz"=>a["freq_hz"],"resistance"=>real.(diag(load)),
            "reactance"=>imag.(diag(load)),"ideal_resistance"=>real.(diag(ideal_load)),
            "ideal_reactance"=>imag.(diag(ideal_load)),"load_operator_relative"=>last(load_errors),
            "field_operator_relative"=>last(field_errors),"direct_implied_relative"=>last(direct_errors)))
        push!(diagnostics,d)
        println("AGREEMENT_CONTROL_SAMPLE ",JSON.json(merge(d,Dict(
            "ideal_self_load"=>complex_pairs(diag(ideal_load)),"ideal_field"=>complex_pairs(ideal_field)))))
    end
    Dict("load_operator_relative"=>maximum(load_errors),"field_operator_relative"=>maximum(field_errors),
        "direct_implied_relative"=>maximum(direct_errors),"diagnostics"=>diagnostics),
        Dict("load"=>ideal_loads,"pressure"=>ideal_fields)
end

function ratio_metrics(ratios)
    Dict("magnitude"=>maximum(abs.(abs.(ratios).-1)),
        "phase_deg"=>maximum(abs.(rad2deg.(angle.(ratios)))),"complex"=>maximum(abs.(ratios.-1)))
end

function qualification()
    records=Any[]
    mktempdir() do dir
        for case in ("sphere","box","box_pair"), level in (1,2)
            mesh=case=="sphere" ? sphere(0.1,SVector(0.,0.,0.),2;refinements=level+1) : box(2^(level+2))
            # Single patch: its neighbour is rigid; pair: neighbour is shorted.
            case=="box" && (mesh=BoundaryMesh(mesh.vertices,mesh.faces,replace(mesh.physical_tags,3=>1)))
            tags=case=="box_pair" ? [2,3] : [2]
            exteriorpath=joinpath(dir,"exterior.msh")
            gmsh(exteriorpath,mesh.vertices,mesh.faces,mesh.physical_tags)
            exterior=request(exteriorpath,tags;control=case!="sphere")
            case=="sphere" && filter!(b->b["id"]!="rigid",exterior["compiled_system"]["boundaries"])
            for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
                exterior["solver_options"]["phasor_convention"]=convention
                er=captured(exterior)
                ideal_exterior=case=="sphere" ? nothing : request(exteriorpath,tags;ideal=true)
                if ideal_exterior!==nothing
                    ideal_exterior["solver_options"]["phasor_convention"]=convention
                end
                ie=ideal_exterior===nothing ? nothing : captured(ideal_exterior)
                thicknesses=case=="sphere" ? [0.01,0.005] : [0.00125/2^(level-1)]
                for h in thicknesses
                    scale=1+h/(case=="sphere" ? 0.1 : 0.06)
                    vertices,inner,outer,tets=shell(mesh,scale)
                    fempath,bempath=joinpath(dir,"shell.msh"),joinpath(dir,"interface.msh")
                    gmsh(bempath,scale.*mesh.vertices,mesh.faces,fill(4,length(mesh.faces)))
                    cr=request(bempath,tags;coupled=true,fempath=fempath,nodes=length(mesh.vertices),faces=length(mesh.faces),control=case!="sphere")
                    cr["compiled_system"]["meshes"][2]["mesh_data"]=fem_payload(vertices,inner,outer,mesh.physical_tags,tets)
                    cr["compiled_system"]["meshes"][2]["file"]=""
                    case=="sphere" && filter!(b->b["id"]!="rigid",cr["compiled_system"]["boundaries"])
                    cr["solver_options"]["phasor_convention"]=convention
                    actual_coupled=captured(cr)
                    label=Dict("case"=>case,"level"=>level,"thickness_m"=>h,"phasor"=>convention)
                    metrics=compare(er,actual_coupled,convention;label=label)
                    record=merge(label,Dict("bem_triangles"=>length(mesh.faces),"fem_nodes"=>length(vertices),
                        "tetrahedra"=>length(tets),"metrics"=>metrics))
                    if case!="sphere"
                        ideal_coupled=request(bempath,tags;coupled=true,fempath=fempath,
                            nodes=length(mesh.vertices),faces=length(mesh.faces),ideal=true)
                        ideal_coupled["compiled_system"]["meshes"][2]["mesh_data"]=cr["compiled_system"]["meshes"][2]["mesh_data"]
                        ideal_coupled["compiled_system"]["meshes"][2]["file"]=""
                        ideal_coupled["solver_options"]["phasor_convention"]=convention
                        ic=captured(ideal_coupled)
                        ce,oe=control_metrics(er,ie,mesh,tags,convention;coupled=false,label=label)
                        cc,oc=control_metrics(actual_coupled,ic,mesh,tags,convention;coupled=true,label=label)
                        record["controls"]=Dict("exterior"=>ce,"coupled"=>cc)
                        record["ideal_metrics"]=Dict(q=>ratio_metrics(oc[q]./oe[q]) for q in ("load","pressure"))
                    end
                    push!(records,record)
                    println("AGREEMENT ",JSON.json(record))
                    flush(stdout)
                end
            end
        end
    end
    records
end

@testset "exterior / coupled thin-air agreement (fixed budgets)" begin
    # Match the diagnostic's CPU configuration, restoring ambient BLAS afterwards.
    previous_threads=BLAS.get_num_threads()
    started=time_ns()
    records=try
        BLAS.set_num_threads(1)
        qualification()
    finally
        BLAS.set_num_threads(previous_threads)
    end
    println("AGREEMENT_RUNTIME_SECONDS ",(time_ns()-started)/1e9)
    println("AGREEMENT_COUNTS ",JSON.json(COUNTS))
    report=get(ENV,"BEAT_AGREEMENT_REPORT","")
    isempty(report) || open(io->JSON.print(io,records,2),report,"w")
    # All measurements are emitted before budget assertions, preserving evidence.
    for record in records
        record["level"]==2 || continue
        for (q,(mag,phase)) in BUDGETS
            @test record["metrics"][q]["magnitude"] <= mag
            @test record["metrics"][q]["phase_deg"] <= phase
        end
        if haskey(record,"ideal_metrics")
            for q in ("load","pressure")
                @test record["ideal_metrics"][q]["magnitude"] <= BUDGETS[q][1]
                @test record["ideal_metrics"][q]["phase_deg"] <= BUDGETS[q][2]
            end
        end
    end
    for case in ("sphere","box","box_pair"), convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR), q in keys(BUDGETS)
        selected=filter(r->r["case"]==case && r["phasor"]==convention,records)
        for h in (case=="sphere" ? [0.01,0.005] : [nothing])
            levels=[only(filter(r->r["level"]==l && (h===nothing || r["thickness_m"]==h),selected)) for l in (1,2)]
            @test levels[2]["metrics"][q]["complex"] < levels[1]["metrics"][q]["complex"]
            if case!="sphere" && q in ("load","pressure")
                @test levels[2]["ideal_metrics"][q]["complex"] < levels[1]["ideal_metrics"][q]["complex"]
            end
        end
    end
    for level in (1,2), convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
        selected=filter(r->r["case"]=="sphere" && r["level"]==level && r["phasor"]==convention,records)
        worst(h)=maximum(values(only(filter(r->r["thickness_m"]==h,selected))["metrics"])) do metric
            metric["complex"]
        end
        @test worst(0.005) < worst(0.01)
    end
end
end # module
