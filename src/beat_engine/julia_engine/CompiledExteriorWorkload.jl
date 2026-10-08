# Shared by the compiled CPU and Metal bundles. The plate touches both symmetry
# planes, has non-adjacent triangles, and has singular pairs with its images.
function workload_plate_mesh()
    io = IOBuffer()
    print(io, WORKLOAD_HEAD, "\$Nodes\n9\n")
    for y in 0:2, x in 0:2
        println(io, 1 + x + 3y, " ", 0.04x, " ", 0.04y, " 0.0")
    end
    print(io, "\$EndNodes\n\$Elements\n8\n")
    face = 0
    for y in 0:1, x in 0:1
        a = 1 + x + 3y
        for vertices in ((a, a + 1, a + 4), (a, a + 4, a + 3))
            face += 1
            println(io, face, " 2 2 2 2 ", join(vertices, " "))
        end
    end
    print(io, "\$EndElements\n")
    return String(take!(io))
end

function representative_workload_request(mesh)
    request = workload_request(mesh, "xy")
    request["frequencies_hz"] = [1000.0, 20000.0]
    request["solver_options"] = merge(Dict{String,Any}(request["solver_options"]), Dict(
        "quadrature_order" => 4, "singular_order" => 4,
        "regular_quadrature_mode" => "fixed",
    ))
    sphere = [[sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
              for theta in range(0.0, pi; length=37)
              for phi in range(0.0, 2pi; length=73)[1:72]]
    diagonal = [[sin(a) / sqrt(2), sin(a) / sqrt(2), cos(a)]
                for a in range(0.0, pi; length=37)]
    append!(request["outputs"], [
        Dict("id" => "sphere", "quantity" => "exterior_pressure", "target_ids" => [],
             "options" => Dict("points_m" => sphere)),
        Dict("id" => "diagonal", "quantity" => "exterior_pressure", "target_ids" => [],
             "options" => Dict("points_m" => diagonal)),
        Dict("id" => "surface:p", "quantity" => "bem_boundary_pressure", "target_ids" => [],
             "options" => Dict()),
        Dict("id" => "surface:q", "quantity" => "bem_boundary_neumann", "target_ids" => [],
             "options" => Dict()),
    ])
    return request
end

# Closed octahedron, and its exact quadrant cut. Host-only network compilation
# avoids GPU launches during package image generation.
function workload_transducer_mesh(symmetry)
    vertices = ((0.1,0.,0.),(-0.1,0.,0.),(0.,0.1,0.),(0.,-0.1,0.),(0.,0.,0.1),(0.,0.,-0.1))
    faces = ((1,3,5),(3,2,5),(2,4,5),(4,1,5),(3,1,6),(2,3,6),(4,2,6),(1,4,6))
    if symmetry == "x"
        vertices = (vertices[1],vertices[3],vertices[4],vertices[5],vertices[6])
        faces = ((1,2,4),(3,1,4),(2,1,5),(1,3,5))
    elseif symmetry == "xy"
        vertices = (vertices[1],vertices[3],vertices[5],vertices[6])
        faces = ((1,2,3),(2,1,4))
    end
    io = IOBuffer()
    print(io, WORKLOAD_HEAD, "\$Nodes\n",length(vertices),"\n")
    for (i,v) in enumerate(vertices)
        println(io,i," ",join(v," "))
    end
    print(io,"\$EndNodes\n\$Elements\n",length(faces),"\n")
    for (i,f) in enumerate(faces)
        println(io,i," 2 2 2 2 ",join(f," "))
    end
    print(io,"\$EndElements\n")
    String(take!(io))
end

function transducer_workload_request(mesh, symmetry; precision="float64")
    r=workload_request(mesh,symmetry)
    s=r["compiled_system"];s["contract_version"]=3
    s["meshes"][1]["scale_to_m"]=1.0
    component=only(s["components"]);component["kind"]="electrodynamic_transducer"
    c=symmetry=="xy" ? 4 : symmetry=="x" ? 2 : 1
    component["parameters"]=Dict("re_ohm"=>6.,"le_h"=>0.0005,"bl_n_per_a"=>7.,"mmd_kg"=>0.015,
        "cms_m_per_n"=>0.0005,"rms_n_s_per_m"=>0.6,"motion_axis"=>[0.,0.,1.],
        "surface_completion_factor"=>c,"physical_driver_orbit_count"=>1,
        "symmetry_role"=>c>1 ? "fractional_driver" : "complete_representative",
        "fractional_symmetry_axes"=>c==4 ? ["x","y"] : c==2 ? ["x"] : String[])
    only(s["excitation_ports"])["kind"]="voltage"
    r["frequencies_hz"]=[54.,100.]
    r["solver_options"]["precision"]=precision
    r["outputs"]=[Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=> q=="exterior_pressure" ?
        Dict("points_m"=>[[0.3,0.2,1.]]) : Dict()) for q in
        ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure")]
    r
end

function precompile_exterior_transducer_workload()
    mktempdir() do directory
        for symmetry in ("off","xy"), precision in ("float64","float32")
            path=joinpath(directory,"transducer-$symmetry.msh")
            write(path,workload_transducer_mesh(symmetry))
            request=JSON.parse(JSON.json(transducer_workload_request(path,symmetry;precision=precision)))
            redirect_stdout(devnull) do
                try
                    solve_request(request;event_mode=true)
                catch exception
                    @warn "BEAT compiled exterior transducer workload failed" symmetry exception=(exception,catch_backtrace())
                end
            end
        end
    end
end
