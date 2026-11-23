include("continuous_cuda_interior.jl")

# Continuous full-domain acoustic interior for Level 3 speaker export.
function continuous_prepare_interior(request)
    system = request["compiled_system"]
    options = request["solver_options"]
    get(options, "symmetry", "off") == "off" || error("Stage a fully expanded cabinet first")
    regions = system["regions"]
    air = only(filter(r -> r["kind"] == "unbounded_air", regions))
    bounded = filter(r -> r["kind"] == "bounded_air", regions)
    c, rho = air["sound_speed_m_per_s"], air["density_kg_per_m3"]
    all(r -> r["sound_speed_m_per_s"] == c && r["density_kg_per_m3"] == rho, bounded) ||
        error("Shared acoustic medium required")
    meshes, boundaries, components = system["meshes"], system["boundaries"], system["components"]
    all(comp -> comp["kind"] == "electrodynamic_transducer", components) || error("Voltage drivers required")
    fem = aggregate_fem_domains(meshes, bounded, boundaries, Float64;
        moving_boundary_ids=Set(String(id) for comp in components for id in comp["boundary_ids"]))
    bem = aggregate_bem_region(meshes, air, boundaries, Float64)
    !is_quadratic(fem.mesh) && isempty(fem.thermoviscous_walls) || error("Requires P1 FEM without thermoviscous walls")
    map = combined_interface_map_from_wire(system["interfaces"], fem, boundaries, bem).map
    interface = assemble_interface_operators(fem.mesh, bem.mesh, map)
    drivers, driver_index = electrodynamic_transducers_from_wire(components, boundaries,
        fem.fem_boundary_tag_by_id, bem.boundary_tag_by_id, fem.mesh, bem.mesh, Float64, :off)
    motion = assemble_transducer_operators(fem.mesh, bem.mesh, drivers)
    ports = String.(request["excitation_port_ids"])
    drive = zeros(length(drivers), length(ports))
    for (column, id) in enumerate(ports)
        port = only(filter(p -> p["id"] == id, system["excitation_ports"]))
        port["kind"] == "voltage" || error("Voltage port required")
        drive[driver_index[port["component_id"]], column] = 1
    end
    K, M = assemble_p1_fem_matrices(fem.mesh)
    g = map.fem_vertex_indices
    order = vcat(g, setdiff(axes(K, 1), g))
    permute(A) = A[order, order]
    walls = [(matrix=permute(assemble_boundary_mass_matrix(fem.mesh,
        findall(==(spec.tag), fem.mesh.boundary_physical_tags), collect(axes(K, 1)))),
        spec.thickness_m, spec.flow_resistivity_pa_s_per_m2) for spec in fem.wall_impedances]
    model = (; K=permute(K), M=permute(M), loss=permute(Diagonal(fem.bulk_loss_factor_by_vertex)*M),
        walls, surface=motion.fem_surface[order,:], force=motion.fem_force[order,:],
        trace=interface.bem_trace, gram=interface.fem_load[g,:], flux=interface.bem_flux,
        bem_force=motion.bem_force, bem_velocity=motion.bem_normal_velocity,
        drivers, drive, ports, c, rho, band=Tuple(Float64.(options["continuous_speaker_rom"]["band_hz"])))
    chambers = [findall(i -> d.vertex_offset < i <= d.vertex_offset+d.vertex_count, order) for d in fem.domains]
    return model, bem.mesh, chambers
end

function continuous_dynamic_matrix(m, f)
    omega = 2pi*f
    k2 = (omega/m.c)^2
    A = m.K - k2*m.M - im*propagation_sign()*k2*m.loss
    for w in m.walls
        Y = miki_rigid_backed_surface_admittance(f, m.c, m.rho, w.thickness_m, w.flow_resistivity_pa_s_per_m2)
        A -= neumann_scale(m.rho, omega)*Y*w.matrix
    end
    return A
end

function continuous_response(m, f, p, u, workspace=nothing)
    g = axes(m.gram, 1)
    inside = last(g)+1:size(m.K, 1)
    gamma = m.trace*p
    A = continuous_dynamic_matrix(m, f)
    scale = neumann_scale(m.rho, 2pi*f)
    zm = Diagonal([mechanical_impedance(d, 2pi*f, m.rho, m.c) for d in m.drivers])
    ze = Diagonal([electrical_impedance(d, 2pi*f) for d in m.drivers])
    bl = Diagonal([d.bl_n_per_a for d in m.drivers])
    # Prescribed interface pressures. Eliminate interior acoustic pressure, then
    # solve the remaining mechanical and electrical equations for all load cases.
    interior = A[inside,inside]
    factor = isnothing(workspace) || isempty(inside) ? lu(interior) : continuous_interior_factor!(workspace, interior, f)
    free = continuous_interior_solve(factor, -A[inside,g]*gamma)
    forced = continuous_interior_solve(factor, Matrix(scale*m.surface[inside,:]))
    force = transpose(m.force[inside,:])
    rhs = vcat(force*free + transpose(m.force[g,:])*gamma - transpose(m.bem_force)*p, m.drive*u)
    state = [zm-force*forced -bl; bl ze] \ rhs
    v, current = state[axes(zm,1),:], state[size(zm,1)+1:end,:]
    internal = free + forced*v
    residual = A[g,g]*gamma + A[g,inside]*internal - scale*m.surface[g,:]*v
    q = m.flux*(m.gram \ residual) + scale*m.bem_velocity*v
    return (; q, v, current, internal)
end

function continuous_probes(m, points, f, count, offset)
    p = BeatEngineSpeakerRom._sample_patterns(points, 2pi*f/m.c, count, offset, Float64)
    p ./= maximum(abs, p; dims=1)
    n = length(m.ports)
    return hcat(p, zeros(size(p,1), n)), hcat(zeros(n,count), Matrix{Float64}(I,n,n))
end

# Data-only exchange format for the continuous interior. Dense data
# is column-major little-endian Float64. Sparse coordinates are zero-based COO.
function continuous_pack_matrix(A::SparseMatrixCSC)
    rows, columns, values = findnz(A)
    return (; shape=size(A), storage="coo", rows=rows .- 1, columns=columns .- 1, values)
end

function continuous_pack_matrix(A::AbstractMatrix)
    bits = htol.(reinterpret(UInt64, vec(Matrix{Float64}(A))))
    return (; shape=size(A), storage="f64le-column-major", base64=base64encode(reinterpret(UInt8,bits)))
end

const CONTINUOUS_MODEL_MATRICES = (:K,:M,:loss,:surface,:force,:trace,:gram,:flux,:bem_force,:bem_velocity,:drive)

function continuous_model_payload(metadata, model, mesh)
    matrices = Dict(string(key) => continuous_pack_matrix(getproperty(model,key)) for key in CONTINUOUS_MODEL_MATRICES)
    walls = [(; matrix=continuous_pack_matrix(w.matrix), w.thickness_m, w.flow_resistivity_pa_s_per_m2) for w in model.walls]
    geometry = (; frame="right-handed +Y forward", length_unit="m", index_base=0,
        points=[[p[1],p[3],-p[2]] for p in mesh.vertices], triangles=[f .- 1 for f in mesh.faces])
    payload = (; metadata, geometry, matrices, walls, model.drivers, model.ports, model.c, model.rho, model.band)
    return JSON.json(payload)
end


function continuous_project(full, lift, modes, rank)
    ng = size(full.gram, 1)
    V = [Matrix{Float64}(I,ng,ng) zeros(ng,rank); lift modes[:,1:rank]]
    project(A) = transpose(V)*(A*V)
    m = merge(full, (; K=project(full.K), M=project(full.M), loss=project(full.loss),
        walls=[merge(w,(matrix=project(w.matrix),)) for w in full.walls],
        surface=transpose(V)*full.surface, force=transpose(V)*full.force))
    return V,m
end

function continuous_check(candidate, V, reference, chambers, f)
    p,u,field,q,v = reference
    b = continuous_response(candidate,f,p,u)
    estimated = V*vcat(candidate.trace*p,b.internal)
    return continuous_errors(estimated, b.q, b.v, field, q, v, chambers, f)
end

function continuous_errors(estimated, flux, velocity, field, q, v, chambers, f)
    field_error,flux_error,velocity_error = 0.,0.,0.
    for j in axes(field,2)
        for (idx,mass) in chambers
            x = field[idx,j]
            d = estimated[idx,j]-x
            field_error = max(field_error,sqrt(abs(dot(d,mass*d))/max(abs(dot(x,mass*x)),1e-28)))
        end
        flux_error = max(flux_error,norm(flux[:,j]-q[:,j])/max(norm(q[:,j]),1e-12))
        velocity_error = max(velocity_error,norm(velocity[:,j]-v[:,j])/max(norm(v[:,j]),1e-12))
    end
    return (;frequency_hz=f,field_error,flux_error,velocity_error)
end

# Independent coupled audit. The two monopoles are off-axis, one beyond the
# front of the cabinet and one beside it. Solve for total surface pressure with
# zero drive voltage, including both the incident pressure and its normal derivative.
function continuous_incident_fields(mesh, k)
    lo = reduce((a,b) -> min.(a,b), mesh.vertices)
    hi = reduce((a,b) -> max.(a,b), mesh.vertices)
    mid = (lo+hi)/2
    gap = 0.2
    sources = [SVector(mid[1]+0.19*(hi[1]-lo[1]),mid[2],hi[3]+gap),
               SVector(hi[1]+gap,mid[2]+0.12*(hi[2]-lo[2]),hi[3]-min(gap,(hi[3]-lo[3])/2))]
    p = zeros(ComplexF64,length(mesh.vertices),2)
    q = zeros(ComplexF64,length(mesh.faces),2)
    for (j,source) in enumerate(sources)
        wave(x) = cis(-k*norm(x-source))/norm(x-source)
        p[:,j] = wave.(mesh.vertices)
        for (i,vs) in enumerate(mesh.face_vertices)
            for t in 1:3
                x = (4vs[t]+vs[mod1(t+1,3)]+vs[mod1(t+2,3)])/6
                r = x-source
                q[i,j] += wave(x)*(-im*k-inv(norm(r)))*dot(r,mesh.normals[i])/(3norm(r))
            end
        end
    end
    return p,q
end

function continuous_loaded_response(m, V, f, A, B, rhs, active, inputs, workspace=nothing)
    transfer = continuous_response(m, f, inputs, zeros(length(m.ports),length(active)),workspace)
    coupled = copy(A)
    coupled[:,active] -= B*transfer.q
    p = coupled \ rhs
    response = continuous_response(m, f, p, zeros(length(m.ports),size(p,2)),workspace)
    state = vcat(m.trace*p,response.internal)
    return (isnothing(V) ? state : V*state),response.q,response.v
end

function continuous_coupled_check(full, m, V, mesh, chambers, f, workspace)
    k = 2pi*f/full.c
    p1,dp,rule = build_p1_space(mesh),build_dp0_space(mesh),triangle_rule(Float64,2)
    ipp = assemble_l2_identity_matrix(mesh,p1,dp,rule,:p1,:p1)
    ipq = assemble_l2_identity_matrix(mesh,p1,dp,rule,:p1,:dp0)
    op = assemble_regular_galerkin_operators(mesh,p1,dp,k,rule;backend=:cpu,singular_order=2,skip_singular=false)
    A,B = BeatEngineCore.burton_miller_neumann_matrices(op,ipp,ipq,k)
    pin,qin = continuous_incident_fields(mesh,k)
    rhs = A*pin-B*qin
    active = sort(union(findnz(full.trace)[2],findnz(full.bem_force)[1]))
    inputs = zeros(size(full.trace,2),length(active))
    for (j,node) in enumerate(active)
        inputs[node,j] = 1
    end
    reference = continuous_loaded_response(full,nothing,f,A,B,rhs,active,inputs,workspace)
    candidate = continuous_loaded_response(m,V,f,A,B,rhs,active,inputs)
    return continuous_errors(candidate...,reference...,chambers,f)
end

function train_continuous_speaker(request, cancelled, bem_backend)
    workspace = bem_backend == :cuda ? ContinuousCudaInterior() : nothing
    try
        return train_continuous_speaker_model(request, cancelled, workspace)
    finally
        isnothing(workspace) || release_continuous_interior!(workspace)
    end
end

function train_continuous_speaker_model(request, cancelled, workspace)
    phasor_convention()==POSITIVE_TIME_PHASOR || error("Continuous speaker export requires exp(+i omega t)")
    options = request["solver_options"]["continuous_speaker_rom"]
    full,mesh,domains = continuous_prepare_interior(request)
    lo,hi = full.band
    0 < lo < hi && isfinite(hi) || error("Invalid continuous speaker frequency band")
    maxrank,maxtrain,ncheck = Int(options["max_rank"]),Int(options["max_training"]),Int(options["validation_count"])
    maxrank>=1 && maxtrain>=2 && ncheck>=1 || error("Invalid continuous speaker ROM training limits")
    training = exp.(range(log(lo),log(hi);length=min(8,maxtrain)))
    training[1],training[end] = lo,hi
    ng = size(full.gram,1)
    inside = ng+1:size(full.K,1)
    shifted = full.K+full.M
    lift = -(lu(shifted[inside,inside]) \ Matrix(shifted[inside,1:ng]))
    chambers = [(idx,full.M[idx,idx]) for idx in domains]
    snapshots = Dict{Float64,Matrix{Float64}}()
    references = Dict()
    target = 0.01
    while true
        println(stderr, "Continuous ROM: $(length(training)) training frequencies over $lo to $hi Hz")
        for f in training
            cancelled() && return nothing
            get!(snapshots,f) do
                p,u = continuous_probes(full,mesh.vertices,f,8,0)
                delta = continuous_response(full,f,p,u,workspace).internal-lift*(full.trace*p)
                delta ./= max.(sqrt.(sum(abs2,delta;dims=1)),eps())
                hcat(real(delta),imag(delta))
            end
        end
        modes = svd(hcat([snapshots[f] for f in training]...);full=false).U
        ranklimit = min(maxrank,size(modes,2))
        # Spread independent checks across the band and recheck between newly inserted snapshots.
        uniform = [lo+(hi-lo)*(i-0.5)/ncheck for i in 1:ncheck]
        checks = sort(setdiff(unique(vcat(uniform,sqrt.(training[1:end-1].*training[2:end]))),training))
        failures = Float64[]
        for rank in unique(min.([32,64,96,ranklimit],ranklimit))
            V,m = continuous_project(full,lift,modes,rank)
            rows = NamedTuple[]
            for f in checks
                cancelled() && return nothing
                reference = get!(references,f) do
                    p,u = continuous_probes(full,mesh.vertices,f,6,101)
                    a = continuous_response(full,f,p,u,workspace)
                    (p,u,vcat(full.trace*p,a.internal),a.q,a.v)
                end
                push!(rows,continuous_check(m,V,reference,chambers,f))
            end
            errors = [max(a.field_error,a.flux_error,a.velocity_error) for a in rows]
            all(isfinite,errors) || error("Nonfinite continuous ROM validation error")
            if maximum(errors)<=target
                # Choose stress points from this design plus a frequency strictly
                # inside its largest gap. These coupled loads never generate snapshots.
                largest = argmax(diff(training))
                metrics = (:field_error, :flux_error, :velocity_error)
                audit = [argmax(row -> getproperty(row, metric), rows).frequency_hz for metric in metrics]
                push!(audit, training[largest]+0.37*(training[largest+1]-training[largest]))
                unique!(audit)
                println(stderr, "Checking rank $rank against external sources at $(length(audit)) frequencies")
                external = NamedTuple[]
                for f in audit
                    cancelled() && return nothing
                    push!(external,continuous_coupled_check(full,m,V,mesh,chambers,f,workspace))
                end
                external_errors = [max(a.field_error,a.flux_error,a.velocity_error) for a in external]
                all(isfinite,external_errors) || error("Nonfinite coupled external-source validation error")
                if maximum(external_errors)>target
                    failures = audit[filter(i -> external_errors[i]>target,sortperm(external_errors;rev=true))]
                    continue
                end
                metadata = (;format="continuous-speaker-v1",phasor=phasor_convention(),rank,training,
                    interior_factorization=isnothing(workspace) ? "cpu_fp64" : "cuda_cudss_fp64",
                    meshes=[(;id=t["id"],sha256=t["sha256"]) for t in RUN_MESH_PROVENANCE[]],validation=(;kind="prescribed_and_external_loading",
                    relative_tolerance=target, checks=rows, external_source_checks=external))
                return continuous_model_payload(metadata,m,mesh)
            end
            failures = checks[filter(i -> errors[i]>target,sortperm(errors;rev=true))]
        end
        length(training)<maxtrain || error("Continuous speaker ROM failed its 1% validation target near $(first(failures)) Hz at rank $ranklimit")
        append!(training,first(failures,min(length(failures),4,maxtrain-length(training))))
        sort!(training)
    end
end
