# CPU-resident continuous-speaker-v1 models for the Deploy exterior worker.
using Base64, SHA, SparseArrays, JSON
import p7zip_jll

const DEPLOY_CONTINUOUS_MODELS = Dict{String,Any}()
const DeployFrequencyKey = Tuple{String,Float64,DataType}
const DEPLOY_CONTINUOUS_RECENT = Dict{DeployFrequencyKey,Tuple{Any,Int}}()
const DEPLOY_CONTINUOUS_LRU = DeployFrequencyKey[]
const DEPLOY_CONTINUOUS_CACHE_BYTES = 128 * 1024^2

function prune_deploy_continuous_models!(rom)
    models = rom isa AbstractDict ? values(get(rom, "models", Dict("" => rom))) : ()
    active = Set(m["compact_model"]["sha256"] for m in models if haskey(m, "compact_model"))
    filter!(kv -> first(kv) in active, DEPLOY_CONTINUOUS_MODELS)
    filter!(kv -> first(kv)[1] in active, DEPLOY_CONTINUOUS_RECENT)
    filter!(key -> key[1] in active, DEPLOY_CONTINUOUS_LRU)
end

function unpack_deploy_matrix(entry)
    rows, cols = Int.(entry["shape"])
    if entry["storage"] == "coo"
        values = Float64.(entry["values"])
        all(isfinite, values) || error("Nonfinite continuous ROM coefficients")
        return sparse(Int.(entry["rows"]) .+ 1, Int.(entry["columns"]) .+ 1, values, rows, cols)
    end
    entry["storage"] == "f64le-column-major" || error("Unsupported continuous ROM matrix encoding")
    raw = reinterpret(UInt64, base64decode(entry["base64"]))
    matrix = reshape(reinterpret(Float64, ltoh.(raw)), rows, cols)
    all(isfinite, matrix) || error("Nonfinite continuous ROM coefficients")
    return matrix
end

function read_deploy_continuous_model(ref)
    path, member = ref["package"], ref["member"]
    isabspath(path) && endswith(lowercase(path), ".blabsp") || error("Invalid continuous speaker package path")
    !isempty(member) && !startswith(member, "@") || error("Invalid model member name")
    bytes = read(`$(p7zip_jll.p7zip()) x -so -tzip -spd -- $path $member`)
    bytes2hex(sha256(bytes)) == ref["sha256"] || error("Continuous speaker model checksum mismatch")
    data = JSON.parse(String(bytes))
    meta = data["metadata"]
    meta["format"] == "continuous-speaker-v1" && meta["phasor"] == "exp(+i omega t)" ||
        error("Unsupported continuous speaker model version or phasor")
    mats = Dict(Symbol(k) => unpack_deploy_matrix(v) for (k,v) in data["matrices"])
    m = (; mats...)
    nodes = findall(vec(any(!iszero, m.trace; dims=1)) .| vec(any(!iszero, m.bem_force; dims=2)))
    faces = findall(vec(any(!iszero, m.flux; dims=2)) .| vec(any(!iszero, m.bem_velocity; dims=2)))
    trace = m.trace[:,nodes]
    ng = size(trace,1)
    # The exterior pressure trace is independent of Hz. Project each coefficient
    # once, retaining only columns that feed free coordinates or boundary pressure.
    function project(matrix)
        blocks = hcat(matrix[:,ng+1:end], matrix[:,1:ng] * trace)
        blocks[1:ng,:] .*= -1
        return blocks
    end
    stiffness = project(m.K)
    inertia = project(m.M) - im*project(m.loss)
    walls = [(project(unpack_deploy_matrix(w["matrix"])), w["thickness_m"], w["flow_resistivity_pa_s_per_m2"])
             for w in data["walls"]]
    driver_pressure = transpose(m.bem_force[nodes,:]) - transpose(m.force[1:ng,:])*trace
    return (; stiffness, inertia, walls, driver_pressure, m.gram, m.surface, m.force, m.drive,
        flux=m.flux[faces,:], bem_velocity=m.bem_velocity[faces,:],
        nodes=[[i-1] for i in nodes], faces=[[i-1] for i in faces], drivers=data["drivers"],
        c=data["c"], rho=data["rho"], band=data["band"])
end

function assemble_deploy_continuous(m, hz, ::Type{T}) where T
    m.band[1] <= hz <= m.band[2] || error("Frequency outside continuous speaker band")
    s = 2im*pi*hz
    scale = -m.rho*s
    k = 2pi*hz/m.c
    blocks = m.stiffness .- k^2 .* m.inertia
    for (wall, thickness, resistivity) in m.walls
        ratio = hz/resistivity
        impedance = m.rho*m.c*(1 + (0.070-0.107im)*ratio^-0.632)
        porous_k = k*(1 + (0.109-0.160im)*ratio^-0.618)
        blocks .-= (scale*im*tan(porous_k*thickness)/impedance) .* wall
    end
    ng, nn = size(m.gram,1), length(m.nodes)
    n = size(blocks,1)
    nd, np = size(m.drive)
    rank = n+2nd
    g, free, vel, amp = 1:ng, ng+1:n, n+1:n+nd, n+nd+1:rank
    system = zeros(Complex{T}, rank, rank)
    pressure = zeros(Complex{T}, rank, nn)
    drive = zeros(Complex{T}, rank, np)
    flux = zeros(Complex{T}, length(m.faces), rank)
    system[g,g] = m.gram
    system[1:n,free] .= view(blocks,:,1:n-ng)
    system[g,vel] = scale*m.surface[g,:]
    pressure[1:n,:] .= view(blocks,:,n-ng+1:size(blocks,2))
    system[free,vel] = -scale*m.surface[free,:]
    system[vel,free] = -transpose(m.force[free,:])
    pressure[vel,:] = m.driver_pressure
    for (i,d) in enumerate(m.drivers)
        stiffness = inv(d["cms_m_per_n"])
        rear = d["lumped_sealed_rear_chamber"]
        if rear !== nothing
            stiffness += m.rho*m.c^2*rear["projected_area_m2"]^2/rear["volume_m3"]
        end
        coil = d["semi_inductance"]
        electrical = if coil === nothing
            d["re_ohm"] + s*d["le_h"]
        else
            parallel = inv(coil["rss_ohm"]) + inv(s*coil["le_h"]) + inv(sqrt(s)*coil["ke_semi_h"])
            coil["re_prime_ohm"] + s*coil["leb_h"] + inv(parallel)
        end
        system[vel[i],vel[i]] = d["rms_n_s_per_m"] + s*d["mmd_kg"] + stiffness/s
        system[amp[i],amp[i]] = electrical
        system[vel[i],amp[i]] = -d["bl_n_per_a"]
        system[amp[i],vel[i]] = d["bl_n_per_a"]
    end
    drive[amp,:] = m.drive
    flux[:,g] = m.flux
    flux[:,vel] = scale*m.bem_velocity
    velocity = zeros(Complex{T}, nd, rank)
    current = zeros(Complex{T}, nd, rank)
    for i in 1:nd
        velocity[i,vel[i]] = current[i,amp[i]] = 1
    end
    matrices = (; k=system, c=pressure, b=drive, d=flux, e=zeros(Complex{T},length(m.faces),np),
        velocity, current, velocity_drive=zeros(Complex{T},nd,np), current_drive=zeros(Complex{T},nd,np))
    arrays = Dict(string(k) => reshape(v,1,size(v)...) for (k,v) in pairs(matrices))
    return (; arrays, factors=[lu(system)], m.nodes, m.faces)
end

function deploy_continuous_frequency(ref, hz, ::Type{T}; budget=DEPLOY_CONTINUOUS_CACHE_BYTES) where T
    phasor_convention() == POSITIVE_TIME_PHASOR || error("Continuous speaker ROM requires positive-time phasors")
    key = (String(ref["sha256"]), Float64(hz), T)
    if haskey(DEPLOY_CONTINUOUS_RECENT, key)
        deleteat!(DEPLOY_CONTINUOUS_LRU, findfirst(==(key), DEPLOY_CONTINUOUS_LRU))
        push!(DEPLOY_CONTINUOUS_LRU, key)
        return first(DEPLOY_CONTINUOUS_RECENT[key])
    end
    model = get!(() -> read_deploy_continuous_model(ref), DEPLOY_CONTINUOUS_MODELS, key[1])
    prepared = assemble_deploy_continuous(model, key[2], T)
    bytes = Base.summarysize(prepared)
    if bytes <= budget
        retained = sum(last, values(DEPLOY_CONTINUOUS_RECENT); init=0)
        while retained + bytes > budget
            _, evicted_bytes = pop!(DEPLOY_CONTINUOUS_RECENT, popfirst!(DEPLOY_CONTINUOUS_LRU))
            retained -= evicted_bytes
        end
        DEPLOY_CONTINUOUS_RECENT[key] = (prepared, bytes)
        push!(DEPLOY_CONTINUOUS_LRU, key)
    end
    return prepared
end
