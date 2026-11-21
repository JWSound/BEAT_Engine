# Entry-wise regular Galerkin Burton–Miller kernel (KernelAbstractions: CPU / CUDA / Metal).
# out[m] = A_reg[EI[m], EJ[m]] = sum over faces e∋i (local a), faces f∋j (local b), image sets s,
#   quadrature q,p:  We[q] Wf[p] * ( phi_a(q) phi_b(p) * (-dG/dn_y - eta k^2 (n_x·n_y) G) + eta (curl_a·curl_b) G )
# G = e^{ikr}/(4 pi r), y on the (image) face f of set s, x on the base face e.
# Exactly the integrand of reg_block! in hmat_fast.jl; singular/near corrections, row weights and
# the 0.5*M term stay on the CPU (sparse).
using KernelAbstractions, Adapt

struct KAGeom{T,A3,A4,A2,AI}
    pts::A4      # 3 x nq x ne x nsets
    nrm::A3      # 3 x ne x nsets
    curl::A4     # 3 x 3 x ne x nsets   (component, local basis, elem, set)
    W::A2        # nq x ne
    basis::A2    # 3 x nq
    vf_ptr::AI   # CSR: faces of vertex v are vf_face[vf_ptr[v]:vf_ptr[v+1]-1]
    vf_face::AI
    vf_loc::AI   # local index (1..3) of v in that face
    nq::Int32; nsets::Int32
    k::T; eta_re::T; eta_im::T
end

function ka_geom(o, ::Type{T}, todev = identity) where {T}
    ne = length(o.dofs); nq = length(o.basis); ns = length(o.pts)
    pts = zeros(T, 3, nq, ne, ns); nrm = zeros(T, 3, ne, ns); curl = zeros(T, 3, 3, ne, ns)
    for s in 1:ns, e in 1:ne
        pts[:, :, e, s] .= o.pts[s][e]; nrm[:, e, s] .= o.nrm[s][e]
        for a in 1:3; curl[:, a, e, s] .= o.curl[s][e][a]; end
    end
    W = zeros(T, nq, ne); for e in 1:ne; W[:, e] .= o.W[e]; end
    basis = zeros(T, 3, nq); for q in 1:nq; basis[:, q] .= o.basis[q]; end
    n = length(o.vfaces)
    ptr = Int32[1]; fc = Int32[]; lc = Int32[]
    for v in 1:n
        for e in o.vfaces[v]
            a = findfirst(==(v), o.dofs[e]); a === nothing && continue
            push!(fc, e); push!(lc, a)
        end
        push!(ptr, length(fc) + 1)
    end
    KAGeom{T,typeof(todev(nrm)),typeof(todev(pts)),typeof(todev(W)),typeof(todev(ptr))}(
        todev(pts), todev(nrm), todev(curl), todev(W), todev(basis), todev(ptr), todev(fc), todev(lc),
        Int32(nq), Int32(ns), T(o.k), T(real(o.eta)), T(imag(o.eta)))
end

@kernel function _reg_entries!(out_re, out_im, @Const(EI), @Const(EJ), g)
    m = @index(Global, Linear)
    T = eltype(out_re)
    i = EI[m]; j = EJ[m]
    k = g.k; er = g.eta_re; ei = g.eta_im; k2 = k * k
    inv4pi = T(1) / (T(4) * T(pi))
    accr = zero(T); acci = zero(T)
    for pe in g.vf_ptr[i]:(g.vf_ptr[i + 1] - 1)
        e = g.vf_face[pe]; a = g.vf_loc[pe]
        for pf in g.vf_ptr[j]:(g.vf_ptr[j + 1] - 1)
            f = g.vf_face[pf]; b = g.vf_loc[pf]
            for s in 1:g.nsets
                n1 = g.nrm[1, f, s]; n2 = g.nrm[2, f, s]; n3 = g.nrm[3, f, s]
                nn = g.nrm[1, e, 1] * n1 + g.nrm[2, e, 1] * n2 + g.nrm[3, e, 1] * n3
                cc = g.curl[1, a, e, 1] * g.curl[1, b, f, s] + g.curl[2, a, e, 1] * g.curl[2, b, f, s] +
                     g.curl[3, a, e, 1] * g.curl[3, b, f, s]
                for q in 1:g.nq
                    x1 = g.pts[1, q, e, 1]; x2 = g.pts[2, q, e, 1]; x3 = g.pts[3, q, e, 1]
                    wq = g.W[q, e]; bq = g.basis[a, q]
                    for p in 1:g.nq
                        d1 = g.pts[1, p, f, s] - x1; d2 = g.pts[2, p, f, s] - x2; d3 = g.pts[3, p, f, s] - x3
                        r2 = d1 * d1 + d2 * d2 + d3 * d3
                        if r2 > zero(T)
                            r = sqrt(r2); ir = one(T) / r
                            sn, cs = sincos(k * r)
                            gr = cs * inv4pi * ir; gi = sn * inv4pi * ir          # G
                            dn = (d1 * n1 + d2 * n2 + d3 * n3) * ir
                            # dG = G (ik - 1/r) dn
                            dgr = (gr * (-ir) - gi * k) * dn; dgi = (gr * k + gi * (-ir)) * dn
                            w = wq * g.W[p, f]; bb = bq * g.basis[b, p]
                            # term1 = bb * (-dG - eta k^2 nn G); term2 = eta cc G
                            c = k2 * nn
                            t1r = -dgr - c * (er * gr - ei * gi); t1i = -dgi - c * (er * gi + ei * gr)
                            t2r = cc * (er * gr - ei * gi); t2i = cc * (er * gi + ei * gr)
                            accr += w * (bb * t1r + t2r); acci += w * (bb * t1i + t2i)
                        end
                    end
                end
            end
        end
    end
    out_re[m] = accr; out_im[m] = acci
end

"""Regular part A_reg[EI[m], EJ[m]] for all m, on the backend of g's arrays."""
function ka_reg_entries(g::KAGeom{T}, EI, EJ) where {T}
    be = get_backend(g.pts)
    n = length(EI)
    out_re = KernelAbstractions.zeros(be, T, n); out_im = KernelAbstractions.zeros(be, T, n)
    _reg_entries!(be, 256)(out_re, out_im, EI, EJ, g; ndrange = n)
    KernelAbstractions.synchronize(be)
    return out_re, out_im
end
Adapt.@adapt_structure KAGeom

"""Hybrid build: all dense near-field leaf blocks on the device (one launch, precision of g), ACA blocks on CPU threads."""
function hbuild_hybrid(o, g::KAGeom, todev, root::HCluster, adm, tol::Float64, n::Int)
    adm_pairs = Tuple{HCluster,HCluster}[]; dense_pairs = Tuple{HCluster,HCluster}[]
    rec(a, b) = adm(a, b) ? push!(adm_pairs, (a, b)) :
        (isempty(a.kids) || isempty(b.kids)) ? push!(dense_pairs, (a, b)) :
        foreach(((ca, cb),) -> rec(ca, cb), Iterators.product(a.kids, b.kids))
    rec(root, root)
    # --- near entries (where the singular/near correction lives): the regular part there cancels by up to
    # 11 orders of magnitude against the correction (image faces on the symmetry planes), so these
    # entries are computed exactly in Float64 on the CPU: nearEx = reg64 + corr.
    tn = @elapsed begin
        NI, NJ, NV = findnz(o.corr)
        g64 = ka_geom(o, Float64)
        r64, i64 = ka_reg_entries(g64, Int32.(NI), Int32.(NJ))
        nearEx = sparse(NI, NJ, complex.(r64, i64) .+ NV, n, n)
        g64 = nothing
    end
    # --- dense blocks on the device
    EI = Int32[]; EJ = Int32[]
    for (a, b) in dense_pairs, j in b.idx, i in a.idx; push!(EI, i); push!(EJ, j); end
    td = @elapsed begin
        re, im_ = ka_reg_entries(g, todev(EI), todev(EJ))
        re = Array(re); im_ = Array(im_)
    end
    off = 0
    dn = Vector{DenseBlk{ComplexF64}}(undef, length(dense_pairs))
    offs = cumsum([0; [length(a.idx) * length(b.idx) for (a, b) in dense_pairs]])
    Threads.@threads for t in eachindex(dense_pairs)
        a, b = dense_pairs[t]; I = a.idx; J = b.idx; r = offs[t] .+ (1:length(I) * length(J))
        B = reshape(complex.(Float64.(re[r]), Float64.(im_[r])), length(I), length(J))
        S = nearEx[I, J]; rv = rowvals(S); nz = nonzeros(S)
        for c in 1:size(S, 2), q in nzrange(S, c); B[rv[q], c] = nz[q]; end     # exact near entries
        B .*= o.Wrow[I]; B .+= 0.5 .* Matrix(o.Ipp[I, J])
        dn[t] = DenseBlk(I, J, B)
    end
    re = nothing; im_ = nothing
    # --- low-rank blocks: ACA on CPU threads (exact Float64 oracle)
    corrT = sparse(transpose(o.corr)); IppT = sparse(transpose(o.Ipp))
    fcache = IdDict{Vector{Int},Vector{Int}}()
    for (a, b) in adm_pairs
        haskey(fcache, a.idx) || (fcache[a.idx] = _faces(o, a.idx)); haskey(fcache, b.idx) || (fcache[b.idx] = _faces(o, b.idx))
    end
    pool = Channel{Tuple{Vector{Int},Vector{Int}}}(Threads.nthreads() + 1)
    for _ in 1:Threads.nthreads() + 1; put!(pool, (zeros(Int, n), zeros(Int, n))); end
    lr = Vector{LowRank{ComplexF64}}(undef, length(adm_pairs))
    order = sortperm([length(a.idx) * length(b.idx) for (a, b) in adm_pairs]; rev = true)
    ta = @elapsed Threads.@threads :greedy for t in order
        bufs = take!(pool); mI, mJ = bufs
        try
            a, b = adm_pairs[t]; I = a.idx; J = b.idx; FI = fcache[I]; FJ = fcache[J]
            U, V = aca(i -> exact_row(o, corrT, IppT, I[i], J, FJ, mI, mJ),
                       j -> exact_col(o, I, FI, J[j], mI, mJ), length(I), length(J), tol / 10)
            U, V = recompress(U, V, tol)
            lr[t] = LowRank(I, J, U, V)
        finally
            put!(pool, bufs)
        end
    end
    return HMat{ComplexF64}(n, n, lr, dn), td + tn, ta
end

# ---------------- batched ACA on the device ----------------
mutable struct ACAState
    I::Vector{Int}; J::Vector{Int}
    U::Vector{Vector{ComplexF64}}; V::Vector{Vector{ComplexF64}}
    usedr::BitVector; i::Int; j::Int; norm2::Float64
    r::Vector{ComplexF64}; piv::ComplexF64
    phase::Symbol      # :row, :col, :done
    steps::Int
    hasnear::Bool      # block contains near-correction or identity entries (needs the exact fixup)
end
ACAState(I, J, hn = true) = ACAState(I, J, Vector{ComplexF64}[], Vector{ComplexF64}[], falses(length(I)), 1, 0, 0.0,
                          ComplexF64[], 0im, :row, 0, hn)

# exact entries for a list of (i, j) from device regular parts + exact near entries + weights + identity
function _finish_entries!(vals::AbstractVector{ComplexF64}, EI, EJ, nearEx::SparseMatrixCSC, o, need = nothing)
    @inbounds for m in eachindex(vals)
        i = EI[m]; j = EJ[m]
        if need !== nothing && !need[m]
            vals[m] *= o.Wrow[i]; continue
        end
        rng = nzrange(nearEx, j); rv = rowvals(nearEx)
        k = searchsortedfirst(view(rv, rng), i)
        if k <= length(rng) && rv[rng[k]] == i
            vals[m] = nonzeros(nearEx)[rng[k]]
        end
        vals[m] *= o.Wrow[i]
        rngI = nzrange(o.Ipp, j); rvI = rowvals(o.Ipp)
        kI = searchsortedfirst(view(rvI, rngI), i)
        if kI <= length(rngI) && rvI[rngI[kI]] == i
            vals[m] += 0.5 * nonzeros(o.Ipp)[rngI[kI]]
        end
    end
    return vals
end

function _device_entries(g, todev, o, nearEx, EI::Vector{Int32}, EJ::Vector{Int32}, need = nothing)
    isempty(EI) && return ComplexF64[]
    re, im_ = ka_reg_entries(g, todev(EI), todev(EJ))
    vals = complex.(Float64.(Array(re)), Float64.(Array(im_)))
    return _finish_entries!(vals, EI, EJ, nearEx, o, need)
end

"""Batched ACA (partial pivoting, same algorithm as `aca`) for all admissible blocks at once:
every round evaluates the pending row (or column) of every active block in one device launch."""
function aca_batched(states::Vector{ACAState}, entries, tol::Float64)
    rounds = 0; tdev = 0.0
    while true
        active = [s for s in states if s.phase != :done]
        isempty(active) && break
        rounds += 1
        rowS = [s for s in active if s.phase == :row]; colS = [s for s in active if s.phase == :col]
        # ---- rows
        if !isempty(rowS)
            EI = Int32[]; EJ = Int32[]; off = Int[0]
            need = Bool[]
            for s in rowS; ii = s.I[s.i]; for j in s.J; push!(EI, ii); push!(EJ, j); push!(need, s.hasnear); end; push!(off, length(EI)); end
            tdev += @elapsed vals = entries(EI, EJ, need)
            Threads.@threads for t in eachindex(rowS)
                s = rowS[t]; r = vals[off[t]+1:off[t+1]]
                for q in eachindex(s.U); r .-= s.U[q][s.i] .* conj.(s.V[q]); end
                s.usedr[s.i] = true
                j = argmax(abs.(r)); piv = r[j]
                if abs(piv) < 1e-300
                    free = findall(!, s.usedr)
                    if isempty(free); s.phase = :done; else; s.i = free[1]; end
                    continue
                end
                s.r = r; s.piv = piv; s.j = j; s.phase = :col
            end
        end
        # ---- columns
        colS = [s for s in states if s.phase == :col]
        if !isempty(colS)
            EI = Int32[]; EJ = Int32[]; off = Int[0]
            need = Bool[]
            for s in colS; jj = s.J[s.j]; for i in s.I; push!(EI, i); push!(EJ, jj); push!(need, s.hasnear); end; push!(off, length(EI)); end
            tdev += @elapsed vals = entries(EI, EJ, need)
            Threads.@threads for t in eachindex(colS)
                s = colS[t]; c = vals[off[t]+1:off[t+1]]
                for q in eachindex(s.U); c .-= conj(s.V[q][s.j]) .* s.U[q]; end
                v = conj.(s.r ./ s.piv)
                cross = 0.0 + 0im
                for q in eachindex(s.U); cross += dot(s.U[q], c) * dot(v, s.V[q]); end
                nu2 = sum(abs2, c) * sum(abs2, v)
                s.norm2 += nu2 + 2real(cross)
                push!(s.U, c); push!(s.V, v); s.steps += 1
                if sqrt(nu2) <= tol * sqrt(max(s.norm2, eps())) || s.steps >= min(length(s.I), length(s.J))
                    s.phase = :done; continue
                end
                a = abs.(c); a[s.usedr] .= -1; i = argmax(a)
                if a[i] < 0; s.phase = :done; else; s.i = i; s.phase = :row; end
            end
        end
    end
    return rounds, tdev
end

"""Full device build: dense near-field blocks and batched ACA rows/columns on the device; bookkeeping, near entries
(Float64) and recompression on the CPU."""
function hbuild_gpu(o, g::KAGeom, todev, root::HCluster, adm, tol::Float64, n::Int)
    adm_pairs = Tuple{HCluster,HCluster}[]; dense_pairs = Tuple{HCluster,HCluster}[]
    rec(a, b) = adm(a, b) ? push!(adm_pairs, (a, b)) :
        (isempty(a.kids) || isempty(b.kids)) ? push!(dense_pairs, (a, b)) :
        foreach(((ca, cb),) -> rec(ca, cb), Iterators.product(a.kids, b.kids))
    rec(root, root)
    tn = @elapsed begin
        NI, NJ, NV = findnz(o.corr)
        g64 = ka_geom(o, Float64)
        r64, i64 = ka_reg_entries(g64, Int32.(NI), Int32.(NJ))
        nearEx = sparse(NI, NJ, complex.(r64, i64) .+ NV, n, n); g64 = nothing
    end
    # dense blocks
    EI = Int32[]; EJ = Int32[]
    for (a, b) in dense_pairs, j in b.idx, i in a.idx; push!(EI, i); push!(EJ, j); end
    td = @elapsed begin re, im_ = ka_reg_entries(g, todev(EI), todev(EJ)); re = Array(re); im_ = Array(im_); end
    offs = cumsum([0; [length(a.idx) * length(b.idx) for (a, b) in dense_pairs]])
    dn = Vector{DenseBlk{ComplexF64}}(undef, length(dense_pairs))
    Threads.@threads for t in eachindex(dense_pairs)
        a, b = dense_pairs[t]; I = a.idx; J = b.idx; r = offs[t] .+ (1:length(I) * length(J))
        B = reshape(complex.(Float64.(re[r]), Float64.(im_[r])), length(I), length(J))
        S = nearEx[I, J]; rv = rowvals(S); nz = nonzeros(S)
        for c in 1:size(S, 2), q in nzrange(S, c); B[rv[q], c] = nz[q]; end
        B .*= o.Wrow[I]; B .+= 0.5 .* Matrix(o.Ipp[I, J])
        dn[t] = DenseBlk(I, J, B)
    end
    re = nothing; im_ = nothing; EI = nothing; EJ = nothing
    # batched ACA
    P = (nearEx .!= 0) .| (o.Ipp .!= 0)
    states = Vector{ACAState}(undef, length(adm_pairs))
    Threads.@threads for t in eachindex(adm_pairs)
        a, b = adm_pairs[t]; states[t] = ACAState(a.idx, b.idx, nnz(P[a.idx, b.idx]) > 0)
    end
    nnear = count(s -> s.hasnear, states)
    nb = LinearAlgebra.BLAS.get_num_threads(); LinearAlgebra.BLAS.set_num_threads(1)
    ta = @elapsed (rounds, tdev) = aca_batched(states, (EI, EJ, need) -> _device_entries(g, todev, o, nearEx, EI, EJ, need), tol / 10)
    lr = Vector{LowRank{ComplexF64}}(undef, length(states))
    tr = @elapsed Threads.@threads for t in eachindex(states)
        s = states[t]; k = length(s.U)
        U = k == 0 ? zeros(ComplexF64, length(s.I), 0) : reduce(hcat, s.U)
        V = k == 0 ? zeros(ComplexF64, length(s.J), 0) : reduce(hcat, s.V)
        U, V = recompress(U, V, tol)
        lr[t] = LowRank(s.I, s.J, U, V)
    end
    LinearAlgebra.BLAS.set_num_threads(nb)
    @printf("   admissible blocks needing exact near fixup: %d of %d\n", nnear, length(states))
    @printf("   gpu build: near exact %.1fs, dense %.1fs, ACA %.1fs (%d rounds, device %.1fs), recompress %.1fs\n", tn, td, ta, rounds, tdev, tr)
    return HMat{ComplexF64}(n, n, lr, dn), td + tn, ta + tr
end
