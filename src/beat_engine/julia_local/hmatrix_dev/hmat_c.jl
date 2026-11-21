# Entry-wise regular part of the Burton–Miller right-hand operator, written as -C (same sign as fmm_rhs):
#   negC_reg[i, f] = sum_{e∋i (local a), sets s, q, p} W_e[q] phi_a(q) W_f[p] * ( -G(x,y) - eta * dG/dn_x )
#   x on base face e, y on (image) face f of set s, G = e^{ikr}/(4 pi r), dG/dn_x = G (ik - 1/r) n_x·(x-y)/r.
# Exact entry: (negC_reg + rhs_correction[i, dof(f)]) * Wrow[i] - 0.5 * eta * Ipd[i, dof(f)].
@kernel function _negc_entries!(out_re, out_im, @Const(EI), @Const(EF), g)
    m = @index(Global, Linear)
    T = eltype(out_re)
    i = EI[m]; f = EF[m]
    k = g.k; er = g.eta_re; ei = g.eta_im
    inv4pi = T(1) / (T(4) * T(pi))
    accr = zero(T); acci = zero(T)
    for pe in g.vf_ptr[i]:(g.vf_ptr[i + 1] - 1)
        e = g.vf_face[pe]; a = g.vf_loc[pe]
        n1 = g.nrm[1, e, 1]; n2 = g.nrm[2, e, 1]; n3 = g.nrm[3, e, 1]
        for s in 1:g.nsets, q in 1:g.nq
            x1 = g.pts[1, q, e, 1]; x2 = g.pts[2, q, e, 1]; x3 = g.pts[3, q, e, 1]
            wq = g.W[q, e] * g.basis[a, q]
            for p in 1:g.nq
                d1 = x1 - g.pts[1, p, f, s]; d2 = x2 - g.pts[2, p, f, s]; d3 = x3 - g.pts[3, p, f, s]
                r2 = d1 * d1 + d2 * d2 + d3 * d3
                if r2 > zero(T)
                    r = sqrt(r2); ir = one(T) / r
                    sn, cs = sincos(k * r)
                    gr = cs * inv4pi * ir; gi = sn * inv4pi * ir
                    dn = (n1 * d1 + n2 * d2 + n3 * d3) * ir
                    dgr = (gr * (-ir) - gi * k) * dn; dgi = (gr * k + gi * (-ir)) * dn   # G (ik - 1/r) dn
                    w = wq * g.W[p, f]
                    accr += w * (-gr - (er * dgr - ei * dgi)); acci += w * (-gi - (er * dgi + ei * dgr))
                end
            end
        end
    end
    out_re[m] = accr; out_im[m] = acci
end

function ka_negc_entries(g::KAGeom{T}, EI, EF) where {T}
    be = get_backend(g.pts); n = length(EI)
    out_re = KernelAbstractions.zeros(be, T, n); out_im = KernelAbstractions.zeros(be, T, n)
    _negc_entries!(be, 256)(out_re, out_im, EI, EF, g; ndrange = n)
    KernelAbstractions.synchronize(be)
    return out_re, out_im
end

"""Exact -C entries for (P1 row i, face f) lists: device regular part + exact near (Float64) + row weights - 0.5 eta Ipd."""
function _negc_device_entries(g, todev, op, nearC, dofof, EI::Vector{Int32}, EF::Vector{Int32}, need)
    isempty(EI) && return ComplexF64[]
    re, im_ = ka_negc_entries(g, todev(EI), todev(EF))
    vals = complex.(Float64.(Array(re)), Float64.(Array(im_)))
    Ipd = op.identity_p1_dp0; eta = op.coupling; W = op.row_weights
    @inbounds for m in eachindex(vals)
        i = EI[m]; d = dofof[EF[m]]
        if need !== nothing && !need[m]; vals[m] *= W[i]; continue; end
        rng = nzrange(nearC, d); rv = rowvals(nearC); kk = searchsortedfirst(view(rv, rng), i)
        kk <= length(rng) && rv[rng[kk]] == i && (vals[m] = nonzeros(nearC)[rng[kk]])
        vals[m] *= W[i]
        rngI = nzrange(Ipd, d); rvI = rowvals(Ipd); kI = searchsortedfirst(view(rvI, rngI), i)
        kI <= length(rngI) && rvI[rngI[kI]] == i && (vals[m] -= 0.5 * eta * nonzeros(Ipd)[rngI[kI]])
    end
    return vals
end

"""H-matrix of -C restricted to the given columns (faces `cols`), rows = all P1 dofs. Column index of the H-matrix =
position in `cols`. rowroot: P1 vertex tree; colroot: tree over positions 1:length(cols) (face centroids)."""
function hbuild_c_gpu(o, op, g::KAGeom, todev, rowroot::HCluster, colroot::HCluster, cols::Vector{Int}, adm, tol::Float64)
    n = length(o.Wrow); m = length(cols)
    dofof = [el.dp0_dof for el in op.elements]
    adm_pairs = Tuple{HCluster,HCluster}[]; dense_pairs = Tuple{HCluster,HCluster}[]
    rec(a, b) = adm(a, b) ? push!(adm_pairs, (a, b)) :
        (isempty(a.kids) || isempty(b.kids)) ? push!(dense_pairs, (a, b)) :
        foreach(((ca, cb),) -> rec(ca, cb), Iterators.product(a.kids, b.kids))
    rec(rowroot, colroot)
    # exact near entries (Float64): nearC[i, dof] = reg64 + rhs_correction
    tn = @elapsed begin
        NI, ND, NV = findnz(op.rhs_correction)
        face_of_dof = zeros(Int, maximum(dofof)); for (f, d) in enumerate(dofof); face_of_dof[d] = f; end
        g64 = ka_geom(o, Float64)
        r64, i64 = ka_negc_entries(g64, Int32.(NI), Int32.(face_of_dof[ND]))
        nearC = sparse(NI, ND, complex.(r64, i64) .+ NV, n, size(op.rhs_correction, 2)); g64 = nothing
    end
    entries(EI, EJpos, need) = _negc_device_entries(g, todev, op, nearC, dofof, EI, Int32.(cols[EJpos]), need)
    # flags
    P = (nearC .!= 0) .| (op.identity_p1_dp0 .!= 0)
    colsdof = dofof[cols]
    hasnear(I, J) = nnz(P[I, colsdof[J]]) > 0
    # dense blocks
    td = @elapsed begin
        EI = Int32[]; EJ = Int32[]; need = Bool[]
        for (a, b) in dense_pairs, j in b.idx, i in a.idx; push!(EI, i); push!(EJ, j); push!(need, true); end
        vals = entries(EI, EJ, need)
    end
    offs = cumsum([0; [length(a.idx) * length(b.idx) for (a, b) in dense_pairs]])
    dn = [DenseBlk(a.idx, b.idx, reshape(vals[offs[t]+1:offs[t+1]], length(a.idx), length(b.idx))) for (t, (a, b)) in enumerate(dense_pairs)]
    vals = nothing
    states = Vector{ACAState}(undef, length(adm_pairs))
    Threads.@threads for t in eachindex(adm_pairs); a, b = adm_pairs[t]; states[t] = ACAState(a.idx, b.idx, hasnear(a.idx, b.idx)); end
    nb = LinearAlgebra.BLAS.get_num_threads(); LinearAlgebra.BLAS.set_num_threads(1)
    ta = @elapsed (rounds, tdev) = aca_batched(states, entries, tol / 10)
    lr = Vector{LowRank{ComplexF64}}(undef, length(states))
    tr = @elapsed Threads.@threads for t in eachindex(states)
        s = states[t]; kk = length(s.U)
        U = kk == 0 ? zeros(ComplexF64, length(s.I), 0) : reduce(hcat, s.U)
        V = kk == 0 ? zeros(ComplexF64, length(s.J), 0) : reduce(hcat, s.V)
        U, V = recompress(U, V, tol); lr[t] = LowRank(s.I, s.J, U, V)
    end
    LinearAlgebra.BLAS.set_num_threads(nb)
    @printf("   C build: near exact %.1fs, dense %.1fs, ACA %.1fs (%d rounds, device %.1fs), recompress %.1fs\n", tn, td, ta, rounds, tdev, tr)
    return HMat{ComplexF64}(n, m, lr, dn)
end
