# Fast exact-entry oracle: the regular-rule part of A on (I, J) as a direct point sum, with the same
# quadrature, images and kernel split as the dense build (and the FMM operator):
#   A_reg[i,j] = sum_{e∋i,f∋j,img} sum_{q,p} Wq Wp [ phi_i phi_j (-dG/dn_y - eta k^2 n_e.n_f G) + eta curl_i.curl_j G ]
struct FastOracle
    k::Float64; eta::ComplexF64
    # per element (base) and per image set: points (3 x nq), weights (nq), normal, curls (3 local), dofs
    pts::Vector{Vector{Matrix{Float64}}}     # [set][elem] 3 x nq
    W::Vector{Vector{Float64}}               # [elem] weights (same for images)
    nrm::Vector{Vector{SVector{3,Float64}}}  # [set][elem]
    curl::Vector{Vector{NTuple{3,SVector{3,Float64}}}}
    dofs::Vector{NTuple{3,Int}}
    basis::Vector{SVector{3,Float64}}        # P1 values at rule points
    vfaces::Vector{Vector{Int}}
    Wrow::Vector{Float64}
    corr::SparseMatrixCSC{ComplexF64,Int}
    Ipp::SparseMatrixCSC{Float64,Int}
end
function FastOracle(mesh, p1, dp0, k, rule, op, symmetry)
    el = BC._beat_cpu_element_data(mesh, p1, dp0)
    qd = BC._beat_cpu_regular_quadrature_data(mesh, rule)
    sets_el = [el]; sets_q = [qd]
    for t in BC.symmetry_image_transforms(BC.normalized_symmetry_mode(symmetry))
        push!(sets_el, BC._beat_cpu_reflect_element_data(el, t)); push!(sets_q, BC._beat_cpu_reflect_regular_quadrature_data(qd, t))
    end
    nq = length(rule.weights)
    pts = [[reduce(hcat, [Vector{Float64}(q.points[i]) for i in 1:nq]) for q in qs] for qs in sets_q]
    W = [[rule.weights[i] * 2 * e.area for i in 1:nq] for e in el]
    nrm = [[SVector{3,Float64}(e.normal) for e in es] for es in sets_el]
    curl = [[ntuple(a -> SVector{3,Float64}(e.curls[a]), 3) for e in es] for es in sets_el]
    vf = [Int[] for _ in mesh.vertices]
    for (e, face) in enumerate(mesh.faces), v in face; push!(vf[v], e); end
    FastOracle(op.k, op.coupling, pts, W, nrm, curl, [e.p1_dofs for e in el], [SVector{3,Float64}(BC.p1_values(p)) for p in rule.points],
        vf, Float64.(op.row_weights), op.correction, SparseMatrixCSC{Float64,Int}(op.identity_p1_p1))
end
function _faces(o, I)
    s = Int[]; for i in I; append!(s, o.vfaces[i]); end; unique!(sort!(s))
end
"""A_reg[I, J] (dense |I| x |J|)."""
function reg_block(o::FastOracle, I::Vector{Int}, J::Vector{Int})
    B = zeros(ComplexF64, length(I), length(J))
    pI = Dict(v => a for (a, v) in enumerate(I)); pj = Dict(v => a for (a, v) in enumerate(J))
    FI = _faces(o, I); FJ = _faces(o, J)
    k = o.k; eta = o.eta; k2 = k * k; nq = length(o.basis)
    inv4pi = 1 / (4pi)
    for e in FI
        de = o.dofs[e]; ri = (get(pI, de[1], 0), get(pI, de[2], 0), get(pI, de[3], 0))
        Pe = o.pts[1][e]; ne = o.nrm[1][e]; ce = o.curl[1][e]; We = o.W[e]
        for s in eachindex(o.pts), f in FJ
            df = o.dofs[f]; cj = (get(pj, df[1], 0), get(pj, df[2], 0), get(pj, df[3], 0))
            Pf = o.pts[s][f]; nf = o.nrm[s][f]; cf = o.curl[s][f]; Wf = o.W[f]
            nn = dot(ne, nf)
            acc_bb = zero(MMatrix{3,3,ComplexF64}); accG = 0.0im
            for q in 1:nq, p in 1:nq
                d1 = Pf[1,p]-Pe[1,q]; d2 = Pf[2,p]-Pe[2,q]; d3 = Pf[3,p]-Pe[3,q]
                r = sqrt(d1*d1+d2*d2+d3*d3); r == 0 && continue
                g = cis(k*r) * (inv4pi / r)
                dGdny = g * (im*k - 1/r) * (d1*nf[1] + d2*nf[2] + d3*nf[3]) / r
                w = We[q] * Wf[p]
                val = w * (-dGdny - eta * k2 * nn * g)
                bq = o.basis[q]; bp = o.basis[p]
                for a in 1:3, b in 1:3
                    acc_bb[a, b] += bq[a] * bp[b] * val
                end
                accG += w * g
            end
            for a in 1:3
                ri[a] == 0 && continue
                for b in 1:3
                    cj[b] == 0 && continue
                    B[ri[a], cj[b]] += acc_bb[a, b] + eta * dot(ce[a], cf[b]) * accG
                end
            end
        end
    end
    return B
end
function exact_block(o::FastOracle, I::Vector{Int}, J::Vector{Int})
    B = reg_block(o, I, J) .+ Matrix(o.corr[I, J])
    B .*= o.Wrow[I]
    B .+= 0.5 .* Matrix(o.Ipp[I, J])
    return B
end

# ---- faster variant: cached face lists, scratch index maps, no Dicts ----
function reg_block!(B::AbstractMatrix{ComplexF64}, o::FastOracle, FI::Vector{Int}, FJ::Vector{Int},
        mapI::Vector{Int}, mapJ::Vector{Int})
    k = o.k; eta = o.eta; k2 = k * k; nq = length(o.basis); inv4pi = 1 / (4pi)
    @inbounds for e in FI
        de = o.dofs[e]; ri = (mapI[de[1]], mapI[de[2]], mapI[de[3]])
        Pe = o.pts[1][e]; ne = o.nrm[1][e]; ce = o.curl[1][e]; We = o.W[e]
        for s in eachindex(o.pts), f in FJ
            df = o.dofs[f]; cj = (mapJ[df[1]], mapJ[df[2]], mapJ[df[3]])
            Pf = o.pts[s][f]; nf = o.nrm[s][f]; cf = o.curl[s][f]; Wf = o.W[f]
            nn = ne[1]*nf[1] + ne[2]*nf[2] + ne[3]*nf[3]
            a11=a12=a13=a21=a22=a23=a31=a32=a33=0.0im; accG = 0.0im
            for q in 1:nq
                bq = o.basis[q]; xq1 = Pe[1,q]; xq2 = Pe[2,q]; xq3 = Pe[3,q]
                for p in 1:nq
                    d1 = Pf[1,p]-xq1; d2 = Pf[2,p]-xq2; d3 = Pf[3,p]-xq3
                    r = sqrt(d1*d1+d2*d2+d3*d3); r == 0 && continue
                    g = cis(k*r) * (inv4pi / r)
                    dG = g * (im*k - 1/r) * (d1*nf[1] + d2*nf[2] + d3*nf[3]) / r
                    w = We[q] * Wf[p]
                    val = w * (-dG - eta * k2 * nn * g)
                    bp = o.basis[p]
                    v1 = bq[1]*val; v2 = bq[2]*val; v3 = bq[3]*val
                    a11 += v1*bp[1]; a12 += v1*bp[2]; a13 += v1*bp[3]
                    a21 += v2*bp[1]; a22 += v2*bp[2]; a23 += v2*bp[3]
                    a31 += v3*bp[1]; a32 += v3*bp[2]; a33 += v3*bp[3]
                    accG += w * g
                end
            end
            acc = ((a11,a12,a13),(a21,a22,a23),(a31,a32,a33))
            for a in 1:3
                ri[a] == 0 && continue
                for b in 1:3
                    cj[b] == 0 && continue
                    B[ri[a], cj[b]] += acc[a][b] + eta * dot(ce[a], cf[b]) * accG
                end
            end
        end
    end
    return B
end

struct ClusterCache
    faces::Vector{Int}
end

function exact_block_fast(o::FastOracle, I::Vector{Int}, FI::Vector{Int}, J::Vector{Int}, FJ::Vector{Int},
        mapI::Vector{Int}, mapJ::Vector{Int})
    for (a, v) in enumerate(I); mapI[v] = a; end
    for (a, v) in enumerate(J); mapJ[v] = a; end
    B = zeros(ComplexF64, length(I), length(J))
    reg_block!(B, o, FI, FJ, mapI, mapJ)
    for v in I; mapI[v] = 0; end
    for v in J; mapJ[v] = 0; end
    B .+= Matrix(o.corr[I, J])
    B .*= o.Wrow[I]
    B .+= 0.5 .* Matrix(o.Ipp[I, J])
    return B
end

"""Threaded H-matrix build from the exact-entry oracle. Returns HMat."""
function hbuild_threaded(o::FastOracle, root::HCluster, adm, tol::Float64, n::Int)
    adm_pairs = Tuple{HCluster,HCluster}[]; dense_pairs = Tuple{HCluster,HCluster}[]
    function rec(a, b)
        if adm(a, b)
            push!(adm_pairs, (a, b))
        elseif isempty(a.kids) || isempty(b.kids)
            push!(dense_pairs, (a, b))
        else
            for ca in a.kids, cb in b.kids; rec(ca, cb); end
        end
    end
    rec(root, root)
    fcache = IdDict{Vector{Int},Vector{Int}}()
    for (a, b) in Iterators.flatten((adm_pairs, dense_pairs))
        haskey(fcache, a.idx) || (fcache[a.idx] = _faces(o, a.idx))
        haskey(fcache, b.idx) || (fcache[b.idx] = _faces(o, b.idx))
    end
    corrT = sparse(transpose(o.corr)); IppT = sparse(transpose(o.Ipp))
    # dynamic load balancing: buffers come from a pool, not from threadid()
    pool = Channel{Tuple{Vector{Int},Vector{Int}}}(Threads.nthreads() + 1)
    for _ in 1:Threads.nthreads() + 1; put!(pool, (zeros(Int, n), zeros(Int, n))); end
    lr = Vector{LowRank{ComplexF64}}(undef, length(adm_pairs))
    dn = Vector{DenseBlk{ComplexF64}}(undef, length(dense_pairs))
    # largest work first
    work = vcat([(1, t, length(adm_pairs[t][1].idx) * length(adm_pairs[t][2].idx)) for t in eachindex(adm_pairs)],
                [(2, t, 4 * length(dense_pairs[t][1].idx) * length(dense_pairs[t][2].idx)) for t in eachindex(dense_pairs)])
    sort!(work; by = w -> -w[3])
    Threads.@threads :greedy for w in work
        bufs = take!(pool); mI, mJ = bufs
        try
            if w[1] == 1
                a, b = adm_pairs[w[2]]; I = a.idx; J = b.idx; FI = fcache[I]; FJ = fcache[J]
                U, V = aca(i -> exact_row(o, corrT, IppT, I[i], J, FJ, mI, mJ),
                           j -> exact_col(o, I, FI, J[j], mI, mJ),
                           length(I), length(J), tol / 10)
                U, V = recompress(U, V, tol)
                lr[w[2]] = LowRank(I, J, U, V)
            else
                a, b = dense_pairs[w[2]]
                dn[w[2]] = DenseBlk(a.idx, b.idx, exact_block_fast(o, a.idx, fcache[a.idx], b.idx, fcache[b.idx], mI, mJ))
            end
        finally
            put!(pool, bufs)
        end
    end
    return HMat{ComplexF64}(n, n, lr, dn)
end

# sparse add helpers using index maps (fast for single row / single column)
function _add_col_sparse!(B, S::SparseMatrixCSC, j::Int, col::Int, mapI, scale)
    rv = rowvals(S); nz = nonzeros(S)
    @inbounds for p in nzrange(S, j)
        a = mapI[rv[p]]; a == 0 && continue
        B[a, col] += scale * nz[p]
    end
end
"""Exact single row (i, J) of A as vector; uses transposed sparse parts."""
function exact_row(o::FastOracle, corrT, IppT, i::Int, J::Vector{Int}, FJ::Vector{Int}, mapI, mapJ)
    mapI[i] = 1; for (a, v) in enumerate(J); mapJ[v] = a; end
    B = zeros(ComplexF64, 1, length(J))
    reg_block!(B, o, o.vfaces[i], FJ, mapI, mapJ)
    B .*= o.Wrow[i]
    rv = rowvals(corrT); nz = nonzeros(corrT)
    @inbounds for p in nzrange(corrT, i); b = mapJ[rv[p]]; b == 0 || (B[1, b] += o.Wrow[i] * nz[p]); end
    rv = rowvals(IppT); nz2 = nonzeros(IppT)
    @inbounds for p in nzrange(IppT, i); b = mapJ[rv[p]]; b == 0 || (B[1, b] += 0.5 * nz2[p]); end
    mapI[i] = 0; for v in J; mapJ[v] = 0; end
    return vec(B)
end
function exact_col(o::FastOracle, I::Vector{Int}, FI::Vector{Int}, j::Int, mapI, mapJ)
    for (a, v) in enumerate(I); mapI[v] = a; end; mapJ[j] = 1
    B = zeros(ComplexF64, length(I), 1)
    reg_block!(B, o, FI, o.vfaces[j], mapI, mapJ)
    _add_col_sparse!(B, o.corr, j, 1, mapI, 1.0)
    B .*= o.Wrow[I]
    _add_col_sparse!(B, o.Ipp, j, 1, mapI, 0.5)
    for v in I; mapI[v] = 0; end; mapJ[j] = 0
    return vec(B)
end

"""fmm_rhs with a direct sum over nonzero-charge sources (q is sparse: driver faces only)."""
function rhs_direct(op, q::AbstractMatrix)
    nq = op.nq; nd = size(q, 2)
    srcs = Int[]; chg = ComplexF64[]
    for (s_el, el) in enumerate(op.source_elements)
        jac = 2 * el.area
        for qq in 1:nq
            c = op.weights[qq] * jac * q[el.dp0_dof, 1]
            c == 0 && continue
            push!(srcs, (s_el - 1) * nq + qq); push!(chg, c)
        end
    end
    S = op.sources[:, srcs]; T = op.targets; nt = size(T, 2); k = op.k
    pot = zeros(ComplexF64, nt); grad = zeros(ComplexF64, 3, nt)
    Threads.@threads for t in 1:nt
        p = 0.0im; g1 = g2 = g3 = 0.0im
        @inbounds for s in eachindex(chg)
            d1 = T[1,t]-S[1,s]; d2 = T[2,t]-S[2,s]; d3 = T[3,t]-S[3,s]
            r = sqrt(d1*d1+d2*d2+d3*d3); r == 0 && continue
            e = chg[s] * cis(k*r) / r; p += e
            f = e * (im*k - 1/r) / r
            g1 += f*d1; g2 += f*d2; g3 += f*d3
        end
        pot[t] = p; grad[1,t] = g1; grad[2,t] = g2; grad[3,t] = g3
    end
    pot ./= 4pi; grad ./= 4pi   # FMM3D kernel is e^{ikr}/(4pi r)
    pot ./= 4pi * op.charge_factor; grad ./= 4pi * op.charge_factor
    eta = op.coupling; out = zeros(ComplexF64, op.n_p1)
    for (e, el) in enumerate(op.elements)
        jac = 2 * el.area
        for qq in 1:nq
            col = (e - 1) * nq + qq; W = op.weights[qq] * jac; b = op.basis[qq]
            dn = el.normal[1]*grad[1,col] + el.normal[2]*grad[2,col] + el.normal[3]*grad[3,col]
            val = W * (-pot[col] - eta * dn)
            for a in 1:3; out[el.p1_dofs[a]] += b[a] * val; end
        end
    end
    out .+= (op.rhs_correction * q)[:, 1]
    out .*= op.row_weights
    out .-= (0.5 * eta) .* (op.identity_p1_dp0 * q)[:, 1]
    return out
end
