# Minimal H-matrix: cluster tree, block tree, ACA from a row/column oracle, product, storage.
using LinearAlgebra

struct HCluster
    idx::Vector{Int}
    lo::Vector{Float64}
    hi::Vector{Float64}
    kids::Vector{HCluster}
end

function hcluster(X::AbstractMatrix, idx::Vector{Int}, leaf::Int)
    P = X[:, idx]; lo = vec(minimum(P, dims=2)); hi = vec(maximum(P, dims=2))
    length(idx) <= leaf && return HCluster(idx, lo, hi, HCluster[])
    ax = argmax(hi .- lo); mid = (lo[ax] + hi[ax]) / 2
    l = idx[X[ax, idx] .<= mid]; r = idx[X[ax, idx] .> mid]
    (isempty(l) || isempty(r)) && return HCluster(idx, lo, hi, HCluster[])
    HCluster(idx, lo, hi, [hcluster(X, l, leaf), hcluster(X, r, leaf)])
end
hdiam(c) = norm(c.hi .- c.lo)
hdist(a, b) = norm(max.(0, max.(a.lo .- b.hi, b.lo .- a.hi)))

struct LowRank{T}
    rows::Vector{Int}; cols::Vector{Int}
    U::Matrix{T}; V::Matrix{T}      # block ≈ U * V'   (V conj-transposed)
end
struct DenseBlk{T}
    rows::Vector{Int}; cols::Vector{Int}
    M::Matrix{T}
end
struct HMat{T}
    n::Int; m::Int
    lr::Vector{LowRank{T}}
    dense::Vector{DenseBlk{T}}
end

"""ACA with partial pivoting. row(i) -> entries of row I[i] over J, col(j) -> column J[j] over I."""
function aca(row, col, m::Int, n::Int, tol::Float64; maxrank::Int=min(m, n))
    U = Vector{Vector{ComplexF64}}(); V = Vector{Vector{ComplexF64}}()
    usedr = falses(m); i = 1; norm2 = 0.0
    for _ in 1:maxrank
        r = Vector{ComplexF64}(row(i))
        for t in eachindex(U); r .-= U[t][i] .* conj.(V[t]); end
        usedr[i] = true
        j = argmax(abs.(r)); piv = r[j]
        if abs(piv) < 1e-300
            free = findall(!, usedr); isempty(free) && break
            i = free[1]; continue
        end
        c = Vector{ComplexF64}(col(j))
        for t in eachindex(U); c .-= conj(V[t][j]) .* U[t]; end
        v = conj.(r ./ piv)
        # update ||S||^2 incrementally
        cross = 0.0 + 0im
        for t in eachindex(U); cross += dot(U[t], c) * dot(v, V[t]); end
        nu2 = sum(abs2, c) * sum(abs2, v)
        norm2 += nu2 + 2real(cross)
        push!(U, c); push!(V, v)
        sqrt(nu2) <= tol * sqrt(max(norm2, eps())) && break
        a = abs.(c); a[usedr] .= -1; i = argmax(a)
        a[i] < 0 && break
    end
    k = length(U)
    Um = k == 0 ? zeros(ComplexF64, m, 0) : reduce(hcat, U)
    Vm = k == 0 ? zeros(ComplexF64, n, 0) : reduce(hcat, V)
    return Um, Vm
end

"""Recompress U V' with QR + SVD to tolerance tol (relative to the block's largest singular value)."""
function recompress(U, V, tol)
    size(U, 2) == 0 && return U, V
    QU, RU = qr(U); QV, RV = qr(V)
    F = svd(RU * RV')
    s = F.S; k = count(>(tol * s[1]), s); k = max(k, 1)
    return Matrix(QU) * (F.U[:, 1:k] .* s[1:k]'), Matrix(QV) * F.V[:, 1:k]
end

function hassemble(rowfun, colfun, densefun, root::HCluster, eta::Float64, tol::Float64, n::Int)
    lr = LowRank{ComplexF64}[]; dn = DenseBlk{ComplexF64}[]
    function rec(a, b)
        if min(hdiam(a), hdiam(b)) <= eta * hdist(a, b)
            I = a.idx; J = b.idx
            U, V = aca(i -> rowfun(I[i], J), j -> colfun(I, J[j]), length(I), length(J), tol / 10)
            U, V = recompress(U, V, tol)
            push!(lr, LowRank(I, J, U, V))
        elseif isempty(a.kids) || isempty(b.kids)
            push!(dn, DenseBlk(a.idx, b.idx, densefun(a.idx, b.idx)))
        else
            for ca in a.kids, cb in b.kids; rec(ca, cb); end
        end
    end
    rec(root, root)
    return HMat{ComplexF64}(n, n, lr, dn)
end

function hmul(H::HMat, x::AbstractVector)
    y = zeros(ComplexF64, H.n)
    for b in H.dense
        y[b.rows] .+= b.M * x[b.cols]
    end
    for b in H.lr
        y[b.rows] .+= b.U * (b.V' * x[b.cols])
    end
    return y
end

hstorage(H::HMat) = sum(length(b.M) for b in H.dense; init=0) + sum(length(b.U) + length(b.V) for b in H.lr; init=0)
hmaxrank(H::HMat) = maximum(size(b.U, 2) for b in H.lr; init=0)

"""Convert all blocks to a lower precision type (storage experiment)."""
function hconvert(H::HMat, ::Type{T}) where {T}
    conv(M) = ComplexF64.(Complex{T}.(M))
    HMat{ComplexF64}(H.n, H.m, [LowRank(b.rows, b.cols, conv(b.U), conv(b.V)) for b in H.lr],
        [DenseBlk(b.rows, b.cols, conv(b.M)) for b in H.dense])
end

"""Simulated low-precision storage with per-block scaling (and per-column balancing of U,V)."""
function bf16r(x::Float64)
    u = reinterpret(UInt32, Float32(x)); u += 0x00007fff + ((u >> 16) & 0x00000001); Float64(reinterpret(Float32, u & 0xffff0000))
end
rnd(::Type{Float16}, z) = ComplexF64(Float16(real(z)), Float16(imag(z)))
rndb(z) = ComplexF64(bf16r(real(z)), bf16r(imag(z)))
function hconvert_scaled(H::HMat, T)
    function conv(M)   # per-row scaling (scale kept in Float32)
        s = Float64.(Float32.(vec(maximum(abs, M; dims=2)))); s[s .== 0] .= 1
        T === :bf16 ? rndb.(M ./ s) .* s : rnd.(Float16, M ./ s) .* s
    end
    lr = map(H.lr) do b
        U = copy(b.U); V = copy(b.V)
        for c in axes(U, 2)
            nu = norm(U[:, c]); nv = norm(V[:, c]); (nu == 0 || nv == 0) && continue
            f = sqrt(nv / nu); U[:, c] .*= f; V[:, c] ./= f
        end
        LowRank(b.rows, b.cols, conv(U), conv(V))
    end
    HMat{ComplexF64}(H.n, H.m, lr, [DenseBlk(b.rows, b.cols, conv(b.M)) for b in H.dense])
end

"""Block-Jacobi preconditioner on tree clusters of size <= S, blocks assembled from all H blocks inside each cluster."""
function cluster_block_precond(H::HMat, root::HCluster, S::Int)
    G = Vector{Int}[]
    grp(c) = (length(c.idx) <= S || isempty(c.kids)) ? push!(G, c.idx) : foreach(grp, c.kids)
    grp(root); n = H.n
    gid = zeros(Int, n); pos = zeros(Int, n)
    for (g, I) in enumerate(G); gid[I] .= g; pos[I] .= 1:length(I); end
    B = [zeros(ComplexF64, length(I), length(I)) for I in G]
    inside(r, c) = (g = gid[r[1]]; all(gid[r] .== g) && all(gid[c] .== g)) ? g : 0
    for d in H.dense; g = inside(d.rows, d.cols); g > 0 && (B[g][pos[d.rows], pos[d.cols]] .+= d.M); end
    for l in H.lr; g = inside(l.rows, l.cols); g > 0 && (B[g][pos[l.rows], pos[l.cols]] .+= l.U * l.V'); end
    F = Vector{Any}(undef, length(G))
    Threads.@threads for g in eachindex(G); F[g] = lu!(B[g]); end
    apply(v) = (y = similar(v); for (g, I) in enumerate(G); y[I] = F[g] \ v[I]; end; y)
    return apply, length(G)
end
