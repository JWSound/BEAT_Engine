include(joinpath(pwd(), "tests", "fmm_solve_probe.jl"))
include(joinpath(@__DIR__, "..", "hmat_core.jl"))
using SparseArrays, Serialization, Printf, StaticArrays, LinearAlgebra
const BC = BeatEngineCore
include(joinpath(@__DIR__, "..", "hmat_fast.jl"))
mesh = load_txt_mesh(ARGS[1]); f = parse(Float64, ARGS[2]); tol = parse(Float64, ARGS[3]); k = 2pi * f / 343
check = length(ARGS) > 3
p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh); rule = triangle_rule(Float64, 2)
Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:xy)
Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:xy)
ts = @elapsed op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=:xy, eps=1e-4)
o = FastOracle(mesh, p1, dp0, k, rule, op, :xy)
n = p1.global_dof_count; X = reduce(hcat, mesh.vertices)
imgV = [[BC.reflect_point(t, v) for v in mesh.vertices] for t in BC.symmetry_image_transforms(:xy)]
if check
    d = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, zeros(ComplexF64, dp0.global_dof_count, 1), k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=:xy, singular_order=2)
    A = d.matrix; d = nothing; GC.gc()
    rng = MersenneTwister(3)
    for t in 1:3
        I = sort(unique(rand(rng, 1:n, 30))); J = sort(unique(rand(rng, 1:n, 300)))
        E = exact_block(o, I, J)
        @printf("entry check %d: rel err %.1e\n", t, norm(E - A[I, J]) / norm(A[I, J]))
    end
    I = collect(1:40); J = collect(1:n)
    @printf("full rows check: rel err %.1e\n", norm(exact_block(o, I, J) - A[I, :]) / norm(A[I, :]))
    A = nothing; GC.gc()
end
q = reshape(ComplexF64.(mesh.physical_tags .== 4), :, 1)
b = rhs_direct(op, q)
if check; b0 = fmm_rhs(op, q)[:, 1]; @printf("rhs direct vs fmm: %.1e\n", norm(b - b0) / norm(b0)); end
root = hcluster(X, collect(1:n), 48)
function adm(a, b, eta)
    min(hdiam(a), hdiam(b)) <= eta * hdist(a, b) || return false
    for iv in imgV
        P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)), vec(maximum(P, dims=2)), HCluster[])
        min(hdiam(a), hdiam(b)) <= eta * hdist(a, bb) || return false
    end
    true
end
gpu = get(ENV, "HMAT_GPU", "")
if gpu == ""
    tb = @elapsed H = hbuild_threaded(o, root, (a, b) -> adm(a, b, 1.0), tol, n)
else
    include(joinpath(@__DIR__, "..", "hmat_ka.jl"))
    if gpu == "metal"; @eval using Metal; todev = MtlArray; Tg = Float32
    elseif gpu == "cuda"; @eval using CUDA; todev = CuArray; Tg = Float32
    elseif gpu == "cuda64"; @eval using CUDA; todev = CuArray; Tg = Float64
    else; todev = identity; Tg = gpu == "cpu32" ? Float32 : Float64; end
    gg = Base.invokelatest(ka_geom, o, Tg, todev)
    Base.invokelatest(ka_reg_entries, gg, todev(Int32[1, 2]), todev(Int32[1, 2]))   # compile
    bfun = get(ENV, "HMAT_ACA", "gpu") == "gpu" ? hbuild_gpu : hbuild_hybrid
    tb = @elapsed (H, tdev, taca) = Base.invokelatest(bfun, o, gg, todev, root, (a, b) -> adm(a, b, 1.0), tol, n)
    @printf("   hybrid build (%s %s): dense blocks on device %.1fs, ACA on CPU %.1fs\n", gpu, Tg, tdev, taca); flush(stdout)
    gg = nothing
end
st = hstorage(H)
@printf("mesh %s n %d f %.0f tol %.0e: storage %.3f of dense = %.2f GB complex64 (dense %.2f GB), max rank %d, %d low-rank / %d dense blocks, build %.0fs (near setup %.0fs)\n",
    basename(ARGS[1]), n, f, tol, st / n^2, st * 8 / 2^30, n^2 * 8 / 2^30, hmaxrank(H), length(H.lr), length(H.dense), tb, ts)
flush(stdout)
op = nothing; o = nothing; GC.gc()
Sblk = parse(Int, get(ENV, "HMAT_PREC_BLOCK", "1000"))
bj, ngrp = cluster_block_precond(H, root, Sblk)
@printf("   cluster block-Jacobi: S %d, %d blocks\n", Sblk, ngrp); flush(stdout)
Hm = LinearMap{ComplexF64}(v -> hmul(H, v), n, n); P = LinearMap{ComplexF64}(bj, n, n)
tg = @elapsed x, s = gmres(Hm, b; N=P, ldiv=false, rtol=1e-6, atol=0.0, memory=300, itmax=300)
@printf("   GMRES %d its (%s), %.0fs\n", s.niter, s.solved, tg)
ref = nothing
if check
    ref = deserialize("/tmp/ref5_$(Int(f)).jls")
    @printf("   vs dense: solution err %.1e\n", norm(x - ref.xs) / norm(ref.xs))
end
serialize(joinpath(tempdir(), "hsol_$(basename(ARGS[1]))_$(Int(f)).jls"), x)
# ---- optional dense reference solve (HMAT_DENSE=1): assemble the engine matrix, LU, compare ----
xplain16 = nothing
if get(ENV, "HMAT_DENSE", "0") == "1"
    H16 = hconvert_scaled(H, Float16); M16 = LinearMap{ComplexF64}(v -> hmul(H16, v), n, n)
    xplain16, _ = gmres(M16, b; N=P, ldiv=false, rtol=1e-6, atol=0.0, memory=300, itmax=400); H16 = nothing
    mesh2 = load_txt_mesh(ARGS[1]); p1b = build_p1_space(mesh2); dp0b = build_dp0_space(mesh2)
    Ippb = assemble_l2_identity_matrix(mesh2, p1b, dp0b, rule, :p1, :p1; symmetry_mode=:xy)
    Ipdb = assemble_l2_identity_matrix(mesh2, p1b, dp0b, rule, :p1, :dp0; symmetry_mode=:xy)
    ta = @elapsed d = assemble_burton_miller_neumann_system_cpu(mesh2, p1b, dp0b, zeros(ComplexF64, dp0b.global_dof_count, 1), k, rule;
        identity_p1_p1=Ippb, identity_p1_dp0=Ipdb, symmetry_mode=:xy, singular_order=2)
    A = d.matrix; d = nothing; GC.gc()
    tl = @elapsed xs = lu!(A) \ b
    A = nothing; GC.gc()
    @printf("   DENSE reference: assembly %.0fs, LU+solve %.0fs\n", ta, tl)
    @printf("   DENSE: H (c128) solution err %.1e, H FP16 solution err %.1e\n", norm(x - xs) / norm(xs), norm(xplain16 - xs) / norm(xs))
    flush(stdout)
end
# ---- low-precision storage + refinement ----
check && serialize(joinpath(tempdir(), "H_piece5.jls"), (H, b))
haskey(ENV, "HMAT_SAVE") && serialize(ENV["HMAT_SAVE"], (H, b))
let v = randn(ComplexF64, n), y = hmul(H, v)
    for (T, nm) in ((Float16, "FP16"), (:bf16, "BF16"))
        Hl = hconvert_scaled(H, T); @printf("   %s operator rel err %.1e (any NaN: %s)\n", nm, norm(hmul(Hl, v) - y) / norm(y), any(isnan, hmul(Hl, v)))
    end
end
for (T, name) in ((Float16, "FP16"), (:bf16, "BF16"))
    H16 = hconvert_scaled(H, T); M16 = LinearMap{ComplexF64}(v -> hmul(H16, v), n, n)
    x16, s16 = gmres(M16, b; N=P, ldiv=false, rtol=1e-6, atol=0.0, memory=300, itmax=400)
    @printf("   %s plain: %d its, diff vs c128 H solution %.1e\n", name, s16.niter, norm(x16 - x) / norm(x))
    xr = zeros(ComplexF64, n); tot = 0
    for it in 1:6
        r = b - hmul(H, xr); rn = norm(r) / norm(b)
        rn < 1e-7 && break
        d, sd = gmres(M16, r; N=P, ldiv=false, rtol=1e-3, atol=0.0, memory=300, itmax=400)
        xr .+= d; tot += sd.niter
    end
    @printf("   %s + refinement: %d inner its total, diff vs c128 H solution %.1e, resid %.1e\n", name, tot, norm(xr - x) / norm(x), norm(b - hmul(H, xr)) / norm(b))
    if check
        @printf("   %s + refinement vs dense: %.1e\n", name, norm(xr - ref.xs) / norm(ref.xs))
    end
end
