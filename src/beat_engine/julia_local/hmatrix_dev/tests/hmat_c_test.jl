# H-matrix of the BM right-hand operator -C (all faces as columns): build on the device, check against direct/FMM products.
# usage: julia ... hmat_c_test.jl <mesh prefix> <f> <tol> <backend: cpu|cpu32|metal|cuda>
include(joinpath(pwd(), "tests", "fmm_solve_probe.jl"))
include(joinpath(@__DIR__, "..", "hmat_core.jl"))
using SparseArrays, Printf, StaticArrays, LinearAlgebra, Random
const BC = BeatEngineCore
include(joinpath(@__DIR__, "..", "hmat_fast.jl")); include(joinpath(@__DIR__, "..", "hmat_ka.jl")); include(joinpath(@__DIR__, "..", "hmat_c.jl"))
backend = ARGS[4]; todev = identity; T = backend in ("cpu", "cuda64") ? Float64 : Float32
if backend == "metal"; @eval using Metal; todev = MtlArray
elseif backend in ("cuda", "cuda64"); @eval using CUDA; todev = CuArray; end
mesh = load_txt_mesh(ARGS[1]); f = parse(Float64, ARGS[2]); tol = parse(Float64, ARGS[3]); k = 2pi * f / 343
p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh); rule = triangle_rule(Float64, 2)
Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:xy)
Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:xy)
op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=:xy, eps=1e-6)
o = FastOracle(mesh, p1, dp0, k, rule, op, :xy); n = p1.global_dof_count; nf = length(op.elements)
X = reduce(hcat, mesh.vertices); rowroot = hcluster(X, collect(1:n), 48)
cols = collect(1:nf)
XF = reduce(hcat, [vec(sum(o.pts[1][e], dims=2)) ./ size(o.pts[1][e], 2) for e in cols]); colroot = hcluster(XF, collect(1:nf), 48)
# face images for admissibility (centroids reflected; boxes padded by the face size via hdiam of leaf clusters is implicit)
trs = BC.symmetry_image_transforms(:xy)
imgF = [[BC.reflect_point(t, SVector{3,Float64}(XF[:, j])) for j in 1:nf] for t in trs]
hpad = maximum(sqrt(2 * e.area) for e in op.elements)
function admc(a, b, eta)
    da = hdiam(a); db = hdiam(b) + 2hpad
    bb0 = HCluster(b.idx, b.lo .- hpad, b.hi .+ hpad, HCluster[])
    min(da, db) <= eta * hdist(a, bb0) || return false
    for iv in imgF
        P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)) .- hpad, vec(maximum(P, dims=2)) .+ hpad, HCluster[])
        min(da, db) <= eta * hdist(a, bb) || return false
    end
    true
end
g = ka_geom(o, T, todev)
ka_negc_entries(g, todev(Int32[1, 2]), todev(Int32[1, 2]))
tb = @elapsed HC = hbuild_c_gpu(o, op, g, todev, rowroot, colroot, cols, (a, b) -> admc(a, b, 1.0), tol)
st = hstorage(HC)
@printf("mesh %s n %d faces %d f %.0f tol %.0e: C storage %.3f of dense = %.2f GB c64 (dense %.2f GB), max rank %d, build %.0fs\n",
    basename(ARGS[1]), n, nf, f, tol, st / (n * nf), st * 8 / 2^30, n * nf * 8 / 2^30, hmaxrank(HC), tb)
# check 1: driver excitation vs direct sum
q = ComplexF64.(mesh.physical_tags .== 4); dofof = [el.dp0_dof for el in op.elements]
b0 = rhs_direct(op, reshape(q, :, 1)); b1 = hmul(HC, q[dofof])
@printf("   driver RHS: H(-C) q vs direct: rel err %.1e\n", norm(b1 - b0) / norm(b0))
# check 2: random smooth flux on all faces vs FMM (small meshes only)
if n <= 12000
    rng = MersenneTwister(2); qr = cis.(2pi .* rand(rng, nf)) .* (1 .+ rand(rng, nf))
    qd = zeros(ComplexF64, dp0.global_dof_count, 1); qd[dofof, 1] .= qr
    bf = fmm_rhs(op, qd)[:, 1]; bh = hmul(HC, qr)
    idt = -(0.5 * op.coupling) .* (op.identity_p1_dp0 * qd)[:, 1]     # local identity term, removed from the comparison
    @printf("   random flux: H(-C) q vs FMM(eps 1e-6): rel err %.1e (without the local identity term %.1e)\n",
        norm(bh - bf) / norm(bf), norm((bh - idt) - (bf - idt)) / norm(bf - idt))
end
