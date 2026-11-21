# GPU kernel test: all dense near-field leaf blocks of the H-matrix partition on the GPU (Float32),
# accuracy vs the CPU Float64 oracle on sample blocks, and throughput.
# usage (from the work dir with src/ and tests/):  julia ... hmat_ka_test.jl <mesh prefix> <f> <backend: cpu|metal|cuda>
include(joinpath(pwd(), "tests", "fmm_solve_probe.jl"))
include(joinpath(@__DIR__, "..", "hmat_core.jl"))
using SparseArrays, Printf, StaticArrays, LinearAlgebra, Random
const BC = BeatEngineCore
include(joinpath(@__DIR__, "..", "hmat_fast.jl")); include(joinpath(@__DIR__, "..", "hmat_ka.jl"))
backend = length(ARGS) > 2 ? ARGS[3] : "cpu"
todev = identity; T = Float64
if backend == "metal"
    using Metal; todev = MtlArray; T = Float32
elseif backend == "cuda"
    using CUDA; todev = CuArray; T = Float32
end
mesh = load_txt_mesh(ARGS[1]); f = parse(Float64, ARGS[2]); k = 2pi * f / 343
p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh); rule = triangle_rule(Float64, 2)
Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:xy)
Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:xy)
op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=:xy, eps=1e-4)
o = FastOracle(mesh, p1, dp0, k, rule, op, :xy); n = p1.global_dof_count
X = reduce(hcat, mesh.vertices); root = hcluster(X, collect(1:n), 48)
imgV = [[BC.reflect_point(t, v) for v in mesh.vertices] for t in BC.symmetry_image_transforms(:xy)]
function adm(a, b, eta)
    min(hdiam(a), hdiam(b)) <= eta * hdist(a, b) || return false
    for iv in imgV
        P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)), vec(maximum(P, dims=2)), HCluster[])
        min(hdiam(a), hdiam(b)) <= eta * hdist(a, bb) || return false
    end
    true
end
dense_pairs = Tuple{HCluster,HCluster}[]
rec(a, b) = adm(a, b, 1.0) ? nothing : (isempty(a.kids) || isempty(b.kids)) ? push!(dense_pairs, (a, b)) : foreach(((ca, cb),) -> rec(ca, cb), Iterators.product(a.kids, b.kids))
rec(root, root)
EI = Int32[]; EJ = Int32[]
for (a, b) in dense_pairs, j in b.idx, i in a.idx; push!(EI, i); push!(EJ, j); end
@printf("mesh %s n %d f %.0f: %d dense blocks, %d entries (%.2f of n^2), backend %s %s\n", basename(ARGS[1]), n, f, length(dense_pairs), length(EI), length(EI) / n^2, backend, T)
flush(stdout)
g = ka_geom(o, T, todev); dEI = todev(EI); dEJ = todev(EJ)
ka_reg_entries(g, dEI[1:min(end, 1024)], dEJ[1:min(end, 1024)])   # compile
t = @elapsed (re, im_) = ka_reg_entries(g, dEI, dEJ)
re = Array(re); im_ = Array(im_)
@printf("   kernel: %.2f s for all dense entries (%.1f ns/entry)\n", t, 1e9t / length(EI))
# accuracy on sample blocks vs the Float64 CPU oracle (regular part only)
rng = MersenneTwister(1); mI = zeros(Int, n); mJ = zeros(Int, n); worst = 0.0; off = cumsum([0; [length(a.idx) * length(b.idx) for (a, b) in dense_pairs]])
for t in rand(rng, 1:length(dense_pairs), 20)
    a, b = dense_pairs[t]; I = a.idx; J = b.idx
    for (x, v) in enumerate(I); mI[v] = x; end; for (x, v) in enumerate(J); mJ[v] = x; end
    Bref = zeros(ComplexF64, length(I), length(J)); reg_block!(Bref, o, _faces(o, I), _faces(o, J), mI, mJ)
    for v in I; mI[v] = 0; end; for v in J; mJ[v] = 0; end
    full = Bref .+ Matrix(o.corr[I, J])   # what the H block actually uses (before row weights)
    r = off[t] .+ (1:length(I) * length(J))
    B = reshape(complex.(Float64.(re[r]), Float64.(im_[r])), length(I), length(J))
    global worst = max(worst, norm(B .+ Matrix(o.corr[I, J]) .- full) / norm(full))
end
@printf("   accuracy (20 sample blocks, regular+correction): worst rel err %.1e\n", worst)
