# Coupled FEM/BEM(+transducer) solve with the BEM blocks applied as H-matrices (A and -C), compared with the dense solve.
# Uses the engine's test fixture (femvolume + exterior_conforming). Backend: ENV HMAT_GPU = cpu|cpu32|metal|cuda.
include(joinpath(pwd(), "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(pwd(), "src", "BeatEngineCoupled.jl"))
include(joinpath(pwd(), "tests", "coupled_condensed_test_setup.jl"))
include(joinpath(pwd(), "src", "BeatEngineFmm.jl")); using .BeatEngineFmm
using Krylov, LinearMaps, Random, SparseArrays, Printf, StaticArrays, LinearAlgebra
const BC = BeatEngineCore
include(joinpath(@__DIR__, "..", "hmat_core.jl")); include(joinpath(@__DIR__, "..", "hmat_fast.jl"))
include(joinpath(@__DIR__, "..", "hmat_ka.jl")); include(joinpath(@__DIR__, "..", "hmat_c.jl"))
gpu = get(ENV, "HMAT_GPU", "cpu"); todev = identity; Tg = gpu == "cpu" ? Float64 : Float32
gpu == "metal" && (@eval using Metal; todev = MtlArray); gpu == "cuda" && (@eval using CUDA; todev = CuArray)
tol = parse(Float64, get(ENV, "HMAT_TOL", "1e-4"))
const ROOT = joinpath(pwd(), "tests", "fixtures")
fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), 0.001)
bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), 0.001)
imap = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
@printf("bem vertices %d faces %d, fem vertices %d\n", length(bem.vertices), length(bem.faces), length(fem.vertices))
with_baseline_cpu_coupled() do
  withenv("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "0") do
    f = parse(Float64, get(ENV, "F", "2000"))
    rtag = physical_tag(fem, 2, "Radiator")
    tr = ElectrodynamicTransducer{Float64}("component:test", [rtag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
        2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0)
    sys = build_condensed_coupled_system(fem, bem, imap, f, 343.0, 1.21; quadrature_order=CONDENSED_QUADRATURE_ORDER,
        singular_order=CONDENSED_SINGULAR_ORDER, transducers=[tr])
    Fd = sys.factorization; M = Fd.P' * (Fd.L * Fd.U)
    G, B, Q, Mr, Er = sys.gamma_range, sys.bem_range, sys.flux_range, sys.mechanical_range, sys.electrical_range
    io = sys.cache.base.interface_operators
    k = 2pi * f / 343; omega = 2pi * f
    tops = assemble_transducer_operators(fem, bem, [tr])
    motion_flux = sparse(neumann_scale(1.21, omega) .* Matrix{ComplexF64}(tops.bem_normal_velocity))
    p1 = build_p1_space(bem); dp0 = build_dp0_space(bem); rule = triangle_rule(Float64, CONDENSED_QUADRATURE_ORDER)
    op = fmm_burton_miller_operator(bem, p1, dp0, k, rule; identity_p1_p1=sys.cache.base.identity_p1_p1,
        identity_p1_dp0=sys.cache.base.identity_p1_dp0, singular_order=CONDENSED_SINGULAR_ORDER,
        symmetry_mode=sys.symmetry_mode, eps=1e-10)
    rest = copy(M); rest[B, B] .= 0; rest[B, Q] .= 0; isempty(Mr) || (rest[B, Mr] .= 0)
    flux = sparse(ComplexF64.(io.bem_flux))
    n = size(M, 1); nB = length(B)
    # ---- H(A) and H(-C) (columns: faces carrying interface or motion flux)
    o = FastOracle(bem, p1, dp0, k, rule, op, sys.symmetry_mode)
    g = ka_geom(o, Tg, todev)
    X = reduce(hcat, bem.vertices); root = hcluster(X, collect(1:nB), 48)
    imgV = [[BC.reflect_point(t, v) for v in bem.vertices] for t in BC.symmetry_image_transforms(BC.normalized_symmetry_mode(sys.symmetry_mode))]
    function adm(a, b, eta)
        min(hdiam(a), hdiam(b)) <= eta * hdist(a, b) || return false
        for iv in imgV
            P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)), vec(maximum(P, dims=2)), HCluster[])
            min(hdiam(a), hdiam(b)) <= eta * hdist(a, bb) || return false
        end
        true
    end
    tA = @elapsed (HA, _, _) = hbuild_gpu(o, g, todev, root, (a, b) -> adm(a, b, 1.0), tol, nB)
    dofof = [el.dp0_dof for el in op.elements]
    used = falses(dp0.global_dof_count); used[unique(rowvals(flux))] .= true; used[unique(rowvals(motion_flux))] .= true
    cols = [fc for fc in eachindex(op.elements) if used[dofof[fc]]]
    colsdof = dofof[cols]
    XF = reduce(hcat, [vec(sum(o.pts[1][e], dims=2)) ./ size(o.pts[1][e], 2) for e in cols]); colroot = hcluster(XF, collect(1:length(cols)), 48)
    imgF = [[BC.reflect_point(t, SVector{3,Float64}(XF[:, j])) for j in axes(XF, 2)] for t in BC.symmetry_image_transforms(BC.normalized_symmetry_mode(sys.symmetry_mode))]
    hpad = maximum(sqrt(2 * e.area) for e in op.elements)
    function admc(a, b, eta)
        db = hdiam(b) + 2hpad
        min(hdiam(a), db) <= eta * hdist(a, HCluster(b.idx, b.lo .- hpad, b.hi .+ hpad, HCluster[])) || return false
        for iv in imgF
            P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)) .- hpad, vec(maximum(P, dims=2)) .+ hpad, HCluster[])
            min(hdiam(a), db) <= eta * hdist(a, bb) || return false
        end
        true
    end
    tC = @elapsed HC = hbuild_c_gpu(o, op, g, todev, root, colroot, cols, (a, b) -> admc(a, b, 1.0), tol)
    @printf("f %.0f Hz tol %.0e: n %d (Γ %d, BEM %d, q %d, mech %d, elec %d); H(A) %.2f of dense, H(C) %d cols %.2f of dense; build A %.1fs C %.1fs\n",
        f, tol, n, length(G), nB, length(Q), length(Mr), length(Er), hstorage(HA) / nB^2, length(cols), hstorage(HC) / (nB * length(cols)), tA, tC)
    function hmul_coupled(x)
        y = rest * x
        data = flux * x[Q]; isempty(Mr) || (data .+= motion_flux * x[Mr])
        y[B] .+= hmul(HA, x[B]) .- hmul(HC, data[colsdof])
        return y
    end
    xr = randn(MersenneTwister(4), ComplexF64, n); yd = M * xr; yh = hmul_coupled(xr)
    for (name, r) in (("Γ", G), ("BEM", B), ("q", Q), ("mech", Mr), ("elec", Er))
        isempty(r) && continue
        @printf("   product %-5s rel err %.1e\n", name, norm(yh[r] - yd[r]) / max(norm(yd[r]), 1e-300))
    end
    # ---- solve with the existing bordered near preconditioner + row scaling (from FmmCoupledSolve), H products
    SOL = FmmCoupledSolve(op, rest, B, Q, Mr, flux, motion_flux; rtol=1e-9)
    D = Diagonal(SOL.row_scale)
    Am = LinearMap{ComplexF64}(v -> D * hmul_coupled(v), n, n)
    Pm = LinearMap{ComplexF64}(v -> SOL.precond \ (D \ v), n, n)
    # physical excitation: unit voltage on the transducer's electrical row (as the engine drives it)
    b = zeros(ComplexF64, n); isempty(Er) || (b[first(Er)] = 1.0)
    iszero(b) && (b = M * randn(MersenneTwister(2), ComplexF64, n))
    xs = M \ b
    t = @elapsed x, st = gmres(Am, D * b; N=Pm, ldiv=false, rtol=1e-9, atol=0.0, memory=300, itmax=300)
    @printf("   solve: %d its (%s) %.1fs; rel err vs dense: all %.1e", st.niter, st.solved, t, norm(x - xs) / norm(xs))
    for (name, r) in (("BEM p", B), ("q", Q), ("mech", Mr), ("elec", Er))
        isempty(r) && continue
        @printf(", %s %.1e", name, maximum(abs.(x[r] - xs[r])) / maximum(abs.(xs[r])))
    end
    println()
  end
end
