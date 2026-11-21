include("" * joinpath(@__DIR__, "..") * "/src/BeatEngineCore.jl"); using .BeatEngineCore
include("" * joinpath(@__DIR__, "..") * "/src/BeatEngineCoupled.jl")
include("" * joinpath(@__DIR__, "..") * "/tests/coupled_condensed_test_setup.jl")
include("" * joinpath(@__DIR__, "..") * "/src/BeatEngineFmm.jl")
using .BeatEngineFmm
using Krylov, LinearMaps, Random
const ROOT = "" * joinpath(@__DIR__, "..") * "/tests/fixtures"
fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), 0.001)
bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), 0.001)
imap = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
println("bem vertices $(length(bem.vertices)) faces $(length(bem.faces))  fem vertices $(length(fem.vertices))")
with_baseline_cpu_coupled() do
    f = parse(Float64, get(ENV, "F", "500"))
    sys = build_condensed_coupled_system(fem, bem, imap, f, 343.0, 1.21; quadrature_order=CONDENSED_QUADRATURE_ORDER,
        singular_order=CONDENSED_SINGULAR_ORDER)
    Fd = sys.factorization
    println("factorization type $(typeof(Fd))  layout $(sys.interface_elimination)  ranges Γ $(sys.gamma_range) B $(sys.bem_range) q $(sys.flux_range)")
    M = Fd isa LinearAlgebra.LU ? (Fd.P' * (Fd.L * Fd.U)) : hasproperty(Fd, :matrix) ? Fd.matrix : error("no matrix")
    io = sys.cache.base.interface_operators
    gv = sys.gamma_fem_vertices
    k = 2pi * f / 343
    p1 = build_p1_space(bem); dp0 = build_dp0_space(bem)
    rule = triangle_rule(Float64, CONDENSED_QUADRATURE_ORDER)
    Ipp = sys.cache.base.identity_p1_p1; Ipd = sys.cache.base.identity_p1_dp0
    op = fmm_burton_miller_operator(bem, p1, dp0, k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd,
        singular_order=CONDENSED_SINGULAR_ORDER, symmetry_mode=sys.symmetry_mode, eps=1e-10)
    G, B, Q = sys.gamma_range, sys.bem_range, sys.flux_range
    n = size(M, 1)
    flux = Matrix{ComplexF64}(io.bem_flux)
    function coupled_mul(x)
        y = zeros(ComplexF64, n)
        pG = x[G]; pB = x[B]; q = x[Q]
        y[G] .= sys.condensation.schur * pG .- io.fem_load[gv, :] * q
        # bem_interface_block = -(C * bem_flux): fmm_rhs gives (-S - eta(K'+0.5I)) q' = -C q'
        y[B] .= fmm_lhs_mul(op, pB) .- fmm_rhs(op, reshape(flux * q, :, 1))[:, 1]
        y[Q] .= io.fem_trace[:, gv] * pG .- io.bem_trace * pB
        return y
    end
    x = randn(ComplexF64, n)
    yd = M * x; yf = coupled_mul(x)
    pB = randn(ComplexF64, length(B)); qq = randn(ComplexF64, length(Q))
    a1 = M[B, B] * pB; a2 = fmm_lhs_mul(op, pB)
    c1 = M[B, Q] * qq; c2 = -fmm_rhs(op, reshape(flux * qq, :, 1))[:, 1]
    println("A block err ", norm(a1 - a2) / norm(a1), "   interface block err ", norm(c1 - c2) / norm(c1))
    dense_fused = assemble_burton_miller_neumann_system_cpu(bem, p1, dp0, reshape(flux * qq, :, 1), k, rule;
        identity_p1_p1=Ipp, identity_p1_dp0=Ipd, singular_order=CONDENSED_SINGULAR_ORDER)
    println("fused dense vs coupled A: ", norm(dense_fused.matrix * pB - a1) / norm(a1), "  fused rhs vs coupled C: ", norm(dense_fused.rhs[:, 1] - c1) / norm(c1))
    println("quadrature order used: ", sys.regular_quadrature_order)
    for (name, r) in (("Γ", G), ("BEM", B), ("q", Q))
        println(rpad(name, 4), " rel err ", norm(yf[r] - yd[r]) / norm(yd[r]))
    end
    # preconditioner: the same block system with A and C replaced by their near-field parts
    An, Rn = fmm_near_blocks(op)
    nG, nB, nQ = length(G), length(B), length(Q)
    P = spzeros(ComplexF64, n, n)
    P[G, G] = sparse(sys.condensation.schur)
    P[G, Q] = -io.fem_load[gv, :]
    P[B, B] = An
    P[B, Q] = -(Rn * io.bem_flux)
    P[Q, G] = io.fem_trace[:, gv]
    P[Q, B] = -io.bem_trace
    FP0 = lu(P)
    FP = fmm_coupled_preconditioner(An, -(Rn * io.bem_flux) * (-1), sys.condensation.schur, io.fem_load[gv, :],
        io.fem_trace[:, gv], io.bem_trace, G, B, Q)
    rt = randn(MersenneTwister(9), ComplexF64, n)
    println("block-elimination P^-1 vs sparse LU of P: ", norm(FP \ rt - FP0 \ rt) / norm(FP0 \ rt))
    b = M * randn(MersenneTwister(2), ComplexF64, n)     # right-hand side with a known solution
    xs = M \ b
    Am = LinearMap{ComplexF64}(coupled_mul, n, n)
    println("cond(M) ≈ ", cond(M, 1), "   row-norm range ", extrema([norm(M[i, :]) for i in 1:n]))
    println("block row norms Γ/B/Q: ", [norm(M[r, :]) for r in (G, B, Q)])
    for (name, pc) in (("near block LU", LinearMap{ComplexF64}(v -> FP \ v, n, n)),)
        x, st = pc === nothing ? gmres(Am, b; rtol=1e-9, atol=0.0, memory=400, itmax=400) :
            gmres(Am, b; N=pc, ldiv=false, rtol=1e-9, atol=0.0, memory=400, itmax=400)
        println(rpad(name, 16), " iters $(st.niter) solved $(st.solved) err $(norm(x - xs) / norm(xs))")
        r = b - M * x
        println("   residual per block rel: ", [norm(r[rr]) / norm(b[rr]) for rr in (G, B, Q)])
        # left row scaling by the preconditioner's row norms
        D = Diagonal([1 / norm(P[i, :]) for i in 1:n])
        AmD = LinearMap{ComplexF64}(v -> D * coupled_mul(v), n, n)
        # (D P)^-1 = P^-1 D^-1: the same block preconditioner, applied to the unscaled residual
        x2, st2 = gmres(AmD, D * b; N=LinearMap{ComplexF64}(v -> FP \ (D \ v), n, n), ldiv=false, rtol=1e-9, atol=0.0, memory=400, itmax=400)
        println("   row-scaled: iters $(st2.niter) err $(norm(x2 - xs) / norm(xs))")
        x3, st3 = gmres(Am, b; N=pc, ldiv=false, rtol=1e-13, atol=0.0, memory=400, itmax=400)
        println("   rtol 1e-13: iters $(st3.niter) err $(norm(x3 - xs) / norm(xs))")
    end
end
