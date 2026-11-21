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
  withenv("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "0") do
    f = parse(Float64, get(ENV, "F", "500"))
    rtag = physical_tag(fem, 2, "Radiator")
    tr = ElectrodynamicTransducer{Float64}("component:test", [rtag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
        2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0)
    sys = build_condensed_coupled_system(fem, bem, imap, f, 343.0, 1.21; quadrature_order=CONDENSED_QUADRATURE_ORDER,
        singular_order=CONDENSED_SINGULAR_ORDER, transducers=[tr])
    Fd = sys.factorization
    M = Fd.P' * (Fd.L * Fd.U)
    G, B, Q, Mr, Er = sys.gamma_range, sys.bem_range, sys.flux_range, sys.mechanical_range, sys.electrical_range
    println("layout $(sys.interface_elimination)  Γ $G B $B q $Q mech $Mr elec $Er  transducer condensation $(sys.transducer_condensation)")
    io = sys.cache.base.interface_operators
    k = 2pi * f / 343; omega = 2pi * f
    tops = assemble_transducer_operators(fem, bem, [tr])
    motion_flux = neumann_scale(1.21, omega) .* Matrix{ComplexF64}(tops.bem_normal_velocity)
    println("BEM motion flux nonzeros: ", count(!iszero, motion_flux))
    p1 = build_p1_space(bem); dp0 = build_dp0_space(bem); rule = triangle_rule(Float64, CONDENSED_QUADRATURE_ORDER)
    op = fmm_burton_miller_operator(bem, p1, dp0, k, rule; identity_p1_p1=sys.cache.base.identity_p1_p1,
        identity_p1_dp0=sys.cache.base.identity_p1_dp0, singular_order=CONDENSED_SINGULAR_ORDER,
        symmetry_mode=sys.symmetry_mode, eps=1e-10)
    # Everything that does not involve the BEM operators stays an ordinary (sparse / small dense) block;
    # the BEM row blocks [B,B], [B,Q], [B,mech] come from the FMM operator.
    rest = copy(M); rest[B, B] .= 0; rest[B, Q] .= 0; isempty(Mr) || (rest[B, Mr] .= 0)
    flux = Matrix{ComplexF64}(io.bem_flux)
    function coupled_mul(x)
        y = rest * x
        dp0_data = flux * x[Q]
        isempty(Mr) || (dp0_data .+= motion_flux * x[Mr])
        y[B] .+= fmm_lhs_mul(op, x[B]) .- fmm_rhs(op, reshape(dp0_data, :, 1))[:, 1]
        return y
    end
    n = size(M, 1)
    x = randn(MersenneTwister(4), ComplexF64, n)
    yd = M * x; yf = coupled_mul(x)
    for (name, r) in (("Γ", G), ("BEM", B), ("q", Q), ("mech", Mr), ("elec", Er))
        isempty(r) && continue
        println(rpad(name, 5), " rel err ", norm(yf[r] - yd[r]) / max(norm(yd[r]), 1e-300))
    end
    # near-field bordered preconditioner
    An, Rn = fmm_near_blocks(op)
    S2 = setdiff(1:n, B)
    PBS = spzeros(ComplexF64, length(B), length(S2))
    posQ = [findfirst(==(i), S2) for i in Q]
    PBS[:, posQ] = -(Rn * io.bem_flux)
    if !isempty(Mr)
        posM = [findfirst(==(i), S2) for i in Mr]
        PBS[:, posM] = -(Rn * sparse(motion_flux))
    end
    PSB = sparse(rest[S2, B]); PSS = rest[S2, S2]
    t = @elapsed PC = fmm_bordered_preconditioner(An, PBS, PSB, PSS, B, S2)
    println("preconditioner setup $(round(t, digits=2)) s  |S2| = $(length(S2))")
    b = M * randn(MersenneTwister(2), ComplexF64, n); xs = M \ b
    Pfull = spzeros(ComplexF64, n, n); Pfull[B, B] = An; Pfull[B, S2] = PBS; Pfull[S2, B] = PSB; Pfull[S2, S2] = PSS
    D = Diagonal([1 / norm(Pfull[i, :]) for i in 1:n])
    AmD = LinearMap{ComplexF64}(v -> D * coupled_mul(v), n, n)
    x, st = gmres(AmD, D * b; N=LinearMap{ComplexF64}(v -> PC \ (D \ v), n, n), ldiv=false, rtol=1e-9, atol=0.0, memory=300, itmax=300)
    println("GMRES (row-scaled, bordered near preconditioner): iters $(st.niter) solved $(st.solved) err $(norm(x - xs) / norm(xs))")
    SOL = FmmCoupledSolve(op, rest, B, Q, Mr, io.bem_flux, sparse(motion_flux); rtol=1e-9)
    Rhs = M * randn(MersenneTwister(11), ComplexF64, n, 2)
    Xd = M \ Rhs; Xf = SOL \ Rhs
    println("FmmCoupledSolve vs dense (2 columns): ", norm(Xf - Xd) / norm(Xd), "  stats ", SOL.stats)
  end
end
