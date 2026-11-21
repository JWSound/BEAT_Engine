# GMRES on the FMM operator vs the dense solve; iteration counts against k*a.
isdefined(@__MODULE__, :BeatEngineCore) || include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
isdefined(@__MODULE__, :BeatEngineFmm) || include(joinpath(@__DIR__, "..", "src", "BeatEngineFmm.jl"))
using .BeatEngineFmm
using LinearAlgebra, StaticArrays, Random, Krylov, LinearMaps, Printf, DelimitedFiles
include(joinpath(@__DIR__, "fmm_test_meshes.jl"))

function probe(mesh, f; symmetry=:off, eps=1e-6, rtol=1e-6, precond=true)
    k = 2pi * f / 343
    p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float64, 2)
    Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    q = ones(ComplexF64, dp0.global_dof_count, 1)            # uniform normal velocity drive
    t_dense = @elapsed begin
        dense = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, k, rule;
            identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=symmetry, singular_order=2)
        xd = dense.matrix \ dense.rhs[:, 1]
    end
    t_setup = @elapsed op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp,
        identity_p1_dp0=Ipd, symmetry_mode=symmetry, eps)
    n = p1.global_dof_count
    A = LinearMap{ComplexF64}(x -> fmm_lhs_mul(op, x), n, n)
    b = fmm_rhs(op, q)[:, 1]
    t_pc = @elapsed F = fmm_near_preconditioner(op)
    P = LinearMap{ComplexF64}(x -> F \ x, n, n)
    t_solve = @elapsed begin
        x, stats = precond ? gmres(A, b; N=P, ldiv=false, rtol, restart=false, memory=300, itmax=300) :
            gmres(A, b; rtol, memory=300, itmax=300)
    end
    err = norm(x - xd) / norm(xd)
    @printf("faces %6d  f %6.0f  ka %5.2f  iters %4d  rel.err vs dense %.2e  dense %.1fs  fmm setup %.1fs pc %.1fs solve %.1fs\n",
        length(mesh.faces), f, k * 0.1, stats.niter, err, t_dense, t_setup, t_pc, t_solve)
    return stats.niter, err
end

function load_txt_mesh(prefix)
    V = [SVector{3,Float64}(r...) for r in eachrow(readdlm(prefix * "_v.txt"))]
    F = [Tuple(Int.(r)) for r in eachrow(readdlm(prefix * "_f.txt"))]
    tags = Int.(vec(readdlm(prefix * "_t.txt")))
    return BoundaryMesh(V, NTuple{3,Int}.(F), tags)
end

function probe_drive(mesh, f, drive_tag; symmetry=:off, eps=1e-6, rtol=1e-6, dense_check=true, memory=500)
    k = 2pi * f / 343
    p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float64, 2)
    Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    q = reshape(ComplexF64.(mesh.physical_tags .== drive_tag), :, 1)
    xd = nothing; t_dense = NaN
    if dense_check
        t_dense = @elapsed begin
            dense = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, k, rule;
                identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=symmetry, singular_order=2)
            rhs1 = dense.rhs[:, 1]
            Fd = lu!(dense.matrix)
            xd = Fd \ rhs1
            dense = nothing; Fd = nothing; GC.gc()
        end
    end
    t_setup = @elapsed op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp,
        identity_p1_dp0=Ipd, symmetry_mode=symmetry, eps)
    n = p1.global_dof_count
    A = LinearMap{ComplexF64}(x -> fmm_lhs_mul(op, x), n, n)
    b = fmm_rhs(op, q)[:, 1]
    F = fmm_near_preconditioner(op)
    P = LinearMap{ComplexF64}(x -> F \ x, n, n)
    t_mv = @elapsed fmm_lhs_mul(op, b)
    t_solve = @elapsed x, stats = gmres(A, b; N=P, ldiv=false, rtol, restart=false, memory, itmax=memory)
    err = xd === nothing ? NaN : norm(x - xd) / norm(xd)
    @printf("n %6d  f %6.0f  iters %4d  conv %s  rel.err %.2e  dense %.0fs  setup %.1fs  matvec %.2fs  solve %.0fs\n",
        n, f, stats.niter, stats.solved, err, t_dense, t_setup, t_mv, t_solve)
    return x, xd, stats
end
