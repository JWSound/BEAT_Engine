# Standalone, or included by the numerical test runners. Small actual Galerkin
# systems are the oracle; no H-matrix approximation or physical-accuracy claim.
# Reconstructed during download recovery; qualification uses the fresh run log.
if !isdefined(@__MODULE__, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end

module CpuGalerkinBlockSamplerTests
using Test, LinearAlgebra, SparseArrays, StaticArrays
using ..BeatEngineCore
const BEC = BeatEngineCore

function octahedron_mesh(::Type{T}, mode::Symbol; refine=true) where {T}
    vertices = SVector{3,T}[
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ]
    faces = mode === :xy ? NTuple{3,Int}[(1, 3, 5), (3, 1, 6)] :
        mode === :x ? NTuple{3,Int}[(1, 3, 5), (4, 1, 5), (3, 1, 6), (1, 4, 6)] :
        NTuple{3,Int}[(1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5),
                     (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6)]
    if refine
        midpoints = Dict{Tuple{Int,Int},Int}()
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3,Int}[]
        for (a, b, c) in faces
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            append!(refined, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        end
        faces = refined
    end
    used = sort(unique(collect(Iterators.flatten(faces))))
    mapping = Dict(old => new for (new, old) in enumerate(used))
    return BoundaryMesh(T(0.13) .* vertices[used],
        [(mapping[a], mapping[b], mapping[c]) for (a, b, c) in faces], fill(1, length(faces)))
end

function dense_reference(mesh::BoundaryMesh{T}, p1, dp0, k, rule;
                         identity_rule=rule, symmetry_mode=:off, kwargs...) where {T}
    pp = assemble_l2_identity_matrix(mesh, p1, dp0, identity_rule, :p1, :p1; symmetry_mode)
    pq = assemble_l2_identity_matrix(mesh, p1, dp0, identity_rule, :p1, :dp0; symmetry_mode)
    ops = assemble_regular_galerkin_operators_cpu(
        mesh, p1, dp0, k, rule; skip_singular=false, singular_order=3,
        symmetry_mode, threaded=false, regular_kernel=:scalar, kwargs...,
    )
    coupling = burton_miller_coupling(k)
    return (
        a=T(0.5) .* pp .- ops.double_layer .+ coupling .* ops.hypersingular,
        c=ops.single_layer .+ coupling .* (ops.adjoint_double_layer .+ T(0.5) .* pq),
        pp=pp, pq=pq,
    )
end

function tiled_sample(sampler, operator)
    rows = sampler.pressure_count
    columns = operator === :A ? rows : sampler.neumann_count
    result = Matrix{typeof(sampler.coupling)}(undef, rows, columns)
    for start_col in 1:3:columns, start_row in 1:4:rows
        rr = start_row:min(start_row + 3, rows)
        cc = start_col:min(start_col + 2, columns)
        BEC.sample_burton_miller_block!(view(result, rr, cc), sampler, rr, cc; operator)
    end
    return result
end

function check_relative(actual, reference, tolerance)
    @test size(actual) == size(reference)
    @test all(isfinite, actual)
    @test norm(actual - reference) <= tolerance * max(norm(reference), eps(real(eltype(reference))))
end

@testset "CPU lazy Galerkin blocks against dense production assembly" begin
    for T in (Float32, Float64), mode in (:off, :x, :xy),
        phasor in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        @testset "$T $mode $phasor" begin
            with_phasor_convention(phasor) do
                mesh = octahedron_mesh(T, mode)
                p1 = build_p1_space(mesh)
                # The supplied global DP0 numbering, not the triangle index,
                # determines the selected flux column.
                dp0 = BEC.DP0Space(reverse(collect(eachindex(mesh.faces))), length(mesh.faces))
                rule = triangle_rule(T, 3)
                tolerance = T === Float32 ? 1e-5 : 2e-12
                for k in T.((7.0, 120.0))
                    reference = dense_reference(mesh, p1, dp0, k, rule; symmetry_mode=mode)
                    sampler = BEC.build_cpu_galerkin_block_sampler(
                        mesh, p1, dp0, k, rule; singular_order=3, symmetry_mode=mode,
                    )
                    @test sampler.phasor == phasor
                    @test sampler.mass_pp isa SparseMatrixCSC{T,Int}
                    @test sampler.mass_pq isa SparseMatrixCSC{T,Int}
                    @test nnz(sampler.mass_pp) <= 9 * length(mesh.faces)
                    @test nnz(sampler.mass_pq) <= 3 * length(mesh.faces)
                    @test sum(length, sampler.p1_incidence) == 3 * length(mesh.faces)
                    check_relative(Matrix(sampler.mass_pp), reference.pp, tolerance)
                    check_relative(Matrix(sampler.mass_pq), reference.pq, tolerance)
                    a = tiled_sample(sampler, :A)
                    c = tiled_sample(sampler, :C)
                    check_relative(a, reference.a, tolerance)
                    check_relative(c, reference.c, tolerance)
                    for (operator, expected) in ((:A, reference.a), (:C, reference.c))
                        nr, nc = size(expected)
                        # Preserve reordered/repeated entries and the shapes of
                        # empty row, column, and jointly empty selections.
                        for (rows, columns) in (([nr, 1, nr], [nc, 1, nc, 2]),
                                                (Int[], [nc, 1]),
                                                ([nr, 1], Int[]), (Int[], Int[]))
                            block = BEC.sample_burton_miller_block(sampler, rows, columns; operator)
                            check_relative(block, expected[rows, columns], tolerance)
                        end
                    end
                    # A sampler captures its convention: changing the worker's
                    # current convention later must not change its coefficients.
                    opposite = phasor == NEGATIVE_TIME_PHASOR ? POSITIVE_TIME_PHASOR : NEGATIVE_TIME_PHASOR
                    with_phasor_convention(opposite) do
                        check_relative(BEC.sample_burton_miller_block(sampler, [1, 2], [1, 2]),
                            reference.a[[1, 2], [1, 2]], tolerance)
                        check_relative(BEC.sample_burton_miller_block(sampler, [1, 2], [1, 2]; operator=:C),
                            reference.c[[1, 2], [1, 2]], tolerance)
                    end
                    real_q = T[sin(T(0.7 * row + 1.3 * drive))
                               for row in 1:dp0.global_dof_count, drive in 1:3]
                    complex_q = Complex{T}[complex(real_q[row, drive], cos(T(0.4 * row - 0.9 * drive)))
                                           for row in 1:dp0.global_dof_count, drive in 1:3]
                    for q in (real_q, complex_q)
                        fused = assemble_burton_miller_neumann_system_cpu(
                            mesh, p1, dp0, q, k, rule;
                            identity_p1_p1=reference.pp, identity_p1_dp0=reference.pq,
                            singular_order=3, symmetry_mode=mode, threaded=false,
                        )
                        check_relative(a, fused.matrix, tolerance)
                        check_relative(-(c * q), fused.rhs, tolerance)
                        check_relative(a \ (-(c * q)), fused.matrix \ fused.rhs,
                            T === Float32 ? 2e-4 : 1e-10)
                    end
                end
            end
        end
    end
end

function separated_triangles(::Type{T}) where {T}
    return BoundaryMesh(SVector{3,T}[
        (0.02, 0.03, 0), (0.07, 0.03, 0), (0.02, 0.08, 0),
        (0.022, 0.031, 0.012), (0.072, 0.031, 0.012), (0.022, 0.081, 0.012),
    ], [(1, 2, 3), (4, 5, 6)], [1, 1])
end

@testset "CPU lazy blocks preserve near, image and separate mass rules" begin
    for T in (Float32, Float64), mode in (:off, :xy),
        phasor in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        with_phasor_convention(phasor) do
            mesh = separated_triangles(T)
            p1, dp0 = build_p1_space(mesh), BEC.DP0Space([2, 1], 2)
            rule, identity_rule = triangle_rule(T, 1), triangle_rule(T, 3)
            k = T(23)
            tolerance = T === Float32 ? 1e-5 : 2e-12
            near = build_near_correction_cache(mesh, [(1, 2, 4), (2, 1, 6)], 6)
            images = [build_near_correction_cache(mesh, [(1, 2, 4), (2, 1, 6)], 6;
                        trial_transform=transform) for transform in BEC.symmetry_image_transforms(mode)]
            @test near.correction_orders == [4, 6]
            @test near.pair_count == 2
            reference = dense_reference(mesh, p1, dp0, k, rule; symmetry_mode=mode,
                identity_rule, near_correction_cache=near, image_near_correction_cache=images)
            @test norm(reference.pp - assemble_l2_identity_matrix(
                mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=mode)) > T(1e-7)
            for (direct, reflected) in ((near, images), ((near,), Tuple(images)),
                                        ((near, images...), nothing))
                sampler = BEC.build_cpu_galerkin_block_sampler(mesh, p1, dp0, k, rule;
                    singular_order=3, symmetry_mode=mode, identity_rule,
                    near_correction_cache=direct, image_near_correction_cache=reflected)
                check_relative(tiled_sample(sampler, :A), reference.a, tolerance)
                check_relative(tiled_sample(sampler, :C), reference.c, tolerance)
                check_relative(Matrix(sampler.mass_pp), reference.pp, tolerance)
                check_relative(Matrix(sampler.mass_pq), reference.pq, tolerance)
            end
        end
    end
end

@testset "CPU lazy block contracts fail explicitly" begin
    mesh = separated_triangles(Float64)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    k, rule = 7.0, triangle_rule(Float64, 3)
    build(; kwargs...) = BEC.build_cpu_galerkin_block_sampler(mesh, p1, dp0, k, rule; kwargs...)
    sampler = build()
    @test_throws ArgumentError build(symmetry_mode=:ground)
    @test_throws ArgumentError build(singular_order=0)
    for bad_k in (0.0, Inf, NaN)
        @test_throws ArgumentError BEC.build_cpu_galerkin_block_sampler(mesh, p1, dp0, bad_k, rule)
    end
    @test_throws BoundsError BEC.sample_burton_miller_block(sampler, [0], [1])
    @test_throws BoundsError BEC.sample_burton_miller_block(sampler, [1], [dp0.global_dof_count + 1]; operator=:C)
    @test_throws ArgumentError BEC.sample_burton_miller_block(sampler, [1], [1]; operator=:other)
    @test_throws ArgumentError BEC.sample_burton_miller_block!(zeros(ComplexF32, 1, 1), sampler, [1], [1])
    @test_throws DimensionMismatch BEC.sample_burton_miller_block!(zeros(ComplexF64, 2, 1), sampler, [1], [1])
    @test_throws ArgumentError BEC.build_cpu_galerkin_block_sampler(mesh,
        BEC.P1Space(reverse(p1.local_to_global), p1.global_dof_count), dp0, k, rule)
    @test_throws ArgumentError BEC.build_cpu_galerkin_block_sampler(mesh,
        p1, BEC.DP0Space([1, 1], 2), k, rule)
    @test_throws ArgumentError build(near_correction_cache=Dict())
    inactive = build_near_correction_cache(mesh, [(1, 2)], 4;
        trial_transform=first(BEC.symmetry_image_transforms(:x)))
    @test_throws ArgumentError build(image_near_correction_cache=inactive)
    near = build_near_correction_cache(mesh, [(1, 2)], 4)
    malformed = BEC.NearCorrectionCache(near.pairs_by_test, near.pairs,
        near.correction_rules, near.correction_orders, near.trial_transform,
        near.correction_order, near.pair_count + 1)
    @test_throws ArgumentError build(near_correction_cache=malformed)
    empty_rule = BEC.TriangleRule(SVector{2,Float64}[], Float64[])
    @test_throws ArgumentError BEC.build_cpu_galerkin_block_sampler(mesh, p1, dp0, k, empty_rule)
end

end # module
