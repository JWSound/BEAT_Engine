# Split-precision dense coupled storage (BeatEngineSplitDense.jl): exactness against the
# ComplexF64 matrix, equivalence with RefinedDenseLU, spilled residual store, GMRES-IR and the
# last-resort Float64 fallback.
isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
using Test, LinearAlgebra, Random

const SD = BeatEngineCoupledCondensed

# A system laid out like the flux-eliminated dense build: ComplexF32 BEM block, Float64 border
# rows/columns, Float64 accumulations into a few "Γ-mapped" BEM columns.
function split_fixture(rng, n, border; condition=1e3)
    m = n - border
    bem = randn(rng, ComplexF32, m, m) ./ Float32(sqrt(m)) + 2.0f0I
    if condition > 1e3
        u, _ = qr(randn(rng, ComplexF64, m, m))
        v, _ = qr(randn(rng, ComplexF64, m, m))
        sigma = exp10.(range(0, -log10(condition); length=m))
        bem = ComplexF32.(Matrix(u) * Diagonal(sigma) * Matrix(v)')
    end
    gamma_columns = sort(randperm(rng, m)[1:max(1, m ÷ 10)])
    schur = randn(rng, ComplexF64, m, length(gamma_columns)) .* 1e-3
    border_cols = randn(rng, ComplexF64, n, border)
    border_rows = randn(rng, ComplexF64, border, m)
    return (; n, m, bem, gamma_columns, schur, border_cols, border_rows)
end

function split_fill!(target, f)
    m, n = f.m, f.n
    target[1:m, 1:m] = f.bem
    for (local_column, column) in enumerate(f.gamma_columns)
        @views target[1:m, column] .+= f.schur[:, local_column]
    end
    target[1:n, (m + 1):n] = f.border_cols
    target[(m + 1):n, 1:m] = f.border_rows
    @views target[(m + 1):n, f.gamma_columns[1]] .+= 1e-7          # an accumulation in a border row
    return target
end

split_build(f) = split_fill!(SD.SplitDenseMatrix(f.n, (f.m + 1):f.n, vcat((f.m + 1):f.n, f.gamma_columns)), f)
dense_build(f) = split_fill!(zeros(ComplexF64, f.n, f.n), f)

@testset "SplitDenseMatrix reproduces the ComplexF64 dense build exactly" begin
    rng = MersenneTwister(11)
    f = split_fixture(rng, 160, 12)
    split = split_build(f)
    dense = dense_build(f)
    @test Matrix(split) == dense                                   # bit-exact reconstruction
    @test split.hi == ComplexF32.(dense)                            # the factor RefinedDenseLU computes
    @test isempty(split.stray) && !split.overflow
    @test SD._split_opnorm_inf(split.hi, split) ≈ opnorm(dense, Inf) rtol = 1e-14
    x = randn(rng, ComplexF64, f.n, 3)
    y = zeros(ComplexF64, f.n, 3)
    SD._split_mul!(y, split.hi, split, x, one(ComplexF64), zero(ComplexF64))
    @test y ≈ dense * x rtol = 1e-13
    # A Float64 value outside the declared border is still stored exactly (as a stray).
    j = first(setdiff(1:f.m, f.gamma_columns))
    split[1, j] = 0.1
    @test split[1, j] === ComplexF64(0.1) && length(split.stray) == 1
    split[1, j] = 0.5                                               # representable: stray removed
    @test split[1, j] === ComplexF64(0.5) && isempty(split.stray)
end

@testset "SplitRefinedLU matches RefinedDenseLU" begin
    rng = MersenneTwister(12)
    f = split_fixture(rng, 220, 16)
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 4)
    reference = SD.RefinedDenseLU(copy(dense)) \ rhs
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            factorization = SD.SplitRefinedLU(split_build(f))
            @test (factorization.spill_path !== nothing) == (spill == "on")
            @test spill == "off" || isempty(factorization.matrix.hi)   # factored in place when spilled
            solution = factorization \ rhs
            @test norm(solution - reference) / norm(reference) < 1e-12
            @test factorization.backward_error <= 1 && isnothing(factorization.fallback)
            @test factorization.gmres_iterations == 0
            @test vec(factorization \ rhs[:, 1]) ≈ solution[:, 1] rtol = 1e-13
            @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_solver"] == "lu_float32_split_refined"
            @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_spill"] == (spill == "on")
        end
    end
end

@testset "GMRES-IR replaces the Float64 fallback when refinement stalls" begin
    rng = MersenneTwister(13)
    f = split_fixture(rng, 200, 10; condition=1e9)                  # κ eps(Float32) ≫ 1
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 2)
    exact = lu(dense) \ rhs
    # Both answers are Float64 backward stable; they may differ by up to ~κ eps(Float64).
    forward_tolerance = 100 * cond(dense, 1) * eps(Float64)
    @test cond(dense, 1) * eps(Float32) > 10                        # classical refinement cannot converge
    factorization = SD.SplitRefinedLU(split_build(f))
    solution = factorization \ rhs
    @test isnothing(factorization.fallback)
    @test factorization.gmres_iterations > 0
    @test factorization.backward_error <= 1
    @test norm(dense * solution - rhs) / (opnorm(dense, Inf) * norm(solution, Inf)) < 1e-13
    @test norm(solution - exact) / norm(exact) < forward_tolerance
    withenv("BLAB_COUPLED_DENSE_GMRES_IR" => "0") do               # old behaviour: Float64 LU
        legacy = SD.SplitRefinedLU(split_build(f))
        legacy_solution = (@test_logs (:warn,) match_mode=:any legacy \ rhs)
        @test !isnothing(legacy.fallback) && occursin("stalled", something(legacy.fallback_reason, ""))
        @test norm(legacy_solution - exact) / norm(exact) < forward_tolerance
    end
end

@testset "SplitRefinedLU falls back to Float64 for out-of-range entries" begin
    rng = MersenneTwister(14)
    f = split_fixture(rng, 120, 8)
    split = split_build(f)
    split[f.n, f.n] = 1.0e39                                        # beyond floatmax(Float32)
    @test split.overflow && split[f.n, f.n] == 1.0e39
    dense = Matrix(split)
    rhs = randn(rng, ComplexF64, f.n)
    factorization = @test_logs (:warn,) match_mode=:any SD.SplitRefinedLU(split)
    @test !isnothing(factorization.fallback) && occursin("Float32 range", factorization.fallback_reason)
    @test factorization \ rhs ≈ dense \ rhs rtol = 1e-10
    @test_throws ArgumentError SD.SplitRefinedLU(split_build(merge(f, (; bem=fill(NaN32 + 0im, f.m, f.m)))))
end
