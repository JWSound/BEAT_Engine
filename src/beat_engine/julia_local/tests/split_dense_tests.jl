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

@testset "GMRES-IR rejects non-finite residuals after a finite RHS overflows Float32" begin
    # The original system is finite and exactly solvable. Converting the RHS for
    # the initial Float32 solve overflows; NaN residual ratios must not make the
    # GMRES-IR active set look converged. Include an already-solved finite column.
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill, "BLAB_COUPLED_DENSE_GMRES_IR" => "1") do
            for rhs in (ComplexF64[1e39, 2e39], ComplexF64[1 1e39; 2 2e39])
                split = SD.SplitDenseMatrix(2, Int[], Int[])
                split[1, 1] = 1
                split[2, 2] = 1
                factorization = SD.SplitRefinedLU(split)
                solution = (@test_logs (:warn,) match_mode=:any factorization \ rhs)
                @test size(solution) == size(rhs)
                @test all(isfinite, solution)
                @test solution == rhs
                @test !isnothing(factorization.fallback)
                @test occursin("non-finite residual", something(factorization.fallback_reason, ""))
                @test isfinite(factorization.backward_error) && factorization.backward_error <= 1
                @test factorization.gmres_iterations == 0
            end
        end
    end
end

@testset "SplitRefinedLU around a device-held factor (CUDA path, simulated on the host)" begin
    rng = MersenneTwister(15)
    n = 180
    matrix = randn(rng, ComplexF32, n, n) ./ Float32(sqrt(n)) + 2.0f0I
    rhs = randn(rng, ComplexF64, n, 3)
    reference = lu(ComplexF64.(matrix)) \ rhs
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            host = SD.SplitDenseMatrix(n, Int[], Int[])
            copyto!(host.hi, matrix)
            device_factor = lu(copy(matrix))                       # stands in for the CUDA LU
            solves = Ref(0)
            released = Ref(false)
            factorization = SD.SplitRefinedLU(host,
                x -> (solves[] += 1; ComplexF64.(device_factor \ x)), () -> (released[] = true; nothing))
            @test isnothing(factorization.factor) && (factorization.spill_path !== nothing) == (spill == "on")
            solution = factorization \ rhs
            @test norm(solution - reference) / norm(reference) < 1e-12
            @test factorization.backward_error <= 1 && solves[] >= 2  # initial solve + refinement
            @test vec(factorization \ rhs[:, 2]) ≈ solution[:, 2] rtol = 1e-13
            SD.release_split_factor!(factorization)
            @test released[] && factorization.device_solve === nothing
            SD.release_split_factor!(factorization)                  # idempotent
        end
    end
end

@testset "dense LU library switch (BLAB_DENSE_LU_LIBRARY)" begin
    core = BeatEngineCore
    rng = MersenneTwister(7)
    for T in (ComplexF32, ComplexF64, Float64)
        A = randn(rng, T, 40, 40) + 8I
        withenv("BLAB_DENSE_LU_LIBRARY" => nothing) do
            @test core.dense_lu_library(T) == "blas"
            F = core.dense_lu!(copy(A))
            @test F.factors == lu(A).factors && F.ipiv == lu(A).ipiv
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "openblas") do
            @test core.dense_lu_library(T) == "blas"
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "auto") do
            library = core.dense_lu_library(T)
            @test library in ("blas", "accelerate")
            F = core.dense_lu!(copy(A))
            @test F isa LinearAlgebra.LU
            b = randn(rng, T, 40)
            @test norm(A * (F \ b) - b) <= 100 * eps(real(T)) * opnorm(A, 1) * norm(F \ b) * 40
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "accelerate") do
            if core._accelerate_getrf(T) == C_NULL
                @test_throws ErrorException core.dense_lu_library(T)
                @test startswith(core.dense_lu_library_diagnostic(T), "unavailable")
            else
                F = core.dense_lu!(copy(A))
                b = randn(rng, T, 40)
                @test norm(A * (F \ b) - b) <= 100 * eps(real(T)) * opnorm(A, 1) * norm(F \ b) * 40
                @test_throws SingularException core.dense_lu!(zeros(T, 4, 4))
                @test !issuccess(core.dense_lu!(zeros(T, 4, 4); check=false))
            end
        end
    end
    withenv("BLAB_DENSE_LU_LIBRARY" => "mkl") do
        @test_throws ErrorException core.dense_lu_library(ComplexF32)
    end
    @test core.dense_lu!(view(randn(rng, 6, 6) + 4I, :, :)) isa LinearAlgebra.LU   # non-Matrix: lu!
end

@testset "SplitGMRES (BLAB_COUPLED_DENSE_SOLVER=gmres): no factorization, falls back to LU" begin
    rng = MersenneTwister(21)
    f = split_fixture(rng, 200, 10)
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 5)
    reference = dense \ rhs
    for precond in ("jacobi", "none")
        withenv("BLAB_COUPLED_DENSE_GMRES_PRECOND" => precond, "BLAB_COUPLED_DENSE_GMRES_TOL" => "1e-9") do
            solver = SD.SplitGMRES(split_build(f))
            solution = solver \ rhs
            @test isnothing(solver.fallback)
            @test solver.relative_residual <= 1e-9
            @test norm(solution - reference) / norm(reference) < 1e-6
            @test solver.inner_iterations > 0 && solver.outer_iterations >= 1
            d = SD.dense_solver_diagnostics((factorization=solver,))
            @test d["dense_solver"] == "gmres_float32_ir" && d["dense_gmres_preconditioner"] == precond
            @test vec(solver \ rhs[:, 2]) ≈ solution[:, 2] rtol = 1e-6
        end
    end
    withenv("BLAB_COUPLED_DENSE_GMRES_MAXITER" => "3") do              # budget too small: LU fallback
        solver = SD.SplitGMRES(split_build(f))
        solution = (@test_logs (:warn,) match_mode=:any solver \ rhs)
        @test solver.fallback isa SD.SplitRefinedLU
        @test norm(solution - reference) / norm(reference) < 1e-12
        @test SD.dense_solver_diagnostics((factorization=solver,))["dense_solver"] == "gmres_fallback_lu"
    end
end

@testset "SplitGMRES falls back when its Float32 correction is singular" begin
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            # The FP64 border correction makes the exact system nonsingular,
            # but its Float32 surrogate annihilates this right-hand side.
            split = SD.SplitDenseMatrix(2, Int[], [2])
            split[1, 1] = split[1, 2] = split[2, 1] = 1
            split[2, 2] = 1 + 1e-10
            rhs = ComplexF64[1, -1]
            reference = Matrix(split) \ rhs
            solver = SD.SplitGMRES(split)
            solution = (@test_logs (:warn,) (:warn,) match_mode=:any solver \ rhs)
            @test all(isfinite, solution)
            @test solution ≈ reference rtol = 1e-12
            @test solver.fallback isa SD.SplitRefinedLU
            @test !isnothing(solver.fallback.fallback)
            @test occursin("singular projected system", something(solver.fallback_reason, ""))
        end
    end
end

@testset "SplitGMRES checks convergence after its final allowed correction" begin
    withenv("BLAB_COUPLED_DENSE_SPILL" => "off", "BLAB_COUPLED_DENSE_GMRES_MAXITER" => "1") do
        for rhs in (ComplexF64[1], ComplexF64[1 2])
            split = SD.SplitDenseMatrix(1, Int[], Int[])
            split[1, 1] = 1
            solver = SD.SplitGMRES(split)
            solution = (@test_logs solver \ rhs)
            @test solution == rhs
            @test isnothing(solver.fallback)
            @test solver.relative_residual == 0
            @test solver.inner_iterations == 1
            @test solver.outer_iterations == 1
        end
    end
end

@testset "Adopted A (BLAB_METAL_COUPLED_ADOPT_A): restrided in place, same system as a copy" begin
    rng = MersenneTwister(31)
    for (n, border) in ((160, 12), (97, 0), (64, 1))
        f = split_fixture(rng, n, border)
        m = f.m
        # Host memory laid out like the Metal assembly leaves it: A (m x m, leading dimension m)
        # first, then junk up to the n^2 capacity that must not leak into the system.
        address = Libc.malloc(n * n * sizeof(ComplexF32))
        raw = unsafe_wrap(Array, Ptr{ComplexF32}(address), n * n; own=false)
        fill!(raw, ComplexF32(NaN, NaN))
        copyto!(raw, 1, vec(f.bem), 1, m * m)
        hi = SD.adopt_square_storage(address, m, n)
        @test size(hi) == (n, n) && pointer(hi) == Ptr{ComplexF32}(address)
        adopted = SD.SplitDenseMatrix(hi, (m + 1):n, vcat((m + 1):n, f.gamma_columns))
        # Everything the build writes after the BEM block copy, which the adopted path skips.
        for (local_column, column) in enumerate(f.gamma_columns)
            @views adopted[1:m, column] .+= f.schur[:, local_column]
        end
        border > 0 && (adopted[1:n, (m + 1):n] = f.border_cols)
        border > 0 && (adopted[(m + 1):n, 1:m] = f.border_rows)
        border > 0 && (@views adopted[(m + 1):n, f.gamma_columns[1]] .+= 1e-7)
        @test Matrix(adopted) == dense_build(f)
        @test adopted.hi == split_build(f).hi
        rhs = randn(rng, ComplexF64, n, 2)
        @test SD.SplitRefinedLU(adopted) \ rhs ≈ dense_build(f) \ rhs rtol = 1e-10
    end
    @test_throws ArgumentError SD.adopt_square_storage(Ptr{Cvoid}(1), 5, 4)
end
