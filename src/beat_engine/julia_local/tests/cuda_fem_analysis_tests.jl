# Focused hardware gate; also included by the CUDA coupled test suite.
@testset "CUDA FEM symbolic analysis reuse" begin
    cuda = BeatEngineCore.cuda_module()
    cudss = BeatEngineCoupled._cudss_module()
    cache = Ref{Any}(nothing)
    operators = BeatEngineCoupled.InterfaceOperators{Float32}(
        spzeros(Float32, 5, 1), spzeros(Float32, 1, 1),
        spzeros(Float32, 1, 5), spzeros(Float32, 1, 1),
    )
    base = ComplexF32[5 1 0 0 1; 1 6 2 0 0; 0 2 7 1 0; 0 0 1 8 2; 1 0 0 2 9]
    previous_solver = nothing
    try
        for (index, shift) in enumerate((0f0, 2f0, -3f0, 0f0))
            matrix = sparse(base + (shift + 0.02f0im) * I)
            cached = BeatEngineCoupled._build_cuda_fem_condensation(
                matrix, operators, [4, 5]; analysis_cache=cache,
            )
            fresh = BeatEngineCoupled._build_cuda_fem_condensation(matrix, operators, [4, 5])
            try
                @test cached.analysis_reused == (index > 1)
                if index > 1
                    @test cached.solver === previous_solver
                    @test cached.timings.analysis_s == 0
                end
                @test_throws ErrorException BeatEngineCoupled._build_cuda_fem_condensation(
                    matrix, operators, [4, 5]; analysis_cache=cache,
                )
                reference = ComplexF64.(Matrix(matrix))
                expected = reference[4:5, 4:5] - reference[4:5, 1:3] *
                    (reference[1:3, 1:3] \ reference[1:3, 4:5])
                @test Array(cached.device_schur) ≈ expected rtol=2e-6
                @test Array(cached.device_schur) ≈ Array(fresh.device_schur) rtol=2e-6
                rhs = cuda.CuArray(reshape(ComplexF32.(1:10), 5, 2))
                forward = similar(rhs)
                device_boundary = nothing
                try
                    cudss.cudss("solve_fwd_schur", cached.solver, forward, rhs; asynchronous=false)
                    reduced_rhs = Array(forward)[4:5, :]
                    boundary = ComplexF32.(expected \ reduced_rhs)
                    device_boundary = cuda.CuArray(boundary)
                    view(forward, 4:5, :) .= device_boundary
                    cudss.cudss("solve_bwd_schur", cached.solver, rhs, forward; asynchronous=false)
                    solution = Array(rhs)
                    solution[4:5, :] = boundary
                    @test norm(reference * solution - reshape(1:10, 5, 2)) / norm(1:10) < 2e-6
                finally
                    cuda.unsafe_free!(rhs)
                    cuda.unsafe_free!(forward)
                    isnothing(device_boundary) || cuda.unsafe_free!(device_boundary)
                end
                previous_solver = cached.solver
            finally
                BeatEngineCoupled._release_cuda_fem_condensation!(fresh)
                BeatEngineCoupled._release_cuda_fem_condensation!(cached)
            end
        end
        # Both sparsity and the ordered retained partition are analysis inputs.
        matrix = sparse(base + 0.02f0im * I)
        matrix[1, 3] = 0.5f0
        for retained in ([4, 5], [5, 4])
            cached = BeatEngineCoupled._build_cuda_fem_condensation(
                matrix, operators, retained; analysis_cache=cache,
            )
            @test !cached.analysis_reused
            @test cached.solver !== previous_solver
            fresh = BeatEngineCoupled._build_cuda_fem_condensation(matrix, operators, retained)
            try
                @test Array(cached.device_schur) ≈ Array(fresh.device_schur) rtol=2e-6
            finally
                BeatEngineCoupled._release_cuda_fem_condensation!(fresh)
            end
            previous_solver = cached.solver
            BeatEngineCoupled._release_cuda_fem_condensation!(cached)
        end
        operators64 = BeatEngineCoupled.InterfaceOperators{Float64}(
            spzeros(5, 1), spzeros(1, 1), spzeros(1, 5), spzeros(1, 1),
        )
        cached64 = BeatEngineCoupled._build_cuda_fem_condensation(
            ComplexF64.(matrix), operators64, [5, 4]; analysis_cache=cache,
        )
        @test !cached64.analysis_reused
        @test eltype(cached64.device_system) == ComplexF64
        BeatEngineCoupled._release_cuda_fem_condensation!(cached64)
    finally
        BeatEngineCoupled._release_cuda_fem_analysis!(cache)
    end
    @test isnothing(cache[])
    BeatEngineCoupled._release_cuda_fem_analysis!(cache)

    # Cross an interior pole with weak damping. Compare the *same rounded FP32
    # matrix* in FP64, separating reuse errors from coefficient quantization.
    pole = Float32(eigmin(Symmetric(real.(base[1:3, 1:3]))))
    try
        for (index, offset) in enumerate((-0.01f0, -0.001f0, 0.001f0, 0.01f0))
            matrix = sparse(base + (-pole + offset + 1f-5im) * I)
            cached = BeatEngineCoupled._build_cuda_fem_condensation(
                matrix, operators, [4, 5]; analysis_cache=cache,
            )
            fresh = BeatEngineCoupled._build_cuda_fem_condensation(matrix, operators, [4, 5])
            try
                reference = ComplexF64.(Matrix(matrix))
                expected = reference[4:5, 4:5] - reference[4:5, 1:3] *
                    (reference[1:3, 1:3] \ reference[1:3, 4:5])
                # Fresh cuDSS runs also vary near the pole. Bound each result
                # against FP64 using the interior condition number, rather than
                # applying the well-conditioned cached/fresh agreement tolerance.
                accuracy_bound = eps(Float32) * cond(reference[1:3, 1:3])
                @test cached.analysis_reused == (index > 1)
                @test all(isfinite, Array(cached.device_schur))
                @test Array(fresh.device_schur) ≈ expected rtol=accuracy_bound
                @test Array(cached.device_schur) ≈ expected rtol=accuracy_bound
            finally
                BeatEngineCoupled._release_cuda_fem_condensation!(fresh)
                BeatEngineCoupled._release_cuda_fem_condensation!(cached)
            end
        end
    finally
        BeatEngineCoupled._release_cuda_fem_analysis!(cache)
    end
end
