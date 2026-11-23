# One workspace per full-order training model. Only the last frequency's numerical
# factors survive, while the symbolic analysis can span the whole training band.
mutable struct ContinuousCudaInterior
    matrix
    solver
    rhs
    solution
    colptr::Vector{Int}
    rowval::Vector{Int}
    frequency_hz::Float64
end

ContinuousCudaInterior() = ContinuousCudaInterior(nothing, nothing, nothing, nothing, Int[], Int[], NaN)

function release_continuous_interior!(work::ContinuousCudaInterior)
    cuda = BeatEngineCore.cuda_module()
    if !isnothing(work.solver)
        finalize(work.solver.data)
        finalize(work.solver.config)
        finalize(work.solver.matrix)
        work.solver = nothing
    end
    BeatEngineCoupled._unsafe_free_cuda_fields!(work.matrix, (:rowPtr, :colVal, :nzVal))
    work.matrix = nothing
    isnothing(work.rhs) || cuda.unsafe_free!(work.rhs)
    isnothing(work.solution) || cuda.unsafe_free!(work.solution)
    work.rhs = work.solution = nothing
    work.frequency_hz = NaN
    return nothing
end

function continuous_interior_factor!(work::ContinuousCudaInterior, matrix::SparseMatrixCSC{ComplexF64}, frequency_hz)
    # Caller guarantees a single physical model for the lifetime of this workspace.
    work.frequency_hz == frequency_hz && return work
    cuda = BeatEngineCore.cuda_module()
    cudss = BeatEngineCoupled._cudss_module()
    if isnothing(work.solver) || matrix.colptr != work.colptr || matrix.rowval != work.rowval
        release_continuous_interior!(work)
        work.matrix = cuda.CUSPARSE.CuSparseMatrixCSR(matrix)
        work.solver = cudss.CudssSolver(work.matrix, "G", 'F')
        work.rhs = cuda.zeros(ComplexF64, size(matrix, 1))
        work.solution = similar(work.rhs)
        cudss.cudss("analysis", work.solver, work.solution, work.rhs; asynchronous=false)
        work.colptr = copy(matrix.colptr)
        work.rowval = copy(matrix.rowval)
    else
        # Preserve CSR ordering and the pointers bound to the cuDSS descriptor.
        updated = cuda.CUSPARSE.CuSparseMatrixCSR(matrix)
        try
            copyto!(work.matrix.nzVal, updated.nzVal)
        finally
            BeatEngineCoupled._unsafe_free_cuda_fields!(updated, (:rowPtr, :colVal, :nzVal))
        end
    end
    cudss.cudss("factorization", work.solver, work.solution, work.rhs; asynchronous=false)
    work.frequency_hz = frequency_hz
    return work
end

continuous_interior_solve(factor, rhs) = factor \ rhs

function continuous_interior_solve(work::ContinuousCudaInterior, rhs)
    cuda = BeatEngineCore.cuda_module()
    cudss = BeatEngineCoupled._cudss_module()
    result = Matrix{ComplexF64}(undef, size(rhs))
    # The external-source audit can supply hundreds of interface loads.
    # Bound device scratch to 32 pairs of RHS/solution columns at a time.
    for first_col in 1:32:size(rhs, 2)
        columns = first_col:min(first_col + 31, size(rhs, 2))
        device_rhs = cuda.CuArray(Matrix{ComplexF64}(view(rhs, :, columns)))
        device_solution = nothing
        try
            device_solution = similar(device_rhs)
            cudss.cudss("solve", work.solver, device_solution, device_rhs; asynchronous=false)
            result[:, columns] = Array(device_solution)
        finally
            isnothing(device_solution) || cuda.unsafe_free!(device_solution)
            cuda.unsafe_free!(device_rhs)
        end
    end
    return result
end
