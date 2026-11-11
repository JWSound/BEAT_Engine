# Dense LU library switch: Julia's BLAS/LAPACK (OpenBLAS) or Apple Accelerate's ILP64 new-LAPACK
# getrf (macOS 13.3+). Included by BeatEngineCore before the dense solvers.
#
# `BLAB_DENSE_LU_LIBRARY`:
#   blas        (default) Julia's LAPACK, as before.
#   accelerate  Accelerate's getrf for host Float32/Float64/ComplexF32/ComplexF64 matrices; an error
#               when the entry point is missing.
#   auto        Accelerate only on chips with SME (M4 and later, `hw.optional.arm.FEAT_SME`), where
#               Accelerate uses the matrix unit that OpenBLAS cannot reach; Julia's LAPACK elsewhere.
# On an M1 Max (no SME) Accelerate's cgetrf measured 340-500 GFLOP/s against OpenBLAS's
# 520-570 GFLOP/s at n = 4096-24576, hence the default. Only the factorization changes library;
# the factors are an ordinary `LU` and every solve with them uses Julia's LAPACK.

const _ACCELERATE_GETRF = Dict{Symbol,Ptr{Cvoid}}()
const _ACCELERATE_GETRF_PID = Ref{Int}(0)
const _ACCELERATE_GETRF_LOCK = ReentrantLock()
const _SME_PRESENT = Ref{Tuple{Int,Bool}}((0, false))

_getrf_name(::Type{Float32}) = :sgetrf
_getrf_name(::Type{Float64}) = :dgetrf
_getrf_name(::Type{ComplexF32}) = :cgetrf
_getrf_name(::Type{ComplexF64}) = :zgetrf

# Pointers are looked up per process (a package image may carry stale ones from precompilation).
function _accelerate_getrf(::Type{T}) where {T}
    lock(_ACCELERATE_GETRF_LOCK) do
        if _ACCELERATE_GETRF_PID[] != getpid()
            empty!(_ACCELERATE_GETRF)
            if Sys.isapple() && Sys.ARCH === :aarch64
                handle = Base.Libc.Libdl.dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate";
                    throw_error=false)
                if handle !== nothing
                    for S in (Float32, Float64, ComplexF32, ComplexF64)
                        name = string(_getrf_name(S), "\$NEWLAPACK\$ILP64")
                        symbol = Base.Libc.Libdl.dlsym(handle, name; throw_error=false)
                        symbol === nothing || (_ACCELERATE_GETRF[_getrf_name(S)] = symbol)
                    end
                end
            end
            _ACCELERATE_GETRF_PID[] = getpid()
        end
        return get(_ACCELERATE_GETRF, _getrf_name(T), C_NULL)
    end
end

function _sme_present()
    cached = _SME_PRESENT[]
    cached[1] == getpid() && return cached[2]
    present = false
    if Sys.isapple() && Sys.ARCH === :aarch64
        try
            present = strip(read(`sysctl -n hw.optional.arm.FEAT_SME`, String)) == "1"
        catch
            present = false
        end
    end
    _SME_PRESENT[] = (getpid(), present)
    return present
end

function _dense_lu_library_setting()
    raw = lowercase(strip(get(ENV, "BLAB_DENSE_LU_LIBRARY", "blas")))
    raw == "openblas" && return "blas"
    raw in ("blas", "accelerate", "auto") ||
        error("Unsupported BLAB_DENSE_LU_LIBRARY value: $raw. Expected blas, accelerate, or auto.")
    return raw
end

"""
`"blas"` or `"accelerate"`: the library `dense_lu!` uses for a host matrix of element type `T`.
Throws for an invalid setting or for `accelerate` without Accelerate's entry point.
"""
function dense_lu_library(::Type{T}=ComplexF32) where {T}
    setting = _dense_lu_library_setting()
    setting == "blas" && return "blas"
    T <: Union{Float32,Float64,ComplexF32,ComplexF64} || return "blas"
    available = _accelerate_getrf(T) != C_NULL
    if setting == "accelerate"
        available || error("BLAB_DENSE_LU_LIBRARY=accelerate, but Accelerate's new-LAPACK getrf is not available.")
        return "accelerate"
    end
    return available && _sme_present() ? "accelerate" : "blas"
end

"""
A diagnostic string for the dense LU library; never throws (`"unavailable: <reason>"`).
"""
function dense_lu_library_diagnostic(::Type{T}=ComplexF32) where {T}
    try
        return dense_lu_library(T)
    catch exception
        return "unavailable: " * sprint(showerror, exception)
    end
end

"""
    dense_lu!(A; check=true) -> LU

`lu!(A; check)` with the library chosen by `BLAB_DENSE_LU_LIBRARY` (see the file header). Matrices
that are not host `Matrix{<:BlasFloat}` always use `lu!`.
"""
dense_lu!(A::AbstractMatrix; check::Bool=true) = lu!(A; check=check)

function dense_lu!(A::Matrix{T}; check::Bool=true) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    dense_lu_library(T) == "accelerate" || return lu!(A; check=check)
    symbol = _accelerate_getrf(T)
    m, n = size(A)
    ipiv = Vector{Int64}(undef, min(m, n))
    info = Ref{Int64}(0)
    ccall(symbol, Cvoid, (Ref{Int64}, Ref{Int64}, Ptr{T}, Ref{Int64}, Ptr{Int64}, Ref{Int64}),
          m, n, A, max(1, stride(A, 2)), ipiv, info)
    info[] < 0 && throw(ArgumentError("Accelerate getrf: invalid argument $(-info[])"))
    factorization = LinearAlgebra.LU{T,Matrix{T},Vector{Int64}}(A, ipiv, LinearAlgebra.BlasInt(info[]))
    check && LinearAlgebra.checknonsingular(info[])
    return factorization
end
