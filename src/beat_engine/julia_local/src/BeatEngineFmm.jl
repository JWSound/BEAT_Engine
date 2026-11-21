# Fast-multipole Burton-Miller operator (exterior, rigid/Neumann) for the CPU backend.
#
# The dense CPU assembly (BeatEngineCpuBurtonMiller.jl) builds
#
#   lhs = 0.5 I_p1p1 - D + (i/k) H          rhs = (-S - (i/k)(K' + 0.5 I)) q
#
# from two kinds of element pairs: every non-adjacent pair with one fixed regular
# triangle rule, and every adjacent pair (shared vertex, edge or face) with Duffy rules.
# Image pairs (symmetry planes) use the regular rule for all pairs plus a Duffy-minus-regular
# delta for the pairs that touch their own image.
#
# This module reproduces exactly that matrix without forming it:
#
#   lhs * x = FMM(regular rule, all pairs, images included)              (far + near, O(N))
#           + C * x                                                       (sparse, adjacent pairs)
#           + 0.5 I_p1p1 * x,       rows scaled by the symmetry orbit weights
#
# where C = Duffy(adjacent) - regular(adjacent) + image deltas. The FMM sums the
# regular-rule quadrature over source points (trial quadrature points) at target points
# (test quadrature points); coincident points (the self pair) are skipped by FMM3D exactly
# as `radius == 0 && continue` skips them in the dense kernel, so subtracting the regular
# adjacent blocks computed by the dense kernel cancels them exactly.
#
# Kernel split of one regular pair at quadrature points x (test e) and y (trial f), weights
# W = w_q w_p (2A_e)(2A_f):
#
#   -D + eta H  ->  sum W [ phi_i(x) phi_j(y) ( -dG/dn_y - eta k^2 (n_e . n_f) G ) ]
#                 + eta curl_i^e . curl_j^f  sum W G
#
# so with u = sum_j x_j phi_j (P1 field) and c_f = sum_j x_j curl_j^f (constant per element):
#
#   dipole density   W_p u(y_p) n_f            -> potential D(x) = sum dG/dn_y u
#   charge densities W_p u(y_p) n_f  (3)       -> N(x),  used as n_e . N(x)
#   charge densities W_p c_f         (3)       -> C(x),  used as curl_i^e . C(x)
#
# FMM3D's Helmholtz kernel is exp(ikr)/r without 1/(4pi); its dipole convention is checked
# numerically (`fmm_kernel_factors`).

module BeatEngineFmm

using LinearAlgebra, SparseArrays, StaticArrays
using FMM3D, Krylov, LinearMaps
using ..BeatEngineCore
const BC = BeatEngineCore

export FmmCoupledSolve, fmm_bordered_preconditioner, fmm_coupled_preconditioner, FmmCoupledPreconditioner, fmm_near_blocks, fmm_hfmm3d_checked, fmm_near_preconditioner, FmmBurtonMillerOperator, fmm_burton_miller_operator, fmm_lhs_mul!, fmm_lhs_mul, fmm_rhs

struct FmmBurtonMillerOperator{T<:AbstractFloat}
    k::T
    coupling::Complex{T}
    eps::Float64
    n_p1::Int
    # targets: test quadrature points of every element (element-major)
    targets::Matrix{Float64}
    nq::Int
    elements::Vector{BC.BeatCpuElementData{T}}
    basis::Vector{SVector{3,T}}               # P1 values at the rule points
    weights::Vector{T}                         # rule weights
    # sources: the same points plus their images
    sources::Matrix{Float64}
    source_elements::Vector{BC.BeatCpuElementData{T}}   # identity elements then each image set
    correction::SparseMatrixCSC{Complex{T},Int}
    near::SparseMatrixCSC{Complex{T},Int}
    rhs_near::SparseMatrixCSC{Complex{T},Int}
    rhs_correction::SparseMatrixCSC{Complex{T},Int}
    identity_p1_p1::SparseMatrixCSC{T,Int}
    identity_p1_dp0::SparseMatrixCSC{T,Int}
    row_weights::Vector{T}
    charge_factor::Float64
    dipole_factor::Float64
end

"""
    fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1, singular_order=2,
                               symmetry_mode=:off, eps=1e-6)

Precompute the near-field correction and the point sets. Equivalent to the matrix of
`assemble_burton_miller_neumann_system_cpu` with the same arguments.
"""
function fmm_burton_miller_operator(mesh::BC.BoundaryMesh{T}, p1, dp0, k::T, rule;
        identity_p1_p1, identity_p1_dp0, singular_order::Int=2, symmetry_mode::Symbol=:off,
        eps::Float64=1e-6, far_rule=rule, near_factor::Real=2.0) where {T}
    k = BC.outgoing_wavenumber(k)
    coupling = Complex{T}(0, 1) / k
    symmetry_mode = BC.normalized_symmetry_mode(symmetry_mode)
    elements = BC._beat_cpu_element_data(mesh, p1, dp0)
    quad = BC._beat_cpu_regular_quadrature_data(mesh, rule)          # the dense build's rule
    fquad = BC._beat_cpu_regular_quadrature_data(mesh, far_rule)     # what the FMM sums
    same_rule = far_rule.points == rule.points && far_rule.weights == rule.weights
    nq = length(far_rule.weights)
    ne = length(elements)
    targets = Matrix{Float64}(undef, 3, ne * nq)
    for e in 1:ne, q in 1:nq
        targets[:, (e - 1) * nq + q] .= fquad[e].points[q]
    end
    transforms = BC.symmetry_image_transforms(symmetry_mode)
    source_elements = copy(elements)
    sources = copy(targets)
    image_elements = Vector{Vector{BC.BeatCpuElementData{T}}}()
    image_quads = Vector{Vector{BC.BeatCpuRegularQuadratureData{T}}}()
    image_fquads = Vector{Vector{BC.BeatCpuRegularQuadratureData{T}}}()
    for transform in transforms
        ie = BC._beat_cpu_reflect_element_data(elements, transform)
        iq = BC._beat_cpu_reflect_regular_quadrature_data(quad, transform)
        ifq = BC._beat_cpu_reflect_regular_quadrature_data(fquad, transform)
        push!(image_elements, ie); push!(image_quads, iq); push!(image_fquads, ifq)
        append!(source_elements, ie)
        block = Matrix{Float64}(undef, 3, ne * nq)
        for e in 1:ne, q in 1:nq
            block[:, (e - 1) * nq + q] .= ifq[e].points[q]
        end
        sources = hcat(sources, block)
    end

    # correction = (dense value) - (what the FMM adds) on every near pair; `near` keeps the
    # dense value alone for the preconditioner. Both COO.
    I = Int[]; J = Int[]; V = Complex{T}[]
    RI = Int[]; RJ = Int[]; RV = Complex{T}[]
    NI = Int[]; NJ = Int[]; NV = Complex{T}[]
    NRI = Int[]; NRJ = Int[]; NRV = Complex{T}[]
    function push_block!(test, trial, block, sign, rblock; to_near::Bool=false)
        for a in 1:3, b in 1:3
            push!(I, test.p1_dofs[a]); push!(J, trial.p1_dofs[b]); push!(V, sign * block[a, b])
            if to_near
                push!(NI, test.p1_dofs[a]); push!(NJ, trial.p1_dofs[b]); push!(NV, sign * block[a, b])
            end
        end
        for a in 1:3
            push!(RI, test.p1_dofs[a]); push!(RJ, trial.dp0_dof); push!(RV, sign * rblock[a])
            if to_near
                push!(NRI, test.p1_dofs[a]); push!(NRJ, trial.dp0_dof); push!(NRV, sign * rblock[a])
            end
        end
    end
    push_near_only!(test, trial, block, rblock) = begin
        for a in 1:3, b in 1:3
            push!(NI, test.p1_dofs[a]); push!(NJ, trial.p1_dofs[b]); push!(NV, block[a, b])
        end
        for a in 1:3
            push!(NRI, test.p1_dofs[a]); push!(NRJ, trial.dp0_dof); push!(NRV, rblock[a])
        end
    end
    regular(test, trial, qt, qs) = BC._beat_cpu_bm_regular_pair_blocks(
        test, trial, qt, qs, dot(test.normal, trial.normal), T(4.0) * test.area * trial.area, k, coupling)

    # adjacent pairs: + Duffy
    cache = BC.build_singular_correction_cache(mesh, singular_order, collect(eachindex(mesh.faces)))
    for pair in cache.pairs
        test = elements[pair.test_index]; trial = elements[pair.trial_index]
        duffy = cache.rules[pair.rule_index]
        lhs_block, rhs_block = BC._beat_cpu_bm_pair_blocks(
            test.vertices, trial.vertices, test.normal, trial.normal, test.curls, trial.curls,
            pair.normal_product, pair.jac_scale, k, coupling,
            duffy.test_points, duffy.trial_points, duffy.weights)
        push_block!(test, trial, lhs_block, one(T), rhs_block; to_near=true)
    end

    # near-pair search on element centroids (uniform grid, cell = near radius)
    hmax = maximum(maximum(norm(v[i] - v[j]) for (i, j) in ((1, 2), (2, 3), (3, 1))) for v in mesh.face_vertices)
    radius = T(near_factor) * hmax
    cell(x) = (floor(Int, x[1] / radius), floor(Int, x[2] / radius), floor(Int, x[3] / radius))
    grid = Dict{NTuple{3,Int},Vector{Int}}()
    for (f, c) in enumerate(mesh.centroids)
        push!(get!(grid, cell(c), Int[]), f)
    end
    function near_of(point)
        out = Int[]
        c0 = cell(point)
        for dx in -1:1, dy in -1:1, dz in -1:1
            for f in get(grid, (c0[1] + dx, c0[2] + dy, c0[3] + dz), Int[])
                norm(mesh.centroids[f] - point) <= radius && push!(out, f)
            end
        end
        return out
    end

    # identity pairs: adjacent -> - far value; near non-adjacent -> + exact - far
    for e in 1:ne, f in near_of(mesh.centroids[e])
        adjacent = BC.elements_are_adjacent(elements[e].face, elements[f].face)
        if !adjacent
            if !same_rule
                ex, rex = regular(elements[e], elements[f], quad[e], quad[f])
                push_block!(elements[e], elements[f], ex, one(T), rex; to_near=true)
            else
                ex, rex = regular(elements[e], elements[f], quad[e], quad[f])
                push_near_only!(elements[e], elements[f], ex, rex)
                continue
            end
        end
        fa, rfa = regular(elements[e], elements[f], fquad[e], fquad[f])
        push_block!(elements[e], elements[f], fa, -one(T), rfa)
    end

    # image pairs: the dense build adds the exact regular value on every pair and a
    # Duffy - exact delta on the singular ones; the FMM adds the far value everywhere.
    for (ti, transform) in enumerate(transforms)
        for e in 1:ne, f in near_of(BC.reflect_point(transform, mesh.centroids[e]))
            test = elements[e]; trial = image_elements[ti][f]
            ex, rex = regular(test, trial, quad[e], image_quads[ti][f])
            if same_rule
                push_near_only!(test, trial, ex, rex)
            else
                push_block!(test, trial, ex, one(T), rex; to_near=true)
                fa, rfa = regular(test, trial, fquad[e], image_fquads[ti][f])
                push_block!(test, trial, fa, -one(T), rfa)
            end
        end
        icache = BC._beat_cpu_image_singular_cache(mesh, singular_order, collect(eachindex(mesh.faces)), transform)
        icache.pair_count == 0 && continue
        for pairs in icache.pairs_by_test, pair in pairs
            test = elements[pair.test_index]; trial = image_elements[ti][pair.trial_index]
            duffy = icache.rules[pair.rule_index]
            lhs_block, rhs_block = BC._beat_cpu_bm_pair_blocks(
                test.vertices, trial.vertices, test.normal, trial.normal, test.curls, trial.curls,
                pair.normal_product, pair.jac_scale, k, coupling,
                duffy.test_points, duffy.trial_points, duffy.weights)
            push_block!(test, trial, lhs_block, one(T), rhs_block; to_near=true)
            reg, rreg = regular(test, trial, quad[pair.test_index], image_quads[ti][pair.trial_index])
            push_block!(test, trial, reg, -one(T), rreg; to_near=true)
        end
    end
    n_p1 = p1.global_dof_count
    correction = sparse(I, J, V, n_p1, n_p1)
    near = sparse(NI, NJ, NV, n_p1, n_p1)
    rhs_near = sparse(NRI, NRJ, NRV, n_p1, dp0.global_dof_count)
    rhs_correction = sparse(RI, RJ, RV, n_p1, dp0.global_dof_count)
    row_weights = T.(BC.p1_symmetry_orbit_weights(mesh, symmetry_mode))
    return FmmBurtonMillerOperator{T}(k, coupling, eps, n_p1, targets, nq, elements,
        [SVector{3,T}(BC.p1_values(p)) for p in far_rule.points], T.(far_rule.weights),
        sources, source_elements, correction, near, rhs_near, rhs_correction, SparseMatrixCSC{T,Int}(identity_p1_p1),
        SparseMatrixCSC{T,Int}(identity_p1_dp0),
        row_weights, fmm_kernel_factors()...)
end

p1_values_rule(p) = BC.p1_values(p)

"""
    fmm_kernel_factors() -> (charge, dipole)

Real factors s with FMM3D charge potential = s_c * exp(ik|x-y|)/|x-y| and dipole potential
= s_d * v . grad_y exp(ik|x-y|)/|x-y|, measured on the direct evaluator so a convention
change in the library cannot pass silently.
"""
function fmm_kernel_factors()
    src = reshape([0.1, -0.2, 0.05], 3, 1)
    trg = reshape([0.7, 0.3, -0.4], 3, 1)
    v = [0.3, -0.5, 0.8]
    k = 2.3
    r = trg[:, 1] - src[:, 1]; rr = norm(r)
    g = exp(im * k * rr) / rr
    grad_y = -(im * k - 1 / rr) * g * (r / rr)
    oc = h3ddir(complex(k), src, trg; charges=ComplexF64[1.0], pgt=1)
    od = h3ddir(complex(k), src, trg; dipvecs=reshape(complex.(v), 3, 1), pgt=1)
    sc = oc.pottarg[1] / g
    sd = od.pottarg[1] / dot(v, grad_y)
    (abs(imag(sc)) < 1e-10 && abs(imag(sd)) < 1e-10 && abs(real(sc) - real(sd)) < 1e-10) ||
        error("unexpected FMM3D kernel convention (charge $sc, dipole $sd)")
    return real(sc), real(sd)
end


"""
    fmm_hfmm3d_checked(eps, k, sources, targets; charges=nothing, dipvecs=nothing, nd=1, pgt=1)
        -> (pot, grad)

FMM3D's `hfmm3d` with the library's error flag actually returned: the Julia wrapper passes
`ier` by value, so an allocation failure in the plane-wave workspace (large k*box at high
density count) silently returns garbage. This calls the same Fortran entry point with a
`Ref` and throws `FmmAllocationError` when `ier != 0`, so callers can split the densities.
"""
struct FmmAllocationError <: Exception
    ier::Int
    nd::Int
end

function fmm_hfmm3d_checked(eps::Float64, k::Real, sources::Matrix{Float64}, targets::Matrix{Float64};
        charges=nothing, dipvecs=nothing, nd::Int=1, pgt::Int=1, ndiv::Int=_fmm_ndiv())
    n = size(sources, 2); nt = size(targets, 2)
    ifcharge = charges === nothing ? 0 : 1
    ifdipole = dipvecs === nothing ? 0 : 1
    ch = charges === nothing ? zeros(ComplexF64, 1) : ComplexF64.(charges)
    dv = dipvecs === nothing ? zeros(ComplexF64, 1) : ComplexF64.(dipvecs)
    pottarg = zeros(ComplexF64, nd, nt)
    gradtarg = pgt >= 2 ? zeros(ComplexF64, nd, 3, nt) : zeros(ComplexF64, 1)
    hesstarg = zeros(ComplexF64, 1)
    pot = zeros(ComplexF64, 1); grad = zeros(ComplexF64, 1); hess = zeros(ComplexF64, 1)
    ier = Ref{Int64}(0)
    if ndiv <= 0
        ccall((:hfmm3d_, FMM3D.libfmm3d), Cvoid,
            (Ref{Int64}, Ref{Float64}, Ref{ComplexF64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{ComplexF64},
             Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ref{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64},
             Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Int64}),
            nd, eps, ComplexF64(k), n, sources, ifcharge, ch, ifdipole, dv, 0, 0, pot, grad, hess,
            nt, targets, pgt, pottarg, gradtarg, hesstarg, ier)
    else
        timeinfo = zeros(Float64, 6)
        ccall((:hfmm3d_ndiv_, FMM3D.libfmm3d), Cvoid,
            (Ref{Int64}, Ref{Float64}, Ref{ComplexF64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{ComplexF64},
             Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ref{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64},
             Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64},
             Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}),
            nd, eps, ComplexF64(k), n, sources, ifcharge, ch, ifdipole, dv, 0, 0, pot, grad, hess,
            nt, targets, pgt, pottarg, gradtarg, hesstarg, ndiv, 0, 1, timeinfo, ier)
    end
    ier[] == 0 || throw(FmmAllocationError(Int(ier[]), nd))
    return pottarg, gradtarg
end

"""Leaf size for FMM3D's tree; 0 = the library's own choice. BLAB_FMM_NDIV overrides."""
_fmm_ndiv() = parse(Int, get(ENV, "BLAB_FMM_NDIV", "0"))

# Combining densities into one tree needs nd times the plane-wave workspace (~1.6 GB per
# density for 134k points at kD ~ 60). When it does not fit, the allocation can be fatal (cgroup
# OOM kill) rather than a clean `ier`, so the combined call is opt-in: BLAB_FMM_COMBINE=1.
_fmm_combine() = get(ENV, "BLAB_FMM_COMBINE", "0") == "1"

function _fmm_potentials(eps, k, sources, targets; charges=nothing, dipvecs=nothing, nd::Int, pgt::Int=1)
    try
        (nd == 1 || _fmm_combine()) || throw(FmmAllocationError(0, nd))
        return fmm_hfmm3d_checked(eps, k, sources, targets; charges, dipvecs, nd, pgt)
    catch err
        (err isa FmmAllocationError && nd > 1) || rethrow()
        pot = zeros(ComplexF64, nd, size(targets, 2))
        grad = pgt >= 2 ? zeros(ComplexF64, nd, 3, size(targets, 2)) : zeros(ComplexF64, 1)
        for d in 1:nd
            c = charges === nothing ? nothing : reshape(charges[d, :], 1, :)
            v = dipvecs === nothing ? nothing : reshape(dipvecs[d, :, :], 1, 3, :)
            c !== nothing && iszero(c) && (c = nothing)
            v !== nothing && iszero(v) && (v = nothing)
            c === nothing && v === nothing && continue
            p, g = fmm_hfmm3d_checked(eps, k, sources, targets; charges=c, dipvecs=v, nd=1, pgt)
            pot[d, :] .= vec(p)
            pgt >= 2 && (grad[d, :, :] .= reshape(g, 3, :))
        end
        return pot, grad
    end
end

function fmm_lhs_mul(op::FmmBurtonMillerOperator{T}, x::AbstractVector) where {T}
    y = zeros(Complex{T}, op.n_p1)
    fmm_lhs_mul!(y, op, x)
    return y
end

function fmm_lhs_mul!(y::AbstractVector, op::FmmBurtonMillerOperator{T}, x::AbstractVector) where {T}
    nq = op.nq
    ns = size(op.sources, 2)
    dip = zeros(ComplexF64, 3, ns)
    chg = zeros(ComplexF64, 6, ns)
    for (s_el, el) in enumerate(op.source_elements)
        jac = 2 * el.area
        xs = (x[el.p1_dofs[1]], x[el.p1_dofs[2]], x[el.p1_dofs[3]])
        c = el.curls[1] * xs[1] + el.curls[2] * xs[2] + el.curls[3] * xs[3]
        for q in 1:nq
            col = (s_el - 1) * nq + q
            b = op.basis[q]
            W = op.weights[q] * jac
            u = W * (b[1] * xs[1] + b[2] * xs[2] + b[3] * xs[3])
            for d in 1:3
                dip[d, col] = u * el.normal[d]
                chg[d, col] = u * el.normal[d]
                chg[3 + d, col] = W * c[d]
            end
        end
    end
    # seven densities in one tree: rows 1-6 charges (normal and curl vectors), row 7 dipole
    charges = zeros(ComplexF64, 7, ns); charges[1:6, :] .= chg
    dipvecs = zeros(ComplexF64, 7, 3, ns); dipvecs[7, :, :] .= dip
    pot, _ = _fmm_potentials(op.eps, op.k, op.sources, op.targets; charges, dipvecs, nd=7)
    Cpot = pot[1:6, :] ./ (4pi * op.charge_factor)
    Dpot = pot[7, :] ./ (4pi * op.dipole_factor)
    eta = op.coupling
    k2 = op.k^2
    fill!(y, zero(eltype(y)))
    for (e, el) in enumerate(op.elements)
        jac = 2 * el.area
        curl_acc = zero(SVector{3,ComplexF64})
        for q in 1:nq
            col = (e - 1) * nq + q
            W = op.weights[q] * jac
            nN = el.normal[1] * Cpot[1, col] + el.normal[2] * Cpot[2, col] + el.normal[3] * Cpot[3, col]
            val = W * (-Dpot[col] - eta * k2 * nN)
            b = op.basis[q]
            for a in 1:3
                y[el.p1_dofs[a]] += b[a] * val
            end
            curl_acc += W * SVector(Cpot[4, col], Cpot[5, col], Cpot[6, col])
        end
        for a in 1:3
            y[el.p1_dofs[a]] += eta * dot(el.curls[a], curl_acc)
        end
    end
    y .+= op.correction * x
    y .*= op.row_weights
    y .+= Complex{T}(0.5) .* (op.identity_p1_p1 * x)
    return y
end

"""
    fmm_rhs(op, q) -> rhs   (n_p1 x drives)

`(-S - eta (K' + 0.5 I)) q` for DP0 Neumann data `q` (dp0 dofs x drives), equal to the
dense assembly's `rhs`.
"""
function fmm_rhs(op::FmmBurtonMillerOperator{T}, q::AbstractMatrix) where {T}
    nq = op.nq
    ns = size(op.sources, 2)
    nd = size(q, 2)
    chg = zeros(ComplexF64, nd, ns)
    for (s_el, el) in enumerate(op.source_elements)
        jac = 2 * el.area
        for qq in 1:nq
            col = (s_el - 1) * nq + qq
            W = op.weights[qq] * jac
            for d in 1:nd
                chg[d, col] = W * q[el.dp0_dof, d]
            end
        end
    end
    p, g = _fmm_potentials(op.eps, op.k, op.sources, op.targets; charges=chg, nd=nd, pgt=2)
    pot = p ./ (4pi * op.charge_factor)
    grad = g ./ (4pi * op.charge_factor)
    eta = op.coupling
    out = zeros(Complex{T}, op.n_p1, nd)
    for (e, el) in enumerate(op.elements)
        jac = 2 * el.area
        for qq in 1:nq
            col = (e - 1) * nq + qq
            W = op.weights[qq] * jac
            b = op.basis[qq]
            for d in 1:nd
                dn = el.normal[1] * grad[d, 1, col] + el.normal[2] * grad[d, 2, col] + el.normal[3] * grad[d, 3, col]
                val = W * (-pot[d, col] - eta * dn)
                for a in 1:3
                    out[el.p1_dofs[a], d] += b[a] * val
                end
            end
        end
    end
    out .+= op.rhs_correction * q
    out .*= op.row_weights
    out .-= (Complex{T}(0.5) * eta) .* (op.identity_p1_dp0 * q)
    return out
end

"""
    fmm_near_preconditioner(op) -> factorization

Sparse LU of the adjacent-pair (Duffy) part of the operator plus the identity term, with the
same row weights: the dense matrix restricted to the element-adjacency pattern.
"""
function fmm_near_preconditioner(op::FmmBurtonMillerOperator{T}) where {T}
    P = Diagonal(op.row_weights) * op.near + Complex{T}(0.5) * op.identity_p1_p1
    return lu(P)
end

"""Near-field approximations of the two BEM blocks, with row weights and identity terms as in
the dense build: A ~ `near_lhs`, (-S - eta(K' + 0.5 I)) ~ `near_rhs`."""
function fmm_near_blocks(op::FmmBurtonMillerOperator{T}) where {T}
    W = Diagonal(op.row_weights)
    A = W * op.near + Complex{T}(0.5) * op.identity_p1_p1
    R = W * op.rhs_near - (Complex{T}(0.5) * op.coupling) * op.identity_p1_dp0
    return A, R
end

"""
    FmmCoupledPreconditioner

Exact inverse of the near-field coupled block system

    [ S     0     -L      ] [p_G]   [r_G]
    [ 0     A_n   -R_n F  ] [p_B] = [r_B]
    [ T_G  -T_B    0      ] [q  ]   [r_Q]

(S the dense FEM Schur complement, L the FEM load, A_n / R_n the near-field BEM blocks, F the
DP0 flux map, T the traces) by block elimination, so the dense Schur block never enters a
sparse factorization: one sparse LU of A_n (size of the BEM), one dense LU of S (|Γ|), one dense
LU of the |Q| x |Q| interface system

    Z = T_G S^-1 L - T_B A_n^-1 R_n F,

whose second term is formed in column blocks so no N_B x |Q| dense array is held.
"""
struct FmmCoupledPreconditioner{FA,FS,FZ,TL,TR,TG,TB}
    fa::FA
    fs::FS
    fz::FZ
    L::TL
    RnF::TR
    TGm::TG
    TBm::TB
    G::UnitRange{Int}
    B::UnitRange{Int}
    Q::UnitRange{Int}
end

function fmm_coupled_preconditioner(An, RnF, S, L, TGm, TBm, G, B, Q; block::Int=256)
    fa = lu(An)
    fs = lu(Matrix(S))
    nq = size(RnF, 2)
    K = zeros(ComplexF64, size(TBm, 1), nq)
    for c0 in 1:block:nq
        cols = c0:min(c0 + block - 1, nq)
        X = fa \ Matrix(RnF[:, cols])
        K[:, cols] = TBm * X
    end
    Z = TGm * (fs \ Matrix(L)) - K
    return FmmCoupledPreconditioner(fa, fs, lu(Z), L, RnF, TGm, TBm, G, B, Q)
end

function Base.:\(P::FmmCoupledPreconditioner, r::AbstractVector)
    rG = r[P.G]; rB = r[P.B]; rQ = r[P.Q]
    y = P.fa \ rB
    q = P.fz \ (rQ + P.TBm * y - P.TGm * (P.fs \ rG))
    pG = P.fs \ (rG + P.L * q)
    pB = y + P.fa \ (P.RnF * q)
    out = similar(r, ComplexF64)
    out[P.G] = pG; out[P.B] = pB; out[P.Q] = q
    return out
end

"""
    fmm_bordered_preconditioner(An, PBS, PSB, PSS, B, S2; block=256)

Exact inverse of the near-field coupled system in bordered form

    [ A_n   P_BS ] [x_B ]   [r_B ]
    [ P_SB  P_SS ] [x_S2] = [r_S2]

where B are the BEM unknowns and S2 everything else (FEM interface pressures, interface flux,
transducer motion and current). Works for any layout and any transducer count: the Schur
complement Z = P_SS - P_SB A_n^-1 P_BS is dense |S2| x |S2|, A_n^-1 P_BS is formed in column blocks.
Memory: 16 |S2|^2 bytes (|S2| = 8k -> 1 GB, 16k -> 4 GB).
"""
struct FmmBorderedPreconditioner{FA,FZ,TBS,TSB}
    fa::FA
    fz::FZ
    PBS::TBS
    PSB::TSB
    B::Vector{Int}
    S2::Vector{Int}
end

function fmm_bordered_preconditioner(An, PBS, PSB, PSS, B, S2; block::Int=256)
    fa = lu(An)
    m = length(S2)
    Z = Matrix{ComplexF64}(PSS)
    for c0 in 1:block:m
        cols = c0:min(c0 + block - 1, m)
        rhs = Matrix(PBS[:, cols])
        iszero(rhs) && continue
        Z[:, cols] .-= PSB * (fa \ rhs)
    end
    return FmmBorderedPreconditioner(fa, lu(Z), PBS, PSB, collect(B), collect(S2))
end

function Base.:\(P::FmmBorderedPreconditioner, r::AbstractVector)
    rB = r[P.B]; rS = r[P.S2]
    y = P.fa \ rB
    xS = P.fz \ (rS - P.PSB * y)
    xB = y - P.fa \ (P.PBS * xS)
    out = similar(r, ComplexF64)
    out[P.B] = xB; out[P.S2] = xS
    return out
end

"""
    FmmCoupledSolve(op, rest, B, Q, M, flux, motion_flux; rtol=1e-8, maxiter=500)

Drop-in for the coupled system's `factorization`: `size(S, 1)` and `S \\ rhs` (vector or matrix of
right-hand sides). The operator is `rest * x` plus, on the BEM rows B,
`A x_B + C (flux x_Q + motion_flux x_M)` from the FMM operator. Each column is solved by
row-equilibrated, right-preconditioned GMRES (`atol = 0`) with the bordered near-field
preconditioner; per-column iteration counts and residuals are kept in `stats`.
"""
mutable struct FmmCoupledSolve
    op::FmmBurtonMillerOperator
    rest::SparseMatrixCSC{ComplexF64,Int}
    B::Vector{Int}
    Q::Vector{Int}
    M::Vector{Int}
    flux::SparseMatrixCSC{ComplexF64,Int}
    motion_flux::SparseMatrixCSC{ComplexF64,Int}
    precond::FmmBorderedPreconditioner
    row_scale::Vector{Float64}
    rtol::Float64
    maxiter::Int
    stats::Vector{NamedTuple}
end

Base.size(S::FmmCoupledSolve, d::Integer) = size(S.rest, d)
Base.size(S::FmmCoupledSolve) = size(S.rest)

function FmmCoupledSolve(op::FmmBurtonMillerOperator, rest::AbstractMatrix, B, Q, M, flux, motion_flux;
        rtol::Float64=1e-8, maxiter::Int=500)
    n = size(rest, 1)
    B = collect(B); Q = collect(Q); M = collect(M)
    rest = sparse(ComplexF64.(rest))
    flux = sparse(ComplexF64.(flux)); motion_flux = sparse(ComplexF64.(motion_flux))
    An, Rn = fmm_near_blocks(op)
    S2 = setdiff(1:n, B)
    pos = Dict(j => i for (i, j) in enumerate(S2))
    PBS = spzeros(ComplexF64, length(B), length(S2))
    isempty(Q) || (PBS[:, [pos[j] for j in Q]] = -(Rn * flux))
    isempty(M) || (PBS[:, [pos[j] for j in M]] = -(Rn * motion_flux))
    PSB = rest[S2, B]; PSS = Matrix(rest[S2, S2])
    precond = fmm_bordered_preconditioner(An, PBS, PSB, PSS, B, S2)
    # row norms of the near system (what the preconditioner sees)
    row_scale = zeros(n)
    rows_rest = vec(sqrt.(sum(abs2, rest; dims=2)))
    row_scale .= rows_rest
    nearB = vec(sqrt.(sum(abs2, An; dims=2) .+ sum(abs2, PBS; dims=2)))
    row_scale[B] .= nearB
    row_scale .= 1 ./ max.(row_scale, floatmin(Float64))
    return FmmCoupledSolve(op, rest, B, Q, M, flux, motion_flux, precond, row_scale, rtol, maxiter, NamedTuple[])
end

function _fmm_coupled_mul(S::FmmCoupledSolve, x::AbstractVector)
    y = S.rest * x
    data = isempty(S.Q) ? zeros(ComplexF64, size(S.flux, 1)) : S.flux * x[S.Q]
    isempty(S.M) || (data .+= S.motion_flux * x[S.M])
    y[S.B] .+= fmm_lhs_mul(S.op, x[S.B]) .- fmm_rhs(S.op, reshape(data, :, 1))[:, 1]
    return y
end

function Base.:\(S::FmmCoupledSolve, rhs::AbstractVecOrMat)
    n = size(S, 1)
    D = Diagonal(S.row_scale)
    A = LinearMap{ComplexF64}(v -> D * _fmm_coupled_mul(S, v), n, n)
    P = LinearMap{ComplexF64}(v -> S.precond \ (D \ v), n, n)
    R = rhs isa AbstractVector ? reshape(rhs, :, 1) : rhs
    X = zeros(ComplexF64, n, size(R, 2))
    for c in axes(R, 2)
        b = D * ComplexF64.(R[:, c])
        if iszero(b)
            push!(S.stats, (iterations=0, solved=true, residual=0.0, seconds=0.0)); continue
        end
        t = @elapsed x, st = gmres(A, b; N=P, ldiv=false, rtol=S.rtol, atol=0.0, memory=min(S.maxiter, 300),
            itmax=S.maxiter, restart=S.maxiter > 300)
        X[:, c] = x
        push!(S.stats, (iterations=st.niter, solved=st.solved, residual=norm(b - A * x) / norm(b), seconds=t))
    end
    return rhs isa AbstractVector ? vec(X) : X
end

end # module
