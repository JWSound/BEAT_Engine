using LinearAlgebra, StaticArrays

function _fmm_icosphere(radius, subdivisions; centre=SVector(0.0, 0.0, 0.0))
    phi = (1 + sqrt(5.0)) / 2
    v = SVector{3,Float64}[(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi),
        (0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    f = NTuple{3,Int}[(1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12), (2, 6, 10), (6, 12, 5), (12, 11, 3),
        (11, 8, 7), (8, 2, 9), (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10), (5, 10, 6), (3, 5, 12),
        (7, 3, 11), (9, 7, 8), (10, 9, 2)]
    for _ in 1:subdivisions
        mid = Dict{Tuple{Int,Int},Int}()
        m(a, b) = get!(mid, minmax(a, b)) do
            push!(v, (v[a] + v[b]) / 2); length(v)
        end
        f = [g for (a, b, c) in f for g in ((a, m(a, b), m(c, a)), (b, m(b, c), m(a, b)), (c, m(c, a), m(b, c)), (m(a, b), m(b, c), m(c, a)))]
    end
    v = [centre + radius * x / norm(x) for x in v]
    f = [dot(cross(v[b] - v[a], v[c] - v[a]), (v[a] + v[b] + v[c]) / 3 - centre) >= 0 ? (a, b, c) : (a, c, b) for (a, b, c) in f]
    return BoundaryMesh(v, f, fill(1, length(f)))
end

