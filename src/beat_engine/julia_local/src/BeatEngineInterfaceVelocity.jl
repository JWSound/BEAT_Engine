module BeatEngineInterfaceVelocity

using LinearAlgebra
export interface_average_weights, interface_average_normal_velocity

"""Area-normalized P1 nodal weights on each interface, along FEM outward normals.

Areas refer to the represented mesh patch (before symmetry replication).
"""
function interface_average_weights(mesh, maps, ranges, ::Type{T}) where {T<:AbstractFloat}
    weights = Vector{Vector{T}}()
    areas = T[]
    for (map, range) in zip(maps, ranges)
        local_index = Dict(vertex => index for (index, vertex) in enumerate(map.fem_vertex_indices))
        w = zeros(T, length(range))
        for face_index in map.fem_face_indices
            face = mesh.boundary_faces[face_index]
            a, b, c = (mesh.vertices[vertex] for vertex in face)
            area = T(norm(cross(b - a, c - a)) / 2)
            for vertex in face
                w[local_index[vertex]] += area / T(3)
            end
        end
        area = sum(w)
        isfinite(area) && area > zero(T) || error("Interface area must be finite and positive.")
        push!(areas, area)
        push!(weights, w ./ area)
    end
    return (weights=weights, areas=areas)
end

"""Convert integrated pressure normal derivative to complex normal velocity.

The caller supplies the solver's phasor-aware Neumann scale (Pa/m per m/s).
"""
function interface_average_normal_velocity(flux, ranges, weights, neumann_scale)
    return [sum(w .* view(flux, range)) / neumann_scale for (range, w) in zip(ranges, weights)]
end

end
