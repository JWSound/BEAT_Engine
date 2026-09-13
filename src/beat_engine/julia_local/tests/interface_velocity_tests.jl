using Test, StaticArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineInterfaceVelocity.jl"))
using .BeatEngineInterfaceVelocity

@testset "area averaged interface normal velocity" begin
    mesh = (vertices=[SVector(0.,0.,0.), SVector(2.,0.,0.), SVector(0.,1.,0.),
                      SVector(0.,0.,1.), SVector(4.,0.,1.), SVector(0.,1.,1.)],
            boundary_faces=[(1,2,3), (4,5,6)])
    maps = [(fem_vertex_indices=collect(1:6), fem_face_indices=[1,2])]
    ranges = [1:6]
    for T in (Float32, Float64)
        averaging = interface_average_weights(mesh, maps, ranges, T)
        @test isapprox(averaging.areas, [3])
        scale = -2im
        # Unequal areas 1 and 2; opposite complex velocities integrate to zero.
        flux = scale .* Complex{T}.([2im,2im,2im,-1im,-1im,-1im])
        @test isapprox(only(interface_average_normal_velocity(flux, ranges, averaging.weights, scale)), 0; atol=1e-6)
        uniform = fill(Complex{T}(3+4im)*scale, 6)
        @test isapprox(only(interface_average_normal_velocity(uniform, ranges, averaging.weights, scale)), 3+4im)
        # Negative-time convention conjugates both flux and conversion scale.
        @test isapprox(only(interface_average_normal_velocity(conj.(uniform), ranges, averaging.weights, conj(scale))), 3-4im)
    end
end
