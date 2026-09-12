# Load the actual source driver in an isolated namespace so these tests also
# cover its DSP and Neumann data, not just the core operator conventions.
module SourceDriverPhasorTests
using Test, StaticArrays, LinearAlgebra
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))

@testset "source driver DSP and drives conjugate" begin
    T = Float32
    mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
        [(1,3,2),(1,2,4),(1,4,3),(2,3,4)], ones(Int,4))
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(T,3)
    ipp = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1)
    ipq = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0)
    crossover = Dict("type"=>"lowpass", "filter"=>"butterworth", "order"=>2, "frequency_hz"=>1000.)
    channel = Dict("level_db"=>-3., "polarity"=>1, "delay_ms"=>0.3, "lpf"=>crossover)
    radiator = Dict("tag"=>1, "mesh_id"=>1, "channel"=>"main", "velocity_offset_db"=>-2.,
        "level_db"=>-3., "polarity"=>1, "delay_ms"=>0.3, "lpf"=>crossover)
    results = map((NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)) do convention
        with_phasor_convention(convention) do
            frequency, k = T(500), T(0.7)
            omega, rho = T(2pi)*frequency, T(1.21)
            op = assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule;
                skip_singular=false,singular_order=3,backend=:cpu)
            pressure, q = pressure_for_drives(mesh,ones(Int,4),op,ipp,ipq,[radiator],
                ComplexF32[1],rho,omega,k)
            columns = channel_neumann_columns(mesh,ones(Int,4),[radiator],["main"],rho,omega,T)
            @test vec(columns) ≈ q .* T(10)^(-T(2)/T(20))
            @test q ≈ fill(neumann_scale(rho,omega),4)
            (; pressure,q,columns,filter=butterworth_response("lowpass",2,T(1000),frequency),
                channel=channel_drive(channel,frequency),
                radiator=drive_for_radiator(radiator,Dict(),frequency),
                routed=drive_for_radiator(radiator,Dict("main"=>channel),frequency))
        end
    end
    for key in keys(results[1])
        @test getproperty(results[2],key) ≈ conj.(getproperty(results[1],key)) rtol=1f-5 atol=1f-6
    end
    @test phasor_convention() == NEGATIVE_TIME_PHASOR
end
end
