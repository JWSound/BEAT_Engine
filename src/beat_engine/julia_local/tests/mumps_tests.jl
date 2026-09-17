# MUMPS Schur backend tests. MUMPS_seq_jll ships only with the macOS Metal environment, so run
# this under that project. Everything here runs on the CPU; no Metal device is needed:
#
#     julia --threads=2 --project=src/beat_engine/julia_metal src/beat_engine/julia_local/tests/mumps_tests.jl
using Test, StaticArrays, LinearAlgebra

ENV["BLAB_RUN_COUPLED_REFERENCE"] = "1"
ENV["BLAB_COUPLED_QUADRATURE_ORDER"] = "1"
ENV["BLAB_COUPLED_SINGULAR_ORDER"] = "1"

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))

@testset "MUMPS ships with this environment" begin
    @test !isnothing(Base.locate_package(BeatEngineCoupledCondensed.BeatEngineMumps.MUMPS_SEQ_PKGID))
    @test !isnothing(Base.locate_package(BeatEngineCoupledCondensed.BeatEngineMumps.OPENBLAS32_PKGID))
end
include(joinpath(@__DIR__, "coupled_mumps_tests.jl"))
