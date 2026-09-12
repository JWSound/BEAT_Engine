using Test, StaticArrays, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed
const COUPLED_FIXTURE_ROOT = joinpath(@__DIR__, "fixtures")
const CUDA_MODULE = BeatEngineCore.CUDA_MODULE
cuda_available() = CUDA_MODULE !== nothing && CUDA_MODULE.functional()
const METAL_MODULE = BeatEngineCore.METAL_MODULE
metal_available() = METAL_MODULE !== nothing && METAL_MODULE.functional()
include(joinpath(@__DIR__, "phasor_tests.jl"))
