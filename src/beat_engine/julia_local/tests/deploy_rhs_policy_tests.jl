using Test
include(joinpath(@__DIR__, "..", "deploy_rhs_policy.jl"))
@testset "Deploy feedback cache policy" begin
    fast = (build_s=0.6, apply_s=0.006, direct_s=0.5, applications=9)
    @test deploy_use_rhs_operator(fast, 1000, 1024^3)
    @test !deploy_use_rhs_operator(fast, 1024^3, 1024^3)
    @test !deploy_use_rhs_operator(nothing, 1000, 1024^3)
    @test !deploy_use_rhs_operator(merge(fast, (applications=1,)), 1000, 1024^3)
    @test !deploy_use_rhs_operator(merge(fast, (build_s=5.0,)), 1000, 1024^3)
    @test !deploy_use_rhs_operator(merge(fast, (apply_s=0.5,)), 1000, 1024^3)
    @test !deploy_use_rhs_operator(merge(fast, (build_s=NaN,)), 1000, 1024^3)
    @test !deploy_use_rhs_operator(merge(fast, (direct_s=-1.0,)), 1000, 1024^3)
end
