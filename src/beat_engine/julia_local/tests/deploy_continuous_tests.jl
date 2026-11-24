# Included inside the standalone Deploy ROM algebra test module.
@testset "continuous ROM block factors with independent loads" begin
    model = (gram=[0.5 1.0; 1.0 3.0], gram_factors=Dict{DataType,LU}())
    for T in (ComplexF32, ComplexF64)
        rhs = T[1+im 2; 3 0; -2im 1; 4 -im]
        for z in (1+im, 3-2im)
            system = T[0.5 1 z 4; 1 3 2 im; 0 0 z 1; 0 0 -im 5]
            block = factor_deploy_continuous(model, system)
            @test block.gram === model.gram_factors[T]
            @test block \ rhs ≈ system \ rhs
            @test block \ rhs[:,1] ≈ system \ rhs[:,1]
        end
    end
    @test length(model.gram_factors) == 2
    empty_interface = (gram=zeros(0,0), gram_factors=Dict{DataType,LU}())
    @test factor_deploy_continuous(empty_interface, ComplexF32[1 2; 2 1]) isa LU
end

@testset "native continuous speaker archive, pressure feedback and host cache" begin
    with_phasor_convention(POSITIVE_TIME_PHASOR) do
        mktempdir() do directory
            coefficient(x) = Dict("storage"=>"f64le-column-major", "shape"=>[1,1],
                "base64"=>base64encode(reinterpret(UInt8, htol.([reinterpret(UInt64, Float64(x))]))))
            matrices = Dict(name=>coefficient(x) for (name,x) in pairs((
                K=4, M=0, loss=0, gram=2, trace=1, flux=1, bem_force=1, bem_velocity=1, force=0, surface=0, drive=1)))
            driver = Dict("bl_n_per_a"=>3, "rms_n_s_per_m"=>1, "mmd_kg"=>0.1, "cms_m_per_n"=>0.01,
                "re_ohm"=>4, "le_h"=>0.001, "semi_inductance"=>nothing, "lumped_sealed_rear_chamber"=>nothing)
            bytes = JSON.json(Dict("metadata"=>Dict("format"=>"continuous-speaker-v1","phasor"=>POSITIVE_TIME_PHASOR),
                "band"=>[20,500], "c"=>343, "rho"=>1.21, "matrices"=>matrices, "walls"=>[], "drivers"=>[driver]))
            member = "model.json"
            package = joinpath(directory,"speaker.blabsp")
            write(joinpath(directory,member), bytes)
            run(pipeline(Cmd(`$(p7zip_jll.p7zip()) a -tzip $package $member`; dir=directory); stdout=devnull))
            id = bytes2hex(sha256(bytes))
            ref = Dict("package"=>package,"member"=>member,"sha256"=>id)
            @test_throws ErrorException read_deploy_continuous_model(merge(ref,Dict("sha256"=>repeat("0",64))))
            prepared = deploy_continuous_frequency(ref, 160, Float64)
            @test deploy_continuous_frequency(ref, 160.0, Float64) === prepared
            a = prepared.arrays
            p, u = 2+im, 0
            state = only(prepared.factors) \ (vec(a["b"])*u - vec(a["c"])*p)
            s = 2im*pi*160
            ze = 4 + s*0.001
            v = -p/(1+s*0.1+100/s+9/ze)
            @test isapprox(state, [2p,v,-3v/ze])
            @test isapprox(only(a["d"][1,:,:]*state), 2p-1.21*s*v)
            @test_throws ErrorException deploy_continuous_frequency(ref, 501, Float64)
            size_limit = Base.summarysize(prepared)
            next_frequency = deploy_continuous_frequency(ref, 161, Float64; budget=size_limit)
            @test only(next_frequency.factors).gram === only(prepared.factors).gram
            @test length(DEPLOY_CONTINUOUS_RECENT) == 1
            @test !haskey(DEPLOY_CONTINUOUS_RECENT,(id,160.0,Float64))
            deploy_continuous_frequency(ref, 162, Float64; budget=1)
            @test !haskey(DEPLOY_CONTINUOUS_RECENT,(id,162.0,Float64))
            @test length(DEPLOY_CONTINUOUS_MODELS) == 1
            prune_deploy_continuous_models!(nothing)
            @test isempty(DEPLOY_CONTINUOUS_MODELS) && isempty(DEPLOY_CONTINUOUS_RECENT) && isempty(DEPLOY_CONTINUOUS_LRU)
        end
    end
end
