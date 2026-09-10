using Test, StaticArrays, LinearAlgebra, SparseArrays, CUDA
using .BeatEngineCore

@testset "combined A/C and sparse projection against original operators" begin
    T = Float32
    mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
        [(1,3,2),(1,2,4),(1,4,3),(2,3,4)],ones(Int,4))
    p1,dp0 = build_p1_space(mesh),build_dp0_space(mesh)
    for order in (1,2), symmetry in (:off,:x,:xy,:ground)
        rule = triangle_rule(T,order)
        sc = build_singular_correction_cache(mesh,order)
        dc = build_cuda_regular_assembly_cache(mesh,rule)
        dsc = BeatEngineCore.build_cuda_singular_correction_cache(sc,p1,dp0)
        dic = symmetry == :off ? nothing : build_cuda_image_singular_correction_cache(mesh,p1,dp0,order,eachindex(mesh.faces),symmetry)
        ipp = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1;symmetry_mode=symmetry)
        ipq = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0;symmetry_mode=symmetry)
        identity = build_cuda_burton_miller_identity_cache(ipp,ipq,T)
        prepared = (;p1,dp0,rule,symmetry_mode=symmetry,singular_cache=sc,device_cache=dc,
            device_singular_cache=dsc,device_image_singular_cache=dic,device_identity_cache=identity)
        for convention in (NEGATIVE_TIME_PHASOR,POSITIVE_TIME_PHASOR), k in T[.07,7,70]
            with_phasor_convention(convention) do
                operators = assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule;
                    skip_singular=false,singular_order=order,backend=:cuda,device_cache=dc,
                    singular_cache=sc,device_singular_cache=dsc,device_image_singular_cache=dic,
                    symmetry_mode=symmetry,return_device=true,accelerator_quadrature=true)
                alpha = Complex{T}(burton_miller_coupling(k))
                a = T(.5)*ipp-Array(operators.double_layer)+alpha*Array(operators.hypersingular)
                c = Array(operators.single_layer)+alpha*(Array(operators.adjoint_double_layer)+T(.5)*ipq)
                release_operator_storage!(operators)
                for (fused, cap) in ((false,0),(true,0),(true,160))
                    combined = BeatEngineCore.assemble_coupled_burton_miller_cuda(mesh,prepared,k;fused=fused,max_registers=cap)
                    @test isapprox(Array(combined.a),a;rtol=2f-4,atol=2f-5)
                    @test isapprox(Array(combined.c),c;rtol=2f-4,atol=2f-5)
                    q = sparse(ComplexF32[1 0;0 -1;0.3im 0.7;0 0])
                    dq = (colptr=CuArray(Int32.(q.colptr)),rowval=CuArray(Int32.(q.rowval)),nzval=CuArray(q.nzval),ncols=2)
                    motion = ComplexF32[1+2im;0.3-im;-.2+.4im;1;;]
                    prescribed = hcat(motion,conj.(motion))
                    blocks = BeatEngineCore.build_cuda_combined_bem_blocks(combined,dq,motion,prescribed)
                    @test isapprox(Array(blocks.bem_interface_block),c*q;rtol=2f-4,atol=2f-5)
                    @test isapprox(Array(blocks.bem_motion_block),c*motion;rtol=2f-4,atol=2f-5)
                    @test isapprox(Array(blocks.bem_prescribed_rhs),-c*prescribed;rtol=2f-4,atol=2f-5)
                    for array in (blocks.bem_lhs,blocks.bem_interface_block,blocks.bem_motion_block,blocks.bem_prescribed_rhs,dq.colptr,dq.rowval,dq.nzval)
                        CUDA.unsafe_free!(array)
                    end
                end
            end
        end
        release_cuda_burton_miller_identity_cache!(identity)
        dic === nothing || release_cuda_image_singular_correction_cache!(dic)
        GC.gc(); CUDA.reclaim()
    end
end

@testset "coupled CUDA assembly selection, cache reuse and diagnostics" begin
    fixtures=joinpath(@__DIR__, "fixtures")
    fem=load_gmsh41_volume(joinpath(fixtures,"femvolume.msh"),0.001f0)
    bem=load_gmsh22_with_tags(joinpath(fixtures,"exterior_conforming.msh"),0.001f0)
    mapping=build_conforming_interface_map(fem,bem,physical_tag(fem,2,"Interface"),2)
    radiator=physical_tag(fem,2,"Radiator")
    velocity=sparse(findall(==(1),bem.physical_tags),ones(Int,count(==(1),bem.physical_tags)),
        ones(Float32,count(==(1),bem.physical_tags)),length(bem.faces),1)
    excitations=[
        (kind=:normal_velocity,fem_boundary_tags=[radiator],fem_boundary_weights=Float32[1],
         bem_source_index=0,transducer_index=0,amplitude=ComplexF32(.3,.7)),
        (kind=:normal_velocity,fem_boundary_tags=Int[],fem_boundary_weights=Float32[],
         bem_source_index=1,transducer_index=0,amplitude=ComplexF32(-.2,.4)),
    ]
    cache=prepare_coupled_cache(fem,bem,mapping;bem_backend=:cuda,quadrature_order=1,singular_order=1,
        bulk_loss_factor_by_vertex=fill(1f-4,length(fem.vertices)))
    @test cache.device_bem_flux === nothing
    sparse_map=cache.device_bem_flux_sparse
    try
        for convention in (NEGATIVE_TIME_PHASOR,POSITIVE_TIME_PHASOR), frequency in (500f0,1000f0), condensed in (false,true)
            with_phasor_convention(convention) do
                results=map((:operators,:auto)) do mode
                    system=build_coupled_system(fem,bem,mapping,frequency,343f0,1.21f0;
                        cache=cache,bem_backend=:cuda,quadrature_order=1,singular_order=1,
                        validation_diagnostics=false,static_condensation=condensed,
                        coupled_bem_assembly=mode,prescribed_bem_normal_velocity=velocity)
                    try
                        @test system.coupled_bem_assembly == (mode == :auto ? :combined : :operators)
                        @test system.cache.device_bem_flux_sparse === sparse_map
                        solve_coupled_excitations(system,excitations)
                    finally
                        release_coupled_system!(system)
                    end
                end
                for (reference,candidate) in zip(results...)
                    for key in (:fem_pressure,:bem_pressure,:bem_neumann,:interface_flux)
                        @test isapprox(getproperty(candidate,key),getproperty(reference,key);rtol=5f-4,atol=1f-5)
                    end
                end
            end
        end
        system=build_coupled_system(fem,bem,mapping,500f0,343f0,1.21f0;
            cache=cache,bem_backend=:cuda,quadrature_order=1,singular_order=1,
            coupled_bem_assembly=:combined,validation_diagnostics=true,
            prescribed_bem_normal_velocity=velocity)
        try
            @test system.coupled_bem_assembly == :operators
            @test system.bem_rhs_operator !== nothing
            @test system.bem_factorization !== nothing
            @test all(s -> s.all_bem_replay_error < 1f-3,solve_coupled_excitations(system,excitations))
        finally
            release_coupled_system!(system)
        end
        @test_throws ErrorException build_coupled_system(fem,bem,mapping,500f0,343f0,1.21f0;
            cache=cache,bem_backend=:cuda,coupled_bem_assembly=:typo)
        @test_throws ErrorException build_coupled_system(fem,bem,mapping,500f0,343f0,1.21f0;
            cache=cache,bem_backend=:cuda,coupled_bem_max_registers=256)
        @test_throws ErrorException build_coupled_system(fem,bem,mapping,500f0,343f0,1.21f0;
            bem_backend=:cpu,coupled_bem_assembly=:combined)
    finally
        release_coupled_cache!(cache)
    end
end
