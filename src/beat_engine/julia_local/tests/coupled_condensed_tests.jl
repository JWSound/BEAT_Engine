isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupled
using .BeatEngineCoupledCondensed
using LinearAlgebra, Random, SparseArrays, StaticArrays

const CONDENSED_FIXTURE_ROOT = joinpath(@__DIR__, "fixtures")
const CONDENSED_QUADRATURE_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
const CONDENSED_SINGULAR_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))

function condensed_synthetic_case(
    ::Type{T};
    vertex_count::Int=60,
    retained_count::Int=10,
    density::Float64=0.08,
    interior_load::Bool=false,
) where {T<:AbstractFloat}
    Random.seed!(20260814)
    retained = sort(randperm(vertex_count)[1:retained_count])
    # Diagonally dominant so the interior block is safely invertible and the manufactured
    # solution isolates the condensation algebra rather than conditioning.
    system = SparseMatrixCSC{Complex{T},Int}(
        sprand(Complex{T}, vertex_count, vertex_count, density) +
        Complex{T}(vertex_count / 4) * I,
    )
    load_rows = interior_load ? vcat(retained[2:end], first(setdiff(1:vertex_count, retained))) : retained
    fem_load = sparse(load_rows, 1:retained_count, ones(T, retained_count), vertex_count, retained_count)
    operators = InterfaceOperators(
        fem_load,
        spzeros(T, 4, retained_count),
        spzeros(T, retained_count, vertex_count),
        spzeros(T, retained_count, 4),
    )
    return system, operators, retained
end

@testset "Schur condensation algebra" begin
    system, operators, retained = condensed_synthetic_case(Float64)
    vertex_count = size(system, 1)
    interior = setdiff(1:vertex_count, retained)

    condensation = BeatEngineCoupledCondensed._build_condensation(system, operators, retained)

    @test condensation.interior_count == length(interior)
    @test condensation.retained_count == length(retained)
    @test condensation.interior_vertices == interior
    @test condensation.retained_vertices == retained
    @test eltype(condensation.schur) == ComplexF64
    @test size(condensation.schur) == (length(retained), length(retained))

    # The Schur complement is unnegated, in the same sign convention as `system`. Computed here
    # independently and densely.
    expected_schur = (
        Matrix(system[retained, retained]) -
        Matrix(system[retained, interior]) *
        (Matrix(system[interior, interior]) \ Matrix(system[interior, retained]))
    )
    @test condensation.schur ≈ expected_schur rtol = 1e-10

    # Manufactured solution: plant u = 1, form f = A u, and require the full round trip
    # (forward -> reduced dense solve -> backward) to recover it.
    planted = ones(ComplexF64, vertex_count, 1)
    forcing = Matrix(system * planted)
    reduced_rhs, interior_rhs = BeatEngineCoupledCondensed._forward_schur(condensation, forcing)
    @test size(reduced_rhs) == (length(retained), 1)
    recovered, interior_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        interior_rhs,
        condensation.schur \ reduced_rhs,
    )
    @test size(recovered) == (vertex_count, 1)
    @test norm(recovered - planted) / norm(planted) < 1e-10
    @test length(interior_residual) == 1
    @test maximum(interior_residual) < 1e-10

    # Two excitations at once, the second scaled, must scale linearly.
    multi_forcing = hcat(forcing, 0.5 .* forcing)
    multi_reduced, multi_interior = BeatEngineCoupledCondensed._forward_schur(condensation, multi_forcing)
    multi_recovered, multi_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        multi_interior,
        condensation.schur \ multi_reduced,
    )
    @test norm(multi_recovered[:, 1] - planted[:, 1]) / norm(planted) < 1e-10
    @test norm(multi_recovered[:, 2] - 0.5 .* planted[:, 1]) / norm(planted) < 1e-10
    @test length(multi_residual) == 2
    @test maximum(multi_residual) < 1e-10

    # A voltage-driven transducer forces the coupled system entirely through the electrical
    # rows, so the FEM interior sees f_I = 0. The residual must stay at round-off there rather
    # than normalizing against nothing.
    gamma_only_forcing = zeros(ComplexF64, vertex_count, 1)
    gamma_only_forcing[retained, 1] .= 1 .+ 0.5im
    @test iszero(gamma_only_forcing[interior, 1])
    gamma_reduced, gamma_interior = BeatEngineCoupledCondensed._forward_schur(
        condensation,
        gamma_only_forcing,
    )
    gamma_recovered, gamma_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        gamma_interior,
        condensation.schur \ gamma_reduced,
    )
    @test maximum(gamma_residual) < 1e-10
    # With f_I = 0 the interior rows must annihilate the recovered solution, so measure that
    # absolutely against the solution magnitude rather than against the zero forcing.
    @test norm(system[interior, :] * gamma_recovered[:, 1]) / norm(gamma_recovered[:, 1]) < 1e-10
    @test norm(gamma_recovered[:, 1]) > 0

    # The chunked column sweep runs one task per block against its own copy of the UMFPACK
    # factorization, so the partition must not move the answer. Every column is solved and
    # accumulated independently of the block it lands in, which makes the agreement exact and not
    # merely close -- two tasks sharing a solve workspace could not survive that. Run this file
    # under `julia -t <n>` for the comparison to actually cover concurrent tasks; at the default
    # single thread it still checks that the block partition itself is neutral.
    single_shot = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=length(retained),
    )
    narrow = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=3,
    )
    # One block per column: more blocks than threads, so the tasks stride over several each.
    per_column = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=1,
    )
    @test single_shot.schur ≈ condensation.schur rtol = 1e-12
    @test narrow.schur ≈ condensation.schur rtol = 1e-12
    @test single_shot.schur == condensation.schur
    @test narrow.schur == condensation.schur
    @test per_column.schur == condensation.schur
    # A block wider than Γ is clamped to it rather than overrunning the accumulator.
    wide = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=4 * length(retained),
    )
    @test wide.schur == condensation.schur
    # The requested width is an upper bound. It is narrowed until Γ splits into at least one
    # block per thread, so no thread sits idle through the sweep whatever the interface size.
    # Stated as a property because the answer depends on the thread count this file is run under.
    for requested in (1, 3, length(retained), 4 * length(retained))
        built = BeatEngineCoupledCondensed._build_condensation(
            system,
            operators,
            retained;
            schur_block_columns=requested,
        )
        @test built.schur == condensation.schur
        @test 1 <= built.schur_block_columns <= requested
        @test cld(length(retained), built.schur_block_columns) >=
              min(Threads.nthreads(), length(retained))
    end
    @test_throws ErrorException BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=0,
    )

    # Mixed precision: the interior is factored in ComplexF64 whatever T is, and only the
    # assembled Schur complement is demoted.
    system32, operators32, retained32 = condensed_synthetic_case(Float32)
    condensation32 = BeatEngineCoupledCondensed._build_condensation(system32, operators32, retained32)
    @test eltype(condensation32.schur) == ComplexF32
    @test eltype(condensation32.interior_system) == ComplexF64
    @test eltype(condensation32.factorization) == ComplexF64
    planted32 = ones(ComplexF32, size(system32, 1), 1)
    forcing32 = Matrix(system32 * planted32)
    reduced32, interior32 = BeatEngineCoupledCondensed._forward_schur(condensation32, forcing32)
    @test eltype(reduced32) == ComplexF32
    # Under double dense assembly the reduced right-hand side is returned before demotion.
    reduced32_double, _ = BeatEngineCoupledCondensed._forward_schur(condensation32, forcing32; result_type=Float64)
    @test eltype(reduced32_double) == ComplexF64
    @test ComplexF32.(reduced32_double) == reduced32
    @test any(value -> ComplexF64(ComplexF32(value)) != value, reduced32_double)
    recovered32, residual32 = BeatEngineCoupledCondensed._backward_schur(
        condensation32,
        interior32,
        condensation32.schur \ reduced32,
    )
    @test eltype(recovered32) == ComplexF32
    @test norm(recovered32 - planted32) / norm(planted32) < 1e-4
    @test eltype(residual32) == Float32
    @test maximum(residual32) < 1e-5

    # A mesh whose every FEM vertex lies on a retained interface or moving surface has no
    # interior block. Condensation is then an exact no-op rather than an invalid 0x0 UMFPACK
    # factorization.
    all_retained_system, all_retained_operators, _ = condensed_synthetic_case(
        Float64;
        vertex_count=10,
        retained_count=10,
    )
    all_retained = collect(1:size(all_retained_system, 1))
    all_retained_condensation = BeatEngineCoupledCondensed._build_condensation(
        all_retained_system,
        all_retained_operators,
        all_retained,
    )
    @test all_retained_condensation.backend == :cpu_noop
    @test all_retained_condensation.interior_count == 0
    @test all_retained_condensation.schur == Matrix(all_retained_system)
    all_retained_planted = ones(ComplexF64, size(all_retained_system, 1), 1)
    all_retained_forcing = Matrix(all_retained_system * all_retained_planted)
    all_retained_rhs, all_retained_interior = BeatEngineCoupledCondensed._forward_schur(
        all_retained_condensation,
        all_retained_forcing,
    )
    all_retained_recovered, all_retained_residual = BeatEngineCoupledCondensed._backward_schur(
        all_retained_condensation,
        all_retained_interior,
        all_retained_condensation.schur \ all_retained_rhs,
    )
    @test all_retained_recovered ≈ all_retained_planted rtol = 1e-10
    @test maximum(all_retained_residual) == 0
    BeatEngineCoupledCondensed._release_condensation!(all_retained_condensation)

    # The structural guard that makes the condensation valid: interface loads must not touch
    # eliminated interior vertices.
    violating_system, violating_operators, violating_retained =
        condensed_synthetic_case(Float64; interior_load=true)
    @test_throws ErrorException BeatEngineCoupledCondensed._build_condensation(
        violating_system,
        violating_operators,
        violating_retained,
    )
end

if get(ENV, "BLAB_RUN_COUPLED_REFERENCE", "0") == "1"
    @testset "Condensed coupled solver matches monolithic" begin
        fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
        bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
        interface_map = build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            physical_tag(fem_mesh, 2, "Interface"),
            2,
        )
        radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
        fem_mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
        bem_mesh32 = load_gmsh22_with_tags(
            joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"),
            Float32(0.001),
        )
        interface_map32 = build_conforming_interface_map(
            fem_mesh32,
            bem_mesh32,
            physical_tag(fem_mesh32, 2, "Interface"),
            2,
        )

        function monolithic_system(::Type{T}, frequency; transducers=ElectrodynamicTransducer{T}[]) where {T}
            mesh, boundary, mapping = T === Float32 ?
                                      (fem_mesh32, bem_mesh32, interface_map32) :
                                      (fem_mesh, bem_mesh, interface_map)
            return build_coupled_system(
                mesh,
                boundary,
                mapping,
                T(frequency),
                T(343.0),
                T(1.21);
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                validation_diagnostics=false,
                bem_backend=:cpu,
                transducers=transducers,
            )
        end

        function condensed_system_at(::Type{T}, frequency; transducers=ElectrodynamicTransducer{T}[]) where {T}
            mesh, boundary, mapping = T === Float32 ?
                                      (fem_mesh32, bem_mesh32, interface_map32) :
                                      (fem_mesh, bem_mesh, interface_map)
            return build_condensed_coupled_system(
                mesh,
                boundary,
                mapping,
                T(frequency),
                T(343.0),
                T(1.21);
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                transducers=transducers,
            )
        end

        relative_error(reference, candidate) = norm(
            ComplexF64.(candidate) .- ComplexF64.(reference),
        ) / norm(ComplexF64.(reference))

        monolithic = monolithic_system(Float64, 500.0)
        condensed = condensed_system_at(Float64, 500.0)
        try
            @test monolithic.formulation == :monolithic
            @test condensed.formulation == :fem_interface_condensed
            @test condensed.solved_system_order < condensed.full_system_order
            @test condensed.full_system_order == monolithic.full_system_order
            @test condensed.linear_backend == :cpu
            @test eltype(condensed.condensation.schur) == ComplexF64
            @test condensed.condensation.retained_count == length(interface_map.fem_vertex_indices)

            reference = only(solve_coupled_systems(monolithic, [radiator_tag]))
            candidates = solve_condensed_coupled_systems(
                condensed,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF64[1, 0.5],
            )
            candidate = candidates[1]

            # Both sides are CPU double and differ only in elimination order.
            @test relative_error(reference.fem_pressure, candidate.fem_pressure) < 1e-9
            @test relative_error(reference.bem_pressure, candidate.bem_pressure) < 1e-9
            @test relative_error(reference.interface_flux, candidate.interface_flux) < 1e-9
            @test relative_error(reference.bem_neumann, candidate.bem_neumann) < 1e-9
            @test candidate.fem_interior_residual < 1e-10
            @test isnothing(candidate.relative_residual)
            @test candidate.pressure_continuity_error < 1e-8
            @test candidate.flux_conservation_error < 1e-10
            @test relative_error(0.5 .* candidate.bem_pressure, candidates[2].bem_pressure) < 1e-9
            @test candidates[2].fem_interior_residual < 1e-10

            condensed32 = condensed_system_at(Float32, 500.0)
            try
                @test eltype(condensed32.condensation.schur) == ComplexF32
                # The interior factorization stays double whatever T is.
                @test eltype(condensed32.condensation.interior_system) == ComplexF64
                solution32 = solve_condensed_coupled_system(
                    condensed32,
                    physical_tag(fem_mesh32, 2, "Radiator"),
                )
                @test relative_error(candidate.fem_pressure, solution32.fem_pressure) < 1e-4
                @test relative_error(candidate.bem_pressure, solution32.bem_pressure) < 1e-4
                @test relative_error(candidate.interface_flux, solution32.interface_flux) < 1e-4
                @test solution32.fem_interior_residual < 1e-5
            finally
                release_condensed_coupled_system!(condensed32)
            end
        finally
            release_coupled_system!(monolithic)
            release_condensed_coupled_system!(condensed)
        end

        # Γ is `interface ∪ transducer_fem_vertices`, so a transducer widens the retained set
        # and exercises the retained-vertex slicing of the transducer blocks.
        transducer = ElectrodynamicTransducer{Float64}(
            "component:test",
            [radiator_tag],
            [1.0],
            [1],
            [-1.0],
            SVector(0.0, 0.0, 1.0),
            2.0,
            1,
            6.0,
            0.0005,
            7.0,
            0.015,
            0.0005,
            1.0,
        )
        transducer_monolithic = monolithic_system(Float64, 500.0; transducers=[transducer])
        transducer_condensed = condensed_system_at(Float64, 500.0; transducers=[transducer])
        try
            @test transducer_condensed.condensation.retained_count >
                  length(interface_map.fem_vertex_indices)
            @test transducer_condensed.solved_system_order < transducer_condensed.full_system_order
            voltage = (
                kind=:voltage,
                radiator_tag=0,
                transducer_index=1,
                amplitude=ComplexF64(1, 0),
            )
            transducer_reference = only(solve_coupled_excitations(transducer_monolithic, [voltage]))
            transducer_candidate = only(
                solve_condensed_coupled_excitations(transducer_condensed, [voltage]),
            )
            @test relative_error(
                transducer_reference.fem_pressure,
                transducer_candidate.fem_pressure,
            ) < 1e-9
            @test relative_error(
                transducer_reference.bem_pressure,
                transducer_candidate.bem_pressure,
            ) < 1e-9
            @test relative_error(
                transducer_reference.diaphragm_velocity,
                transducer_candidate.diaphragm_velocity,
            ) < 1e-9
            @test relative_error(
                transducer_reference.voice_coil_current,
                transducer_candidate.voice_coil_current,
            ) < 1e-9
            @test transducer_candidate.fem_interior_residual < 1e-10

            # Γ is built from the interface map and transducer surfaces only, neither of which
            # depends on symmetry — symmetry enters only the BEM operators — so the retained set,
            # and with it the condensation, is invariant under :x and :xy.
            transducer_fem_vertices = unique(
                findnz(
                    assemble_transducer_operators(fem_mesh, bem_mesh, [transducer]).fem_surface,
                )[1],
            )
            @test transducer_condensed.retained_fem_vertices ==
                  sort(unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)))
            @test transducer_condensed.retained_fem_vertices ==
                  transducer_monolithic.retained_fem_vertices
        finally
            release_coupled_system!(transducer_monolithic)
            release_condensed_coupled_system!(transducer_condensed)
        end

        # Precision switches on the fixture with a transducer. Unset on the CPU backend nothing
        # changes; `auto` (what an unset environment selects on Metal) gives a Float64 FEM system,
        # a double-precision dense system and a refined Float32 LU, agreeing with the plain
        # Float64 LU of the same system.
        transducer32 = ElectrodynamicTransducer{Float32}(
            "component:test", [physical_tag(fem_mesh32, 2, "Radiator")], Float32[1], [1], Float32[-1],
            SVector(0f0, 0f0, 1f0), 2f0, 1, 6f0, 0.0005f0, 7f0, 0.015f0, 0.0005f0, 1f0,
        )
        precision_names = ("BLAB_COUPLED_DENSE_FLOAT64", "BLAB_COUPLED_DENSE_REFINEMENT", "BLAB_COUPLED_FEM_FLOAT64")
        voltage32 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))]
        default32 = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                            (name => nothing for name in precision_names)...)
        refined32 = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                            "BLAB_COUPLED_DENSE_REFINEMENT" => "auto", "BLAB_COUPLED_FEM_FLOAT64" => "auto")
        plain64lu = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                            "BLAB_COUPLED_DENSE_FLOAT64" => "1", "BLAB_COUPLED_FEM_FLOAT64" => "1")
        try
            @test default32.factorization isa LinearAlgebra.LU{ComplexF32}
            @test default32.dense_scalar_type == Float32 && default32.fem_scalar_type == Float32
            @test refined32.factorization isa BeatEngineCoupledCondensed.RefinedDenseLU
            @test refined32.dense_scalar_type == Float64 && refined32.fem_scalar_type == Float64
            # The Schur block is formed in Float64 and must not be demoted on the way to the dense matrix.
            @test eltype(refined32.condensation.schur) == ComplexF64
            @test any(value -> ComplexF64(ComplexF32(value)) != value,
                      refined32.factorization.matrix[refined32.gamma_range, refined32.gamma_range])
            @test plain64lu.factorization isa LinearAlgebra.LU{ComplexF64}
            @test BeatEngineCoupledCondensed.dense_solver_diagnostics(refined32)["dense_solver"] == "lu_float32_refined"
            refined_solution = only(solve_condensed_coupled_excitations(refined32, voltage32))
            plain_solution = only(solve_condensed_coupled_excitations(plain64lu, voltage32))
            @test eltype(refined_solution.bem_pressure) == ComplexF32
            for field in (:fem_pressure, :bem_pressure, :interface_flux, :diaphragm_velocity, :voice_coil_current)
                @test relative_error(getproperty(plain_solution, field), getproperty(refined_solution, field)) < 1e-6
            end
        finally
            foreach(release_condensed_coupled_system!, (default32, refined32, plain64lu))
        end
        # Under precision=float64 the switches change nothing.
        switched64 = withenv(() -> condensed_system_at(Float64, 700.0; transducers=[transducer]),
                             "BLAB_COUPLED_DENSE_REFINEMENT" => "1", "BLAB_COUPLED_FEM_FLOAT64" => "1")
        unswitched64 = withenv(() -> condensed_system_at(Float64, 700.0; transducers=[transducer]),
                               (name => nothing for name in precision_names)...)
        try
            voltage64 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))]
            @test switched64.factorization isa LinearAlgebra.LU{ComplexF64}
            @test only(solve_condensed_coupled_excitations(switched64, voltage64)).bem_pressure ==
                  only(solve_condensed_coupled_excitations(unswitched64, voltage64)).bem_pressure
        finally
            release_condensed_coupled_system!(switched64)
            release_condensed_coupled_system!(unswitched64)
        end
    end

    @testset "Condensed coupled solver interior resonance" begin
        sound_speed = 343.0
        fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
        bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
        interface_map = build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            physical_tag(fem_mesh, 2, "Interface"),
            2,
        )
        radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
        fem_mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
        bem_mesh32 = load_gmsh22_with_tags(
            joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"),
            Float32(0.001),
        )
        interface_map32 = build_conforming_interface_map(
            fem_mesh32,
            bem_mesh32,
            physical_tag(fem_mesh32, 2, "Interface"),
            2,
        )

        # The Schur complement has poles at the eigenvalues of A_II. Restricting the FEM operator
        # to interior rows *and* columns imposes u_Γ = 0, so these are the cavity modes with a
        # pressure-release interface — a different, lower set than the rigid-wall modes
        # `sealed_cavity_modes` reports.
        retained = sort(unique(interface_map.fem_vertex_indices))
        interior = setdiff(1:length(fem_mesh.vertices), retained)
        stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
        interior_eigenvalues = eigen(
            Symmetric(Matrix(stiffness[interior, interior])),
            Symmetric(Matrix(mass[interior, interior])),
        ).values
        scale = maximum(abs, interior_eigenvalues)
        positive = filter(value -> value > 1e-8 * scale, sort(real.(interior_eigenvalues)))
        pole_hz = sound_speed * sqrt(first(positive)) / (2pi)
        @test pole_hz > 0
        @test pole_hz < first(sealed_cavity_modes(fem_mesh, sound_speed; count=1))

        function resonance_solution(::Type{T}, frequency, bulk_loss) where {T}
            mesh, boundary, mapping = T === Float32 ?
                                      (fem_mesh32, bem_mesh32, interface_map32) :
                                      (fem_mesh, bem_mesh, interface_map)
            system = build_condensed_coupled_system(
                mesh,
                boundary,
                mapping,
                T(frequency),
                T(sound_speed),
                T(1.21);
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                bulk_loss_factor=T(bulk_loss),
            )
            try
                return solve_condensed_coupled_system(
                    system,
                    T === Float32 ? physical_tag(fem_mesh32, 2, "Radiator") : radiator_tag,
                )
            finally
                release_condensed_coupled_system!(system)
            end
        end

        function monolithic_solution(frequency, bulk_loss)
            system = build_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                frequency,
                sound_speed,
                1.21;
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
                validation_diagnostics=false,
                bem_backend=:cpu,
                bulk_loss_factor=bulk_loss,
            )
            try
                return solve_coupled_system(system, radiator_tag)
            finally
                release_coupled_system!(system)
            end
        end

        relative_error(reference, candidate) = norm(
            ComplexF64.(candidate) .- ComplexF64.(reference),
        ) / norm(ComplexF64.(reference))

        for bulk_loss in (0.0, 0.02)
            observed = NamedTuple[]
            for offset in (-0.005, 0.0, 0.005)
                frequency = pole_hz * (1 + offset)
                reference = monolithic_solution(frequency, bulk_loss)
                candidate = resonance_solution(Float64, frequency, bulk_loss)
                candidate32 = resonance_solution(Float32, frequency, bulk_loss)
                error64 = relative_error(reference.fem_pressure, candidate.fem_pressure)
                error32 = relative_error(reference.fem_pressure, candidate32.fem_pressure)
                push!(
                    observed,
                    (
                        offset=offset,
                        frequency_hz=frequency,
                        error64=error64,
                        error32=error32,
                        residual64=candidate.fem_interior_residual,
                        residual32=candidate32.fem_interior_residual,
                    ),
                )

                # The interior residual only validates the back-substitution, so it stays small
                # even where the Schur complement is badly conditioned. It is not a resonance
                # detector — hence the separate accuracy assertions.
                @test candidate.fem_interior_residual < 1e-10
                @test candidate32.fem_interior_residual < 1e-4
                @test all(isfinite, real.(candidate.fem_pressure))
                @test all(isfinite, real.(candidate32.fem_pressure))

                if offset == 0.0 && bulk_loss == 0.0
                    # Straddling the pole with no loss is where condensation is weakest and the
                    # tight tolerance genuinely does not hold. Assert only that it degrades
                    # gracefully rather than diverging; the measured value is reported below as
                    # the documented limit rather than pinned here.
                    @test error64 < 1e-1
                    @test error32 < 1e-1
                else
                    @test error64 < 1e-9
                    @test error32 < 1e-3
                end
            end
            @info "Schur condensation interior-resonance sweep" bulk_loss pole_hz observed
        end
    end
else
    @info "Set BLAB_RUN_COUPLED_REFERENCE=1 to run the condensed coupled solver validation."
end

@testset "Condensed regular assembly matches the shared CPU assembly" begin
    # The condensed solver runs its own fork of the CPU regular assembly so it can be optimised
    # without touching the path every other backend shares. This is the contract that makes that
    # safe: bitwise-identical operators, across every symmetry mode, cached and uncached. Any
    # optimisation added to the fork has to keep this passing.
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    # Each symmetry mode needs a mesh that actually lies in its fundamental domain: the half mesh
    # straddles y<0 so :xy rejects it. :xy matters most here -- it is four image sweeps, which is
    # where a fused assembly has the most to gain and the most to get wrong.
    for (symmetry, mesh_name) in ((:off, "sample.msh"), (:x, "sample_half.msh"), (:xy, "sample_quarter.msh"))
        mesh = load_gmsh22_with_tags(
            joinpath(@__DIR__, "..", "test_meshes", mesh_name), Float32(0.001),
        )
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        element_indices = 1:min(24, length(mesh.faces))
        singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
        symmetry == :off || validate_symmetry_fundamental_domain!(mesh, symmetry)
        shared = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, element_indices=element_indices,
            backend=:cpu, singular_cache=singular_cache, symmetry_mode=symmetry,
        )
        forked = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, element_indices=element_indices,
            singular_cache=singular_cache, symmetry_mode=symmetry,
        )
        # With no images the fused sweep reduces to the base sweep, so it must be bitwise
        # identical. With images it sums each pair's contributions before scattering instead of
        # depositing them across four separate passes, which reorders the floating-point
        # summation -- equal to round-off, not bit for bit. Both halves are asserted so a
        # regression cannot hide behind the looser bound.
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            if symmetry == :off
                @test getproperty(forked, op) == getproperty(shared, op)
            else
                @test getproperty(forked, op) ≈ getproperty(shared, op) rtol = 1.0f-5
            end
        end
        for count in (:regular_pairs, :singular_pairs, :skipped_pairs, :image_singular_pairs)
            @test getproperty(forked, count) == getproperty(shared, count)
        end

        # Again through a prebuilt cache, which is how the solver actually calls it.
        cache = build_beat_cpu_assembly_cache(
            mesh, p1, dp0, rule;
            singular_order=2, element_indices=element_indices, symmetry_mode=symmetry,
        )
        shared_cached = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, backend=:cpu,
            singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=symmetry,
        )
        forked_cached = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2,
            singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=symmetry,
        )
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            if symmetry == :off
                @test getproperty(forked_cached, op) == getproperty(shared_cached, op)
            else
                @test getproperty(forked_cached, op) ≈ getproperty(shared_cached, op) rtol = 1.0f-5
            end
        end
        # The cached and uncached fused paths must agree with each other exactly: same order, same
        # data, only the provenance of the reflected element sets differs.
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            @test getproperty(forked_cached, op) == getproperty(forked, op)
        end
    end

    # The fork keeps the strict stale-cache guard: a value-equal but distinct rule is rejected.
    guard_mesh = load_gmsh22_with_tags(
        joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001),
    )
    guard_p1 = build_p1_space(guard_mesh)
    guard_dp0 = build_dp0_space(guard_mesh)
    guard_indices = 1:min(24, length(guard_mesh.faces))
    guard_cache = build_beat_cpu_assembly_cache(
        guard_mesh, guard_p1, guard_dp0, rule;
        singular_order=2, element_indices=guard_indices, symmetry_mode=:off,
    )
    @test_throws "cache quadrature rule does not match" BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
        guard_mesh, guard_p1, guard_dp0, k, triangle_rule(Float32, 2);
        skip_singular=false, singular_order=2,
        singular_cache=build_singular_correction_cache(guard_mesh, 2, guard_indices),
        cpu_cache=guard_cache, symmetry_mode=:off,
    )
end

@testset "Wavelength quadrature selection" begin
    # h = sqrt(area) = 0.01 m exactly, so kh = 2*pi*f/c * 0.01 and a frequency can be placed
    # either side of a cutoff by hand.
    areas = fill(1.0e-4, 16)
    c = 343.0
    kh(f) = 2pi * f / c * 0.01
    low, high = 2000.0, 20000.0
    @test kh(low) < 2.0 && kh(high) > 2.0

    below = wavelength_quadrature_order(areas, low, c, 4)
    above = wavelength_quadrature_order(areas, high, c, 4)
    @test below.order == 2
    @test above.order == 4
    @test above.base_order == 4
    @test below.length ≈ 0.01
    @test below.kh ≈ kh(low)
    @test below.q1_max == 0.0 && below.q2_max == 2.0

    # The one-point tier is unreachable at the shipped cutoff and reachable once it is raised.
    @test wavelength_quadrature_order(areas, low, c, 4; q1_max=0.0).order == 2
    @test wavelength_quadrature_order(areas, low, c, 4; q1_max=kh(low) + 0.1).order == 1

    spread = [1.0e-4, 4.0e-4, 9.0e-4, 1.6e-3]
    stats = [wavelength_quadrature_order(spread, low, c, 4; mesh_stat=s).area
             for s in ("median", "p75", "p90", "max")]
    @test issorted(stats)
    @test stats[end] == maximum(spread)

    @test_throws "Unsupported wavelength mesh stat" wavelength_quadrature_order(
        areas, low, c, 4; mesh_stat="p50",
    )
    @test_throws "must be non-negative" wavelength_quadrature_order(
        areas, low, c, 4; q1_max=-1.0,
    )
    @test_throws "must exceed" wavelength_quadrature_order(
        areas, low, c, 4; q1_max=3.0, q2_max=2.0,
    )
    @test_throws "empty mesh" wavelength_quadrature_order(Float64[], low, c, 4)
end

@testset "Condensed per-order quadrature bundles" begin
    fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_map = build_conforming_interface_map(
        fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2,
    )

    cache = prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=2,
        regular_quadrature_orders=[2, 4],
        singular_order=CONDENSED_SINGULAR_ORDER,
    )
    @test sort(collect(keys(cache.quadrature_bundles))) == [2, 4]
    @test cache.base_quadrature_order == 2
    @test cache.singular_order == CONDENSED_SINGULAR_ORDER
    # The base bundle reuses what the underlying coupled cache already built, so a single-order
    # sweep is identical to using that cache directly.
    @test cache.quadrature_bundles[2].rule === cache.base.rule
    @test cache.quadrature_bundles[2].cpu_assembly_cache === cache.base.cpu_assembly_cache
    for order in (2, 4)
        bundle = cache.quadrature_bundles[order]
        @test bundle.order == order
        @test length(bundle.rule.points) == length(triangle_rule(Float64, order).points)
        # The bundle owns its rule: the CPU assembly guard compares rules by pointer identity.
        @test bundle.rule === bundle.cpu_assembly_cache.rule
    end
    # The base order is carried even when no frequency selects it: base 4 with every frequency
    # picking 1 or 2 must not fail at cache setup.
    absent_base = prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=4, regular_quadrature_orders=[1, 2],
        singular_order=CONDENSED_SINGULAR_ORDER,
    )
    @test sort(collect(keys(absent_base.quadrature_bundles))) == [1, 2, 4]
    @test absent_base.base_quadrature_order == 4
    # A q1 bundle must NOT take its identity matrices from the 1-point rule: that rule integrates
    # the P1xP1 mass matrix inexactly (area/9 uniform instead of area/6, area/12) and would
    # silently degrade the Burton-Miller 0.5*I term. Clamped to order 2, so it matches exactly.
    exact_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, absent_base.base.p1, absent_base.base.dp0,
        triangle_rule(Float64, 2), :p1, :p1,
    )
    wrong_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, absent_base.base.p1, absent_base.base.dp0,
        triangle_rule(Float64, 1), :p1, :p1,
    )
    @test absent_base.quadrature_bundles[1].identity_p1_p1 == exact_p1_p1
    @test wrong_p1_p1 != exact_p1_p1          # the 1-point rule really is wrong here
    # Clamping the identity rule must not disturb the regular rule, which stays at order 1.
    @test length(absent_base.quadrature_bundles[1].rule.points) == 1
    # Orders >= 2 are exact in exact arithmetic but not bitwise: different points and weights
    # round differently, so they agree to round-off rather than identically.
    @test absent_base.quadrature_bundles[4].identity_p1_p1 ≈ exact_p1_p1 rtol = 1e-13
    @test absent_base.quadrature_bundles[4].identity_p1_p1 != exact_p1_p1
    release_condensed_coupled_cache!(absent_base)

    @test_throws "must be positive" prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=2, regular_quadrature_orders=[0], singular_order=CONDENSED_SINGULAR_ORDER,
    )

    # Stale-cache guards, compared as integers rather than as rules. Both fire before assembly.
    for (order, singular, message) in (
        (4, CONDENSED_SINGULAR_ORDER, "base quadrature order does not match"),
        (2, CONDENSED_SINGULAR_ORDER + 1, "singular order does not match"),
    )
        @test_throws message build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=order, singular_order=singular, cache=cache,
        )
    end
    @test_throws "holds no quadrature bundle for order 3" build_condensed_coupled_system(
        fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
        quadrature_order=2, regular_quadrature_order=3,
        singular_order=CONDENSED_SINGULAR_ORDER, cache=cache,
    )

    # A multi-order cache must give each order exactly what a dedicated single-order cache gives.
    # This is what catches a bundle being shared or looked up by the wrong key. The Schur solver
    # never retains the assembled matrix, so the LU factors are the observable -- deterministic
    # for identical input in one process, and order-sensitive through `bem_lhs`.
    factors = Dict{Int,Any}()
    for order in (2, 4)
        dedicated = prepare_condensed_coupled_cache(
            fem_mesh, bem_mesh, interface_map;
            quadrature_order=order, singular_order=CONDENSED_SINGULAR_ORDER,
        )
        shared_system = build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=2, regular_quadrature_order=order,
            singular_order=CONDENSED_SINGULAR_ORDER, cache=cache,
        )
        dedicated_system = build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=order, singular_order=CONDENSED_SINGULAR_ORDER, cache=dedicated,
        )
        @test shared_system.regular_quadrature_order == order
        @test dedicated_system.regular_quadrature_order == order
        @test shared_system.factorization.factors == dedicated_system.factorization.factors
        factors[order] = copy(shared_system.factorization.factors)
        release_condensed_coupled_system!(shared_system)
        release_condensed_coupled_system!(dedicated_system)
        release_condensed_coupled_cache!(dedicated)
    end
    # Changing order must actually change the operator, or the equality above holds trivially.
    @test factors[2] != factors[4]

    release_condensed_coupled_cache!(cache)
end

@testset "Condensed coupled precision switches: Metal defaults" begin
    CC = BeatEngineCoupledCondensed
    names = ("BLAB_COUPLED_DENSE_FLOAT64", "BLAB_COUPLED_DENSE_REFINEMENT", "BLAB_COUPLED_FEM_FLOAT64")
    withenv((name => nothing for name in names)...) do
        for backend in (:cpu, :cuda, :rocm)
            @test !CC._dense_double_assembly(backend) && !CC._dense_refinement_enabled(backend)
            @test !CC._fem_float64_enabled(backend)
        end
        @test !CC._fem_float64_enabled() && !CC._dense_double_assembly()
        @test CC._dense_refinement_enabled(:metal) && !CC._dense_float64_enabled(:metal) && CC._dense_double_assembly(:metal)
        @test CC._fem_float64_enabled(:metal)
    end
    withenv("BLAB_COUPLED_DENSE_REFINEMENT" => "off", "BLAB_COUPLED_FEM_FLOAT64" => "AUTO", "BLAB_COUPLED_DENSE_FLOAT64" => "1") do
        @test !CC._dense_refinement_enabled(:metal) && CC._dense_float64_enabled(:cpu)
        @test CC._coupled_mode("BLAB_COUPLED_FEM_FLOAT64", :cpu) == :auto
    end
    @test_throws "BLAB_COUPLED_FEM_FLOAT64" withenv(() -> CC._fem_float64_enabled(:metal), "BLAB_COUPLED_FEM_FLOAT64" => "maybe")
end

@testset "Float32 dense LU with Float64 refinement (BLAB_COUPLED_DENSE_REFINEMENT)" begin
    Random.seed!(20260917)
    n = 240
    relative(reference, candidate) = norm(candidate - reference) / norm(reference)
    # Moderately conditioned, like the coupled systems (κ ~ 1e5): refinement converges.
    basis = qr(randn(ComplexF64, n, n)).Q
    singular_values = exp10.(range(0, -5; length=n))
    matrix = Matrix(basis * Diagonal(ComplexF64.(singular_values)) * qr(randn(ComplexF64, n, n)).Q')
    rhs = randn(ComplexF64, n, 3)
    reference = lu(matrix) \ rhs
    refined = BeatEngineCoupledCondensed.RefinedDenseLU(matrix)
    @test size(refined, 1) == n
    single = ComplexF64.(refined.factor \ ComplexF32.(rhs))
    @test relative(reference, single) > 1e-6
    solution = refined \ rhs
    @test eltype(solution) == ComplexF64
    @test relative(reference, solution) < 1e-10
    @test 1 <= refined.iterations <= 4
    @test isnothing(refined.fallback) && isnothing(refined.fallback_reason)
    @test relative(reference[:, 2], refined \ rhs[:, 2]) < 1e-10
    diagnostics = BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=refined,))
    @test diagnostics["dense_solver"] == "lu_float32_refined"
    @test diagnostics["dense_refinement_iterations"] == refined.iterations
    @test isnothing(diagnostics["dense_refinement_fallback_reason"])
    @test BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=lu(matrix),))["dense_solver"] == "lu_float64"

    # κ ~ 1e10 is beyond what a Float32 factor can refine: fall back to Float64 LU and say why.
    hard_values = exp10.(range(0, -10; length=n))
    hard = Matrix(basis * Diagonal(ComplexF64.(hard_values)) * qr(randn(ComplexF64, n, n)).Q')
    hard_refined = BeatEngineCoupledCondensed.RefinedDenseLU(hard)
    hard_solution = @test_logs (:warn, r"fell back to a Float64 LU") match_mode=:any hard_refined \ rhs
    @test hard_solution == lu(hard) \ rhs
    @test !isnothing(hard_refined.fallback)
    @test occursin("fell back to a Float64 LU", hard_refined.fallback_reason)
    @test hard_refined \ rhs == hard_solution
    hard_diagnostics = BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=hard_refined,))
    @test hard_diagnostics["dense_solver"] == "lu_float64_fallback"
    @test hard_diagnostics["dense_refinement_fallback_reason"] == hard_refined.fallback_reason

    # Accepted solves record their backward error (in units of the Float64 attainable one).
    @test 0 <= refined.backward_error <= 1
    @test diagnostics["dense_refinement_backward_error"] === nothing ||
          BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=refined,))["dense_refinement_backward_error"] <= 1

    # Right-hand sides of very different scale, including an all-zero column: every column must
    # meet the test on its own, and a zero column solves to exactly zero.
    mixed_rhs = hcat(rhs[:, 1] .* 1e-20, zeros(ComplexF64, n), rhs[:, 2] .* 1e20)
    mixed_reference = lu(matrix) \ mixed_rhs
    mixed = BeatEngineCoupledCondensed.RefinedDenseLU(matrix) \ mixed_rhs
    @test all(iszero, mixed[:, 2])
    @test relative(mixed_reference[:, 1], mixed[:, 1]) < 1e-10
    @test relative(mixed_reference[:, 3], mixed[:, 3]) < 1e-10

    # Entries outside the Float32 range cannot be narrowed: straight to Float64, with the reason.
    huge = matrix .* 1e40
    huge_refined = @test_logs (:warn, r"outside the Float32 range") BeatEngineCoupledCondensed.RefinedDenseLU(huge)
    @test isnothing(huge_refined.factor) && !isnothing(huge_refined.fallback)
    @test huge_refined \ rhs == lu(huge) \ rhs
    @test huge_refined.backward_error <= 1

    # Singular in Float32 but not in Float64: a pivot below Float32's range.
    tiny_pivot = Matrix{ComplexF64}(I, 4, 4)
    tiny_pivot[4, 4] = 1e-50
    tiny_refined = @test_logs (:warn, r"singular or not finite") BeatEngineCoupledCondensed.RefinedDenseLU(tiny_pivot)
    @test tiny_refined \ ones(ComplexF64, 4) == lu(tiny_pivot) \ ones(ComplexF64, 4)

    # Singular in Float64 too: a structured failure, not a silent result.
    singular = copy(matrix); singular[:, 1] .= 0
    @test_throws SingularException BeatEngineCoupledCondensed.RefinedDenseLU(singular)
    # Non-finite inputs are rejected.
    bad_matrix = copy(matrix); bad_matrix[1, 1] = NaN
    @test_throws ArgumentError BeatEngineCoupledCondensed.RefinedDenseLU(bad_matrix)
    bad_rhs = copy(rhs); bad_rhs[2, 2] = Inf
    @test_throws ArgumentError refined \ bad_rhs
end

@testset "Double-precision FEM matrices (BLAB_COUPLED_FEM_FLOAT64)" begin
    mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
    vertex_count = length(mesh32.vertices)
    mesh64 = VolumeMesh{Float64}(
        SVector{3,Float64}.(mesh32.vertices), mesh32.tetrahedra, mesh32.tetra_physical_tags, mesh32.boundary_faces,
        mesh32.boundary_physical_tags, mesh32.physical_names, mesh32.quadratic_tetrahedra, mesh32.quadratic_boundary_faces,
    )
    stiffness32, mass32 = assemble_p1_fem_matrices(mesh32)
    stiffness64, mass64 = assemble_p1_fem_matrices(mesh64)
    radiator_faces = findall(==(physical_tag(mesh32, 2, "Radiator")), mesh32.boundary_physical_tags)
    Random.seed!(20260917)
    loss = rand(Float32, vertex_count) .* 0.05f0
    wall = assemble_boundary_mass_matrix(mesh32, radiator_faces, collect(1:vertex_count))
    prepared = (
        stiffness=stiffness32,
        bulk_loss_factor_by_vertex=loss,
        wall_impedance_operators=[(matrix=wall, thickness_m=0.02f0, flow_resistivity_pa_s_per_m2=12000f0)],
    )
    store = Dict{Symbol,Any}()
    frequency, sound_speed, density = 20f0, 343f0, 1.21f0
    system = BeatEngineCoupledCondensed._fem_system_float64(store, mesh32, prepared, frequency, sound_speed, density)
    omega = 2pi * Float64(frequency)
    expected = assemble_fem_dynamic_stiffness(
        stiffness64, mass64, omega / Float64(sound_speed);
        bulk_loss_mass=spdiagm(0 => Float64.(loss)) * mass64,
    ) - BeatEngineCoupledCondensed.neumann_scale(Float64(density), omega) *
        miki_rigid_backed_surface_admittance(Float64(frequency), Float64(sound_speed), Float64(density),
                                             Float64(0.02f0), Float64(12000f0)) .*
        SparseMatrixCSC{Float64,Int}(wall)
    @test eltype(system) == ComplexF64
    @test system == expected
    # The matrices are cached by what they depend on, across frequencies and mesh objects, and are
    # rebuilt when a dependency changes in place (a persistent worker reusing its cache).
    cached = store[:matrices]
    BeatEngineCoupledCondensed._fem_system_float64(store, mesh32, prepared, 200f0, sound_speed, density)
    @test store[:matrices] === cached
    equal_mesh = deepcopy(mesh32)
    BeatEngineCoupledCondensed._fem_system_float64(store, equal_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] === cached
    moved_mesh = deepcopy(mesh32)
    moved_mesh.vertices[1] = moved_mesh.vertices[1] .+ 1f-4
    moved = BeatEngineCoupledCondensed._fem_system_float64(store, moved_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] !== cached && moved != system
    rebuilt = store[:matrices]
    prepared.bulk_loss_factor_by_vertex[2] += 0.01f0
    BeatEngineCoupledCondensed._fem_system_float64(store, moved_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] !== rebuilt
    prepared.bulk_loss_factor_by_vertex[2] -= 0.01f0
    # Matrices with another element structure than the cached system are refused.
    wrong_structure = (stiffness=sparse(1.0f0 * I, vertex_count, vertex_count), bulk_loss_factor_by_vertex=loss,
                       wall_impedance_operators=NamedTuple[])
    @test_throws "structure" BeatEngineCoupledCondensed._fem_system_float64(nothing, mesh32, wrong_structure, frequency, sound_speed, density)

    # Why: the air spring of a cavity with nothing retained, `Cᵀ A⁻¹ C` for a transducer surface load
    # on the whole volume (the sealed-chamber case). Through the Float32-assembled K it loses digits
    # like 1/k^2; through the double-precision matrices it matches the Float64 assembly exactly.
    surface_load = assemble_boundary_mass_matrix(mesh64, radiator_faces, collect(1:vertex_count)) * ones(vertex_count)
    surface = SparseMatrixCSC{ComplexF64,Int}(reshape(ComplexF64.(surface_load), vertex_count, 1))
    air_spring(fem_system) = transpose(surface) * (lu(SparseMatrixCSC{ComplexF64,Int}(fem_system)) \ Matrix(surface))
    lossless = (stiffness=stiffness32, bulk_loss_factor_by_vertex=zeros(Float32, vertex_count), wall_impedance_operators=NamedTuple[])
    reference = air_spring(assemble_fem_dynamic_stiffness(stiffness64, mass64, omega / Float64(sound_speed)))
    single = air_spring(SparseMatrixCSC{ComplexF64,Int}(
        assemble_fem_dynamic_stiffness(stiffness32, mass32, Float32(2pi) * frequency / sound_speed),
    ))
    double = air_spring(BeatEngineCoupledCondensed._fem_system_float64(nothing, mesh32, lossless, frequency, sound_speed, density))
    @test norm(single - reference) / norm(reference) > 1e-4
    @test norm(double - reference) / norm(reference) < 1e-12
end
