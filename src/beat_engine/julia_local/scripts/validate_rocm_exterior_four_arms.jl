using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

function relative_error(actual, reference)
    denominator = max(norm(reference), eps(real(eltype(reference))))
    return norm(actual - reference) / denominator
end

# Keep-alive diagnostic (off by default): when
# BLAB_VALIDATE_KEEP_ARM_MEMORY=1, each arm's device allocations are pinned in
# this list instead of released at arm end, so the allocator must hand the next
# arm fresh virtual addresses rather than recycling the previous arms' freed
# blocks. This isolates allocation-reuse hazards: a failure that only occurs
# when the allocator reuses freed memory does not reproduce under keep-alive.
# Each pinned arm prints an ARM_MEMORY_RETAINED line. Peak VRAM is about
# 2 GiB instead of ~0.5 GiB because all four arms' memory stays live.
const _KEPT_ARM_MEMORY = Any[]

function _validate_rocm_exterior_arm(convention::AbstractString, arm_index::Int)
    amdgpu = BeatEngineCore.AMDGPU_MODULE
    amdgpu === nothing && error("AMDGPU.jl did not load. Run this script with the julia_rocm project.")
    amdgpu.functional() || error("AMDGPU.functional() is false.")
    amdgpu.functional(:rocblas) || error("rocBLAS is not functional.")
    amdgpu.functional(:rocsolver) || error("rocSOLVER is not functional.")

    mesh_path = joinpath(@__DIR__, "..", "test_meshes", "sample.msh")
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "2"))
    rule = triangle_rule(Float32, regular_order)
    frequency_hz = 500.0f0
    sound_speed = 343.0f0
    k = Float32(2pi) * frequency_hz / sound_speed
    singular_cache = build_singular_correction_cache(mesh, singular_order)

    println("ROCM_EXTERIOR_ARM_START arm=$(arm_index) convention=$(convention)")
    println("fixture=$(mesh_path)")
    println("phasor_convention=$(convention)")
    println("faces=$(length(mesh.faces)) p1_dofs=$(p1.global_dof_count) dp0_dofs=$(dp0.global_dof_count) q=$(regular_order) s=$(singular_order)")
    println("device=$(amdgpu.device())")
    flush(stdout)

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=:off,
    )
    cpu_operators = nothing
    cpu_assembly_s = @elapsed begin
        cpu_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:cpu,
            singular_cache=singular_cache,
            cpu_cache=cpu_cache,
            symmetry_mode=:off,
        )
    end
    println("cpu_assembly_s=$(cpu_assembly_s)")
    flush(stdout)

    rocm_cache = build_rocm_regular_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=:off,
    )
    rocm_operators = nothing
    rocm_assembly_s = @elapsed begin
        rocm_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:rocm,
            device_cache=rocm_cache,
            singular_cache=singular_cache,
            rocm_assembly_mode=:native,
            symmetry_mode=:off,
        )
    end
    println("rocm_assembly_s=$(rocm_assembly_s) mode=$(rocm_operators.regular_assembly_mode)")
    flush(stdout)

    operator_errors = Dict(
        "single_layer" => relative_error(Array(rocm_operators.single_layer), cpu_operators.single_layer),
        "double_layer" => relative_error(Array(rocm_operators.double_layer), cpu_operators.double_layer),
        "adjoint_double_layer" => relative_error(Array(rocm_operators.adjoint_double_layer), cpu_operators.adjoint_double_layer),
        "hypersingular" => relative_error(Array(rocm_operators.hypersingular), cpu_operators.hypersingular),
    )

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    driven_tag = first(mesh.physical_tags)
    for face_index in eachindex(mesh.faces)
        if mesh.physical_tags[face_index] == driven_tag
            q_neumann[face_index] = ComplexF32(0, 1)
        end
    end

    cpu_pressure = nothing
    cpu_solve_s = @elapsed begin
        cpu_pressure = solve_burton_miller_neumann(
            cpu_operators,
            identity_p1_p1,
            identity_p1_dp0,
            q_neumann,
            k,
        )
    end
    identity_cache = build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
    rocm_pressure = nothing
    rocm_solve_s = @elapsed begin
        rocm_pressure = solve_burton_miller_neumann(rocm_operators, identity_cache, q_neumann, k)
    end
    release_rocm_burton_miller_identity_cache!(identity_cache)

    lhs, rhs_operator = burton_miller_neumann_matrices(
        cpu_operators,
        identity_p1_p1,
        identity_p1_dp0,
        k,
    )
    rhs = rhs_operator * q_neumann
    pressure_error = relative_error(rocm_pressure, cpu_pressure)
    cpu_residual = relative_error(lhs * cpu_pressure, rhs)
    rocm_residual = relative_error(lhs * rocm_pressure, rhs)

    eval_points = fibonacci_sphere(16, Float32(2.0))
    cpu_field_cache = build_field_evaluation_cache(mesh, rule)
    cpu_field = evaluate_galerkin_field_cpu(eval_points, mesh, cpu_pressure, q_neumann, k, cpu_field_cache)
    rocm_field_cache = build_rocm_field_evaluation_cache(cpu_field_cache)
    rocm_field = evaluate_galerkin_field_rocm(eval_points, mesh, rocm_pressure, q_neumann, k, rocm_field_cache)
    field_error = relative_error(rocm_field, cpu_field)

    println("operator_relative_errors=$(operator_errors)")
    println("cpu_solve_s=$(cpu_solve_s) rocm_solve_s=$(rocm_solve_s)")
    println("pressure_relative_error=$(pressure_error)")
    println("cpu_residual=$(cpu_residual) rocm_residual=$(rocm_residual)")
    println("field_relative_error=$(field_error)")
    flush(stdout)

    operator_tolerance = regular_order == 1 && singular_order == 2 ? 1.0f-6 : 1.0f-5
    maximum(values(operator_errors)) <= operator_tolerance || error("ROCm native operators differ from BEAT CPU.")
    pressure_error <= 5.0f-3 || error("ROCm pressure differs from BEAT CPU beyond tolerance.")
    rocm_residual <= 5.0f-3 || error("ROCm pressure residual exceeds tolerance.")
    field_error <= 5.0f-3 || error("ROCm field differs from BEAT CPU beyond tolerance.")

    if get(ENV, "BLAB_VALIDATE_KEEP_ARM_MEMORY", "0") == "1"
        # Keep-alive diagnostic: pin this arm's device objects so its blocks stay
        # out of the allocator's free list. singular_cache is pinned as well —
        # its scratch buffers have no explicit release in this script and would
        # otherwise be reclaimed by finalizers and handed back to the next arm.
        # The mid-arm identity_cache release above is deliberately untouched: it
        # happens identically in every arm of every baseline run, so its small
        # blocks are not the cross-arm differentiator under test.
        push!(_KEPT_ARM_MEMORY, rocm_operators, rocm_field_cache, rocm_cache, singular_cache)
        println("ROCM_EXTERIOR_ARM_MEMORY_RETAINED arm=$(arm_index)")
    else
        release_operator_storage!(rocm_operators)
        release_rocm_field_evaluation_cache!(rocm_field_cache)
        release_rocm_regular_assembly_cache!(rocm_cache)
    end
    println("ROCM_EXTERIOR_VALIDATION_OK arm=$(arm_index) convention=$(convention)")
    if get(ENV, "BLAB_VALIDATE_KEEP_ARM_MEMORY", "0") == "1"
        flush(stdout)
    end
end

function validate_rocm_exterior_four_arms()
    arm_sequence = get(
        ENV,
        "BLAB_VALIDATE_PHASOR_SEQUENCE",
        "exp(-i omega t),exp(+i omega t),exp(-i omega t),exp(+i omega t)",
    )
    conventions = [String(strip(value)) for value in split(arm_sequence, ',') if !isempty(strip(value))]
    length(conventions) == 4 || error(
        "BLAB_VALIDATE_PHASOR_SEQUENCE must list exactly 4 comma-separated conventions; got $(conventions).",
    )
    if get(ENV, "BLAB_VALIDATE_KEEP_ARM_MEMORY", "0") == "1"
        println("ROCM_EXTERIOR_4ARM_KEEP_ARM_MEMORY=1 (end-of-arm releases skipped; each arm forced onto fresh device addresses)")
        flush(stdout)
    end
    for (arm_index, convention) in enumerate(conventions)
        try
            with_phasor_convention(() -> _validate_rocm_exterior_arm(convention, arm_index), convention)
        catch err
            println("ROCM_EXTERIOR_ARM_FAIL arm=$(arm_index) convention=$(convention)")
            rethrow(err)
        end
    end
    println("ROCM_EXTERIOR_4ARM_VALIDATION_OK arms=$(length(conventions))")
end

validate_rocm_exterior_four_arms()
