# End-to-end gate for the Metal sweep overlap on compiled exterior requests.
#
# Overlapping the next frequency's assembly with this one's solve may change
# when work happens, never what is computed. This solves the bundled example
# exterior request, retargeted at a test mesh, through the worker's
# compiled-system entry point: once sequentially and once overlapped at each
# depth. Every output must be bit-identical and carry its own frequency.
#
#   julia --threads=4 --project=src/beat_engine/julia_metal \
#       src/beat_engine/julia_local/scripts/validate_metal_exterior_pipeline.jl
#
#   BLAB_VALIDATE_SYMMETRY  off | x | xy (default off)
#   BLAB_VALIDATE_MESH      bundled fixture (default by symmetry: sample, sample_half, sample_quarter)
#   BLAB_VALIDATE_STEPS     frequencies from 100 Hz to 20 kHz (default 12)
using JSON

const JULIA_LOCAL = normpath(joinpath(@__DIR__, ".."))
const SOLVER = joinpath(JULIA_LOCAL, "coupled_solver.jl")
const EXAMPLE = normpath(joinpath(JULIA_LOCAL, "..", "beat_contract", "example-exterior-request.json"))
const DEFAULT_MESH = Dict("off" => "sample.msh", "x" => "sample_half.msh", "xy" => "sample_quarter.msh")

function exterior_request(mesh_name, symmetry, steps)
    request = JSON.parsefile(EXAMPLE)
    request["compiled_system"]["meshes"][1]["file"] = joinpath(JULIA_LOCAL, "test_meshes", mesh_name)
    request["frequencies_hz"] = [exp10(x) for x in range(log10(100.0), log10(20000.0); length=steps)]
    request["solver_options"]["bem_backend"] = "metal"
    request["solver_options"]["symmetry"] = symmetry
    points = [[2.0 * sin(angle), 0.0, 2.0 * cos(angle)] for angle in range(0, pi; length=37)]
    request["outputs"] = [
        Dict("id" => "pressure", "quantity" => "exterior_pressure", "target_ids" => [],
             "options" => Dict("points_m" => points)),
        Dict("id" => "boundary", "quantity" => "bem_boundary_pressure", "target_ids" => [], "options" => Dict()),
    ]
    return request
end

function solve(request, env)
    input, output = tempname() * ".json", tempname() * ".jsonl"
    write(input, JSON.json(request))
    cmd = `$(Base.julia_cmd()) --threads=$(Threads.nthreads()) --startup-file=no --project=$(Base.active_project()) $SOLVER`
    run(pipeline(addenv(cmd, env...); stdin=input, stdout=output))
    return [JSON.parse(line) for line in eachline(output) if startswith(line, "{")]
end

payloads(row) = Dict(q["quantity"] => JSON.json(q["values"]) for q in row["quantities"])

function validate_metal_exterior_pipeline()
    Threads.nthreads() > 1 || error("The overlap needs a second Julia thread; start julia with --threads.")
    symmetry = get(ENV, "BLAB_VALIDATE_SYMMETRY", "off")
    mesh_name = get(ENV, "BLAB_VALIDATE_MESH", DEFAULT_MESH[symmetry])
    steps = parse(Int, get(ENV, "BLAB_VALIDATE_STEPS", "12"))
    request = exterior_request(mesh_name, symmetry, steps)
    println("mesh=$(mesh_name) symmetry=$(symmetry) steps=$(steps) threads=$(Threads.nthreads())")

    sequential = solve(request, ("BLAB_METAL_PIPELINE" => "0",))
    length(sequential) == steps || error("Sequential solve returned $(length(sequential)) of $(steps) results.")
    all(row["diagnostics"]["metal_pipeline"] == false for row in sequential) ||
        error("The sequential reference ran overlapped; BLAB_METAL_PIPELINE=0 was not honoured.")
    failures = 0
    for depth in 1:4
        overlapped = solve(request, ("BLAB_METAL_PIPELINE" => "1", "BLAB_METAL_PIPELINE_DEPTH" => string(depth)))
        mislabelled = count(zip(sequential, overlapped)) do (a, b)
            a["freq_hz"] != b["freq_hz"]
        end
        differing = [
            a["freq_hz"] for (a, b) in zip(sequential, overlapped) if payloads(a) != payloads(b)
        ]
        ran_overlapped = all(row["diagnostics"]["metal_pipeline"] == true &&
                             row["diagnostics"]["sweep_assembly_lookahead"] == depth for row in overlapped)
        ok = length(overlapped) == steps && mislabelled == 0 && isempty(differing) && ran_overlapped
        ok || (failures += 1)
        println("depth=$(depth): results=$(length(overlapped))/$(steps) mislabelled=$(mislabelled) " *
                "differing frequencies=$(length(differing)) overlapped=$(ran_overlapped)  $(ok ? "OK" : "FAILED")")
        for (a, b) in zip(sequential, overlapped)
            a["freq_hz"] in differing || continue
            # A GMRES that hits its wall-clock ceiling under load falls back to
            # the LU and returns a different, equally valid, answer; say so.
            println("  $(a["freq_hz"]) Hz: sequential $(a["diagnostics"]["linear_solver"]), " *
                    "overlapped $(b["diagnostics"]["linear_solver"])")
        end
    end
    failures == 0 || error("Metal exterior overlap gate failed at $(failures) of 4 depths.")
    println("PASS: overlapped exterior sweeps are bit-identical to sequential at depths 1-4.")
end

validate_metal_exterior_pipeline()
