# Performance-only calibration: never reuse numerical data across frequencies.
function deploy_rhs_memory_fits(operator_bytes, free_bytes)
    operator_bytes > 0 && operator_bytes * 1.25 + 256 * 1024^2 <= free_bytes
end

function deploy_use_rhs_operator(calibration, operator_bytes, free_bytes)
    calibration === nothing && return false
    deploy_rhs_memory_fits(operator_bytes, free_bytes) || return false
    (; build_s, apply_s, direct_s, applications) = calibration
    all(isfinite, (build_s, apply_s, direct_s)) || return false
    min(build_s, apply_s, direct_s) >= 0 || return false
    applications > 1 || return false
    # Demand a measured margin over the calibration solve's application count.
    return build_s + applications * apply_s < 0.9 * applications * direct_s
end
