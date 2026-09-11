const NEGATIVE_TIME_PHASOR = "exp(-i omega t)"
const POSITIVE_TIME_PHASOR = "exp(+i omega t)"

# Requests are serialized by each worker. Preserve the original convention for
# unversioned callers and frozen reference tests; new clients negotiate explicitly.
const ACTIVE_PHASOR_CONVENTION = Ref(NEGATIVE_TIME_PHASOR)
phasor_convention() = ACTIVE_PHASOR_CONVENTION[]
propagation_sign() = phasor_convention() == NEGATIVE_TIME_PHASOR ? 1 : -1
outgoing_wavenumber(k) = oftype(k, propagation_sign()) * abs(k)
neumann_scale(density, omega) = complex(zero(omega), propagation_sign() * density * omega)
time_derivative(omega) = complex(zero(omega), -propagation_sign() * omega)
burton_miller_coupling(k, cap=zero(k)) =
    complex(zero(k), oftype(k, propagation_sign()) * burton_miller_coupling_scale(k, cap))

function with_phasor_convention(f::Function, convention)
    convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR) ||
        error("Unsupported phasor convention: $(repr(convention))")
    previous = ACTIVE_PHASOR_CONVENTION[]
    ACTIVE_PHASOR_CONVENTION[] = convention
    try
        return f()
    finally
        ACTIVE_PHASOR_CONVENTION[] = previous
    end
end
