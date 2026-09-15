# Opt-in CUDA worker cleanup reuse

The compiled-system persistent worker preserves its existing aggressive cleanup
after every solve unless a request explicitly selects `cuda_reuse`.

This feature is intended to be utilized for rapid, successive solving use-cases such as optimizers and server-work queueing for multiple clients.

```json
{
  "solver_options": {
    "bem_backend": "cuda",
    "worker_cleanup": {
      "policy": "cuda_reuse",
      "max_requests": 8,
      "min_free_fraction": 0.2
    }
  }
}
```

Other solver options and the normal request contract are omitted here. The ready
event advertises `worker_cleanup_policies`. Clients should check for `cuda_reuse`
before requesting it; older workers do not implement this extension.

On a successful opted-in CUDA solve, the worker releases request-owned field
caches while retaining CUDA allocator and library caches. It skips explicit
full GC and driver reclamation at that boundary. Julia's normal GC and CUDA's
allocation-pressure reclamation remain active. This changes resource lifetime,
not assembly, solve, field evaluation, precision, or pressure serialization.

Full historical cleanup still runs:

- Every `max_requests` successful solves since the previous full cleanup,
  counting the current request (default 8; allowed 1–1024).
- When CUDA device free memory is at or below `min_free_fraction` of total
  device memory (default 20%; allowed strictly between 0 and 1), or memory
  information cannot be read. This is device-wide free memory, including effects
  of other processes, not a count of reusable bytes in the worker's allocator.
- On cancellation, request failure, or any request using the default policy.


The setting is request-local. Switching to a request that omits the option
immediately restores historical post-solve cleanup and resets the counter.
Source-request workers and `bem_field` operations are unchanged. Opt-in on
non-CUDA compiled solves is rejected. Invalid policies or bounds fail before
solving and use the existing failure cleanup path.

Opted-in terminal solve events include `worker_cleanup` diagnostics:
`policy`, `reason` (`reuse`, `interval`, `memory_pressure`, `memory_unknown`, or
`cancelled`), `requests_since_cleanup`, `free_fraction_before`, and `seconds`.
The completion event still follows cleanup; clients need no new event sequencing.

Run the standalone decision tests with:

```powershell
julia src/beat_engine/julia_local/tests/worker_cleanup_tests.jl
```
