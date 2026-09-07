# BEAT Engine

Numerical BEM, FEM, and lumped electroacoustic solving, extracted from Boundary Lab.
The `beat-engine` Python distribution supplies a standard-library worker client,
versioned wire contracts, Julia CPU/CUDA/ROCm environments, and numerical tests.
Project authoring, mesh compilation, Qt, and application preferences remain with
the consuming application. Metal is not yet implemented or advertised.

## Local installation

Use Python 3.11+ and install Julia separately (the current qualification uses Julia
1.12). A wheel does not install Julia or GPU drivers.

```text
python -m pip install -e ".[dev]"
python -m beat_engine instantiate --backend cpu
python -m beat_engine doctor --backend cpu --threads 2
python -m beat_engine paths --backend cpu
```

For CUDA or ROCm, instantiate the matching backend and configure the platform SDK
before running `doctor`. Availability is checked by the worker; it is not inferred
from the package being installed. Importing the package does not install packages,
launch Julia, or modify the environment.

## Public Python API

```python
from beat_engine import EngineWorker, engine_paths

paths = engine_paths("cpu")
worker = EngineWorker(
    julia_executable="julia", solver_script=paths.system_solver,
    julia_project=paths.project, julia_threads=2,
)
try:
    worker.ensure_started()
    print(worker.worker_info)
    # request.json contains a validated compiled-system request with local assets.
    # for event in worker.submit(request_path): ...
finally:
    worker.terminate()
```

`EngineWorker`, `WorkerPool`, `EnginePaths`, and `engine_paths` are the public runtime
API. `beat_engine.beat_contract` exposes model-independent contract validators and
version constants. Submission events are raw JSON records; clients own result
models and display conversion. Drain or close event iterators. Worker paths come
from the installed package, including editable contributor checkouts.

Read [the compiled-system contract](docs/BEAT%20Compiled%20System%20Contract.md),
[worker protocol](docs/BEAT%20Worker%20Protocol.md), and
[provenance format](docs/BEAT%20Run%20Provenance.md).

## Tests and releases

```text
python -m pytest
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/runtests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```

The required CPU reference gate has 464 checks including analytical complex
pressure, independent excitations, symmetry, FEM modes, and coupled transducer
comparisons. See its [coverage notes](src/beat_engine/julia_local/tests/README.md).
The manual Accelerator qualification workflow uses self-hosted runners labeled
`cuda` or `rocm`, with Python, Julia, and the matching SDK preinstalled. It requires
the requested backend to be functional before running the numerical gates.
GPU qualification remains hardware-specific; CPU success does not qualify GPU
backends. Some historical research scripts and the optional noncubic-cavity test
still require Boundary Lab's extended fixtures; they are not installed runtime
requirements or part of the portable release gate.

The first package candidate is `0.1.0rc1`. CI builds a wheel after independent
Python and CPU checks on Windows, Linux, and macOS. The manually dispatched release
workflow requires a successful CI run for the tag's exact commit before attaching
wheel/sdist artifacts to a GitHub prerelease. No PyPI publication is configured.

History and original authorship were retained from Boundary Lab. See
[extraction provenance](docs/EXTRACTION.md) and the repository [LICENSE](LICENSE).
