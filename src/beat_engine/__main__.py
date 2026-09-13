"""Explicit runtime setup and diagnostics; no dependency installation on import."""

import argparse
import json
import subprocess
from dataclasses import asdict

from . import EngineWorker, backend_catalog, engine_paths


def main() -> None:
    parser = argparse.ArgumentParser(prog="beat-engine")
    parser.add_argument("command", choices=("backends", "paths", "instantiate", "doctor"))
    parser.add_argument("--backend", choices=tuple(info.backend_id for info in backend_catalog()), default="cpu")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--threads", default="auto")
    args = parser.parse_args()
    if args.command == "backends":
        print(json.dumps([asdict(info) for info in backend_catalog()], indent=2))
        return
    paths = engine_paths(args.backend)
    if args.command == "paths":
        print(json.dumps({key: str(value) for key, value in asdict(paths).items()}, indent=2))
    elif args.command == "instantiate":
        subprocess.run(
            [args.julia, f"--project={paths.project}", "--startup-file=no", "-e", "using Pkg; Pkg.instantiate()"],
            check=True,
        )
    else:
        worker = EngineWorker(
            julia_executable=args.julia,
            solver_script=paths.system_solver,
            julia_threads=args.threads,
            julia_project=paths.project,
        )
        try:
            worker.ensure_started()
            print(json.dumps(worker.worker_info, indent=2))
        finally:
            worker.terminate()


if __name__ == "__main__":
    main()
