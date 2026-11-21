#!/usr/bin/env bash
# H-mátrix tol 1e-3 a Float64 referenciához képest (valódi modell, 20 kHz).
export HMAT_DUMP_DIR=$(ls -td "$(dirname "$0")"/results/*-hmat-real-f64/dump | head -1)
exec bash "$(dirname "$0")/37_hmat_dump_check_mac.sh"
