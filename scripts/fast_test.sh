#!/usr/bin/env bash
set -euo pipefail

FUNCTION_FILE="${FUNCTION_FILE:-SAPA/output/res/function_test_res.json}"
PARAM_FILE="${PARAM_FILE:-SAPA/data/param_data_laser.json}"

python test_compute.py \
  --function_file "$FUNCTION_FILE" \
  --param_file "$PARAM_FILE" \
  "$@"
