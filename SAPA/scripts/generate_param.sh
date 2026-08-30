#!/usr/bin/env bash
set -euo pipefail

python test_llama_flow_laser.py \
  --input_file "${PARAM_INPUT_FILE:-data/best/param_raw.json}" \
  --index_file "${PWAB_BGE_INDEX:-../PersonalWAB/envs/pwab/functions/search/faiss_dense_bge_m3.index}" \
  --all_products_jsonl "${PWAB_PRODUCTS_JSONL:-../PersonalWAB/envs/pwab/functions/data/Products/all_products.jsonl}" \
  --output_file "${PARAM_OUTPUT_FILE:-data/best/param_data_k10_return1_top6_qwen25_prompt_laser.json}" \
  --model_path "${BGEM3_MODEL:-BAAI/bge-m3}"
