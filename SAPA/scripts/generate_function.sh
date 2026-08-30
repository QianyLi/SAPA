#!/usr/bin/env bash
set -euo pipefail

accelerate launch test_llama.py \
    --model_path "${FUNCTION_MODEL_PATH:?Set FUNCTION_MODEL_PATH to a trained function checkpoint}" \
    --base_model "${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}" \
    --data_path "${FUNCTION_DATA_PATH:-data/function_data.json}" \
    --bf16 \
    --split test \
    --test_on function \
    --batch_size 8 \
    --max_new_tokens 32 \
    --res_file "${FUNCTION_RES_FILE:-output/res/function_test_res.json}"
