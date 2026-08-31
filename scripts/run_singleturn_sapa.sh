#!/usr/bin/env bash
set -euo pipefail

python run.py \
--env pwab \
--model finetune/llama \
--user_mode naive \
--user_model gpt-4o-mini \
--agent_strategy sapa \
--agent_memory taskspe \
--memory_length 100 \
--task_split test \
--max_concurrency 1 \
--max_steps 1 \
--end_index -1 \
--sapa_generate "${SAPA_GENERATE:-1}" \
--sapa_function_file "${SAPA_FUNCTION_FILE:-SAPA/output/res/function_test_res.json}" \
--sapa_param_file "${SAPA_PARAM_FILE:-SAPA/output/res/param_data.json}" \
--sapa_model_path "${SAPA_MODEL_PATH:?Set SAPA_MODEL_PATH to a local checkpoint}"
