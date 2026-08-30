#!/usr/bin/env bash
set -euo pipefail

deepspeed --include="localhost:${GPUS:-0,2,3,4}" --master_port="${MASTER_PORT:-29997}" finetune_llama.py \
    --data_path "${DATA_PATH:-data/best/pre_sft_data_k10_return1_top6_qwen25_prompt.json}" \
    --function_data_path "${FUNCTION_DATA_PATH:-data/function_data.json}" \
    --output_dir "${OUTPUT_DIR:-output/sapa_param}" \
    --model_name "${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}" \
    --train_epoch 10 \
    --learning_rate 3e-4 \
    --train_batch_size 1 \
    --source_length 2048 \
    --warmup_ratio 0.1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 10 \
    --deepspeed_config config/llama_ds_config.json \
    --gradient_accumulation_steps 16 \
    --temperature 1.0 \
    --bf16 \
    --train_on "${TRAIN_ON:-param}"
