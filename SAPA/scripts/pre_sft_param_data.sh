python prepare_param_data.py \
  --instruction_file "${INSTRUCTION_FILE:-data/user_instructions.json}" \
  --retrieval_results_file "${RETRIEVAL_RESULTS_FILE:-data/processed/baseline_results_top500.jsonl}" \
  --history_file "${HISTORY_FILE:-data/processed/user_history.json}" \
  --product_data_file "${PRODUCT_DATA_FILE:-data/processed/all_products.json}" \
  --output_file "${PARAM_DATA_OUTPUT_FILE:-data/final/pre_sft_data.json}" \
  --llama_tokenizer_path "${PWAB_BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}" \
  --sim_model_path "${PWAB_SIM_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
  --mem_token_length "${MEM_TOKEN_LENGTH:-768}" \
  --save_interval "${SAVE_INTERVAL:-50}"
