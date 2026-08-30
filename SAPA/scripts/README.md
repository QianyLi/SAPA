# SAPA scripts

The scripts in this directory are thin wrappers around the Python training and
inference programs in `SAPA/`.

## Supported wrappers

- `pre_sft_func_data.sh` → `prepare_function_data.py`
- `pre_sft_param_data.sh` → parameter-data preparation entry point
- `finetune_function_param.sh` → `finetune_llama.py`
- `generate_function.sh` → `test_llama.py`
- `generate_param_laser.sh` → `test_llama_flow_laser.py`
- `../laser_n10.py` → N-sample BGE-M3 + FAISS expansion with RRF

The training and function-generation wrappers require checkpoint/path
environment variables. See the root README for a complete command sequence;
in particular, set `DATA_PATH`, `FUNCTION_MODEL_PATH`, and `GPUS` before
launching them.
- `analyze_recall.sh` → `analyze_recall.py`
- `run_singleturn_sapa.sh` → SAPA single-turn evaluation via `run.py`

## Legacy wrappers

`cosine_top500.sh` and `bge_rerank.sh` refer to experimental helper programs
that are not part of this source checkout. They are retained for historical
reference only and should not be used as supported entry points.
`ablation/run_pipeline.sh` is the intended recipe, but it also requires local
retrieval helpers and data; see `ablation/README.md` for the exact limitation.
