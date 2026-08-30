# PersonalWAB / SAPA

## 🔍 Overview

**SAPA** (a Situation-Augmented Personalized web Agent) is a framework developed to adapt LLMs to the personalized Web agent task. 

> For more details, refer to our paper accepted to **emnlp 2026 findings**: [SAPA: a Situation-Augmented Personalized web Agent].

## ⚙️ Installation

### Requirements

- Python 3.11
- PyTorch 2.4.1
- CUDA is recommended for model training
- OpenJDK (required by Pyserini/Lucene)

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```


## 📊 PersonalWAB Benchmark

![](https://hongrucai.github.io/images/personalwab.png)

The **PersonalWAB** benchmark includes:

- **Personalized User Data**: 1,000 diverse user profiles and 40,000+ web behaviors, originated from real-world data.
- **User Instructions**: 9,000+ highly personalized natural language instructions tailored to each user's profile.
- **User Simulator**: Simulates interactions aligned with user profiles and historical behaviors.
- **Evaluation Paradigms**:  Single-turn track tests for isolated tasks and multi-turn for more complex interactions.

The dataset is expected under `PersonalWAB/envs/pwab/data/`.

### Task Description

**Personalized Search**: Personalized product search using user instructions and behavioral history.  
**Personalized Recommendation**: Recommend items based on implicit preferences.  
**Personalized Review Generation**: Generate reviews aligned with user preferences.

## Reproduction

The commands below are written for a fresh checkout from the repository root.
The benchmark data, model weights, vector indexes, and generated outputs are
not bundled in the source release. Download or generate them locally first.

### 1. Install and configure

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY for GPT-based agents.
source scripts/set_api_key.sh
```

For model-based runs, set paths or Hugging Face model identifiers for your
environment. The defaults are public model identifiers; gated models may also
require `huggingface-cli login`.

```bash
export PWAB_BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
export PWAB_SIM_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export BGEM3_MODEL="BAAI/bge-m3"
```

### 2. Prepare PersonalWAB data

Download the PersonalWAB package from the
[project page](https://hongrucai.github.io/PersonalWAB/download). Place the
JSON files below `PersonalWAB/envs/pwab/data/`:

```text
all_products_part_*.json
user_history_part_*.json
user_instructions.json
user_profiles.json
```

### 3. Run the PersonalWAB baselines

Single-turn evaluation uses one tool call per task and is the simplest smoke
test:

```bash
bash scripts/run_singleturn_sapa.sh
```

The SAPA wrapper uses a local fine-tuned parameter generator; set
`SAPA_MODEL_PATH`, `SAPA_FUNCTION_FILE`, and `SAPA_PARAM_FILE` before running.
For the API-based vanilla function-calling baseline, invoke `run.py` directly
with `--agent_strategy function_calling`. Results are written to the run log
directory selected by `run.py`.

Multi-turn evaluation requires the local search indexes and a generated
function-selection file. After those artifacts are available, run:

```bash
export FUNCTION_FILE="SAPA/output/res/function_test_res.json"
bash scripts/run_multiturn.sh
```

Use `python run.py --help` to change the model, agent strategy, task split,
concurrency, memory mode, or step limit.

### 4. Reproduce the SAPA data and inference pipeline

The following commands are the supported order for the files present in this
checkout. Run them from `SAPA/`:

```bash
cd SAPA

# Prepare function-selection training data.
bash scripts/pre_sft_func_data.sh

# Prepare parameter-generation SFT data with memory and candidates.
bash scripts/pre_sft_param_data.sh
```

Train a parameter generator with a model available on your machine. To train a
function generator instead, set `TRAIN_ON=function` and use a separate output
directory. The checked-in wrapper is configurable through the variables below:

```bash
TRAIN_ON="param" \
bash scripts/finetune_function_param.sh
```

For the function-selection checkpoint used by `generate_function.sh`, run the
same wrapper with `TRAIN_ON="function"` and a separate output directory:

```bash
TRAIN_ON="function" \
bash scripts/finetune_function_param.sh
```

Generate function predictions and expanded parameter predictions:

```bash
export FUNCTION_MODEL_PATH="output/sapa_function/<checkpoint>"
bash scripts/generate_function.sh

# Generate raw parameter candidates with the parameter checkpoint.
accelerate launch test_llama_flow_recommend.py \
  --model_path output/sapa_param/<checkpoint> \
  --base_model "$PWAB_BASE_MODEL" \
  --data_path data/pre_sft_data.json \
  --tool_file output/res/function_test_res.json \
  --res_file data/param_raw.json \
  --split test --test_on param --bf16 --batch_size 4 \
  --max_new_tokens 512 --memory_token_length 768

export PWAB_BGE_INDEX="../PersonalWAB/envs/pwab/functions/search/faiss_dense_bge_m3.index"
export PWAB_PRODUCTS_JSONL="../PersonalWAB/envs/pwab/functions/data/Products/all_products.jsonl"
export PARAM_INPUT_FILE="data/param_raw.json"
bash scripts/generate_param.sh
```

Evaluate generated results from the repository root. Override paths when your
files use different names:

```bash
cd ..
bash scripts/fast_test.sh \
  --task_file SAPA/data/user_instructions.json \
  --all_products SAPA/data/all_products.json
```

### 5. Amazon transfer and ablation experiments

The ablation scripts process Amazon Reviews 2023 subsets and require raw data
and generated instructions under `ablation/data/amazon/`. See
[`ablation/README.md`](ablation/README.md) for the expected layout. The main
recipe is:

```bash
bash ablation/run_pipeline.sh office
# or: beauty / electronics
```

## 🧠 SAPA implementation

SAPA combines two components: **Situational Intent Grounding**, which injects
historical context and a filtered candidate pool, and **Task-Driven Semantic
Expansion**, which samples and aggregates multiple task-specific outputs. The
training and inference wrappers under `SAPA/scripts/` implement this pipeline;
the complete command order is documented in [Reproduction](#reproduction).

The repository also contains explicitly labelled vanilla-baseline and
compatibility code used to reproduce comparisons with prior work. Those files
are not part of the SAPA method and are kept separate from the SAPA-facing
entry points.

## 📚 Citation

If you use SAPA, please cite the accompanying paper:

```bibtex
@inproceedings{sapa2026,
  title     = {SAPA: A Situation-Augmented Personalized Agent for the Web},
  author    = {Li, Qianyue and Ji, Bin and Liu, Xiaodong and Li, Shasha
               and Li, Xiaopeng and Wang, Yueyu and Hong, Xinran and Ma, Jun
               and Yu, Jie},
  booktitle = {Findings of the Association for Computational Linguistics},
  year      = {2026}
}
```

## 📄 License

The source code is released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license;
see [`LICENSE`](LICENSE). The PersonalWAB and Amazon-derived datasets may have
separate terms. Download and redistribute them only when their original terms
permit it. Components derived from tau-bench retain the upstream MIT license
terms described in the repository documentation.

## 📬 Contact

For questions, please open an issue in your project repository or contact the
SAPA authors through the corresponding-author channel listed in the paper.
