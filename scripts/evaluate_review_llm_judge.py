#!/usr/bin/env python3
"""LLM-as-a-judge evaluation for generated PersonalWAB reviews.

This script compares SAPA and DPO review generations on review tasks only.
SAPA candidates are first reduced to one review per instruction by the same
self-consensus/centroid rule used in the paper code. The default judge is
reference-aware and focuses on GT-aligned review quality.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CRITERIA_SETS = {
    "ref3": [
        (
            "gt_faithfulness",
            "Does the generated review preserve the core claims, product experience, pros/cons, and details in the reference review?",
        ),
        (
            "sentiment_consistency",
            "Does the generated review match the sentiment polarity and intensity of the reference review, including mixed opinions?",
        ),
        (
            "style_length_consistency",
            "Does the generated review match the reference user's style, specificity, tone, and approximate length/detail level?",
        ),
    ],
    "ref5": [
        (
            "gt_faithfulness",
            "Does the generated review preserve the core claims, product experience, pros/cons, and details in the reference review?",
        ),
        (
            "sentiment_consistency",
            "Does the generated review match the sentiment polarity and intensity of the reference review, including mixed opinions?",
        ),
        (
            "style_length_consistency",
            "Does the generated review match the reference user's style, specificity, tone, and approximate length/detail level?",
        ),
        (
            "instruction_compatibility",
            "Does the review address explicit user instructions such as concise/detailed, balanced, pros/cons, or requested focus points while remaining consistent with the available evidence?",
        ),
        (
            "grounded_personalization",
            "Does the review use relevant personalized evidence from the instruction, target product, user history, and reference review instead of generic filler?",
        ),
    ],
}

CRITERIA = [name for name, _ in CRITERIA_SETS["ref3"]]
CRITERIA_DESCRIPTIONS = dict(CRITERIA_SETS["ref3"])


def set_criteria(criteria_set: str) -> None:
    global CRITERIA, CRITERIA_DESCRIPTIONS
    selected = CRITERIA_SETS[criteria_set]
    CRITERIA = [name for name, _ in selected]
    CRITERIA_DESCRIPTIONS = dict(selected)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Convert OpenAI SDK/Pydantic objects into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        return to_jsonable(obj.dict())
    return str(obj)


def truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def normalize_generation(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                out.append(item)
            else:
                out.append(json.dumps(item, ensure_ascii=False))
        return [x.strip() for x in out if str(x).strip()]
    return [json.dumps(value, ensure_ascii=False)]


def load_reference_reviews(reference_path: str) -> Dict[str, str]:
    """Load instruction -> ground-truth review from a few local result formats."""
    data = load_json(reference_path)
    refs: Dict[str, str] = {}

    if isinstance(data, dict) and "test" in data and isinstance(data["test"], list):
        for item in data["test"]:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction") or item.get("task")
            target = item.get("target")
            if instruction and isinstance(target, str):
                refs[instruction] = target
        return refs

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction") or item.get("task")
            target = item.get("target") or item.get("output") or item.get("answer")
            if instruction and isinstance(target, str):
                refs[instruction] = target
        return refs

    if isinstance(data, dict):
        for instruction, value in data.items():
            generations = normalize_generation(value)
            if generations:
                refs[instruction] = generations[0]
        return refs

    raise ValueError(f"Unsupported reference file structure: {reference_path}")


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    tasks_data = load_json(tasks_path)
    if isinstance(tasks_data, dict) and "test" in tasks_data:
        tasks = tasks_data["test"]
    elif isinstance(tasks_data, list):
        tasks = tasks_data
    else:
        raise ValueError(f"Unsupported task file structure: {tasks_path}")
    return [t for t in tasks if t.get("type") == "review"]


def merge_json_glob(pattern: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def get_nested(obj: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def product_context(task: Dict[str, Any], max_chars: int) -> str:
    product = task.get("target", {}).get("product_info", {})
    if not isinstance(product, dict):
        return ""
    fields = []
    for key in ["title", "main_category", "store", "price", "average_rating", "rating_number"]:
        value = product.get(key)
        if value not in (None, "", []):
            fields.append(f"{key}: {value}")
    return truncate_text("\n".join(fields), max_chars)


def reference_review(task: Dict[str, Any], max_chars: int) -> str:
    text = get_nested(task, ["target", "review", "text"], "")
    return truncate_text(text, max_chars)


def format_history(
    task: Dict[str, Any],
    user_history: Dict[str, Any],
    history_k: int,
    max_chars: int,
) -> str:
    user_id = task.get("user_id")
    timestamp = task.get("timestamp")
    hist = user_history.get(user_id, []) if user_id else []
    if not isinstance(hist, list):
        return ""

    filtered = []
    for item in hist:
        item_ts = get_nested(item, ["review", "timestamp"])
        if timestamp is None or item_ts is None or item_ts < timestamp:
            filtered.append(item)

    filtered.sort(key=lambda x: get_nested(x, ["review", "timestamp"], 0), reverse=True)
    lines = []
    for i, item in enumerate(filtered[:history_k], 1):
        pinfo = item.get("product_info", {}) if isinstance(item, dict) else {}
        review = item.get("review", {}) if isinstance(item, dict) else {}
        title = truncate_text(pinfo.get("title", ""), 140)
        rating = review.get("rating", "")
        text = truncate_text(review.get("text", ""), max_chars)
        if title or text:
            lines.append(f"{i}. title: {title}\nrating: {rating}\nreview: {text}")
    return "\n\n".join(lines)


class CentroidSelector:
    def __init__(self, model_name_or_path: str, device: str):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.F = F
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.device = device

    def embed(self, texts: List[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = self.torch.sum(token_embeddings * mask, dim=1) / self.torch.clamp(
            mask.sum(dim=1), min=1e-9
        )
        return self.F.normalize(pooled, p=2, dim=1)

    def select(self, candidates: List[str]) -> Tuple[str, int, List[float]]:
        candidates = [c for c in candidates if c.strip()]
        if not candidates:
            return "", -1, []
        if len(candidates) == 1:
            return candidates[0], 0, [1.0]
        embeddings = self.embed(candidates)
        sim = self.torch.mm(embeddings, embeddings.t())
        scores = sim.sum(dim=1)
        best_idx = int(self.torch.argmax(scores).item())
        return candidates[best_idx], best_idx, [float(x) for x in scores.detach().cpu().tolist()]


def build_judge_prompt(
    instruction: str,
    product: str,
    reference: str,
    history: str,
    review_a: str,
    review_b: str,
) -> str:
    criteria_text = "\n".join(
        f"- {criterion}: {CRITERIA_DESCRIPTIONS[criterion]}" for criterion in CRITERIA
    )
    score_schema = ",\n      ".join(f'"{criterion}": 1' for criterion in CRITERIA)
    winner_schema = ",\n    ".join(f'"{criterion}": "A|B|tie"' for criterion in CRITERIA)
    rationale_schema = ",\n    ".join(f'"{criterion}": "brief reason"' for criterion in CRITERIA)
    return f"""You are evaluating two generated product reviews for the same personalized shopping-agent task.

Score each generated review on a 1-5 integer scale for each criterion:
1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent.

Criteria:
{criteria_text}

Important:
- Evaluate Review A and Review B using the same standards. The method names are hidden from you.
- The reference review is evidence of the user's actual product experience, sentiment, and writing style. The user instruction and target product provide additional task context.
- Do not reward or penalize length by itself. A longer review can be better if its added details are supported and useful; a shorter review can be better if it faithfully captures the necessary content without omissions.
- Penalize unsupported claims, hallucinated product details, contradictions with the reference or instruction, irrelevant content, malformed text, ASIN/code spam, or major style drift.
- Do not require verbatim copying from the reference review. Paraphrases are acceptable when they preserve the same meaning, sentiment, and style.
- Judge Review A and Review B independently before deciding winners.

USER_INSTRUCTION:
{instruction}

TARGET_PRODUCT:
{product or "N/A"}

REFERENCE_REVIEW:
{reference or "N/A"}

RECENT_USER_REVIEW_HISTORY:
{history or "N/A"}

REVIEW_A:
{review_a}

REVIEW_B:
{review_b}

Return only valid JSON with this schema:
{{
  "scores": {{
    "A": {{
      {score_schema}
    }},
    "B": {{
      {score_schema}
    }}
  }},
  "winner_by_criterion": {{
    {winner_schema}
  }},
  "rationale": {{
    {rationale_schema}
  }}
}}"""


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def validate_judge_result(result: Dict[str, Any]) -> None:
    if "scores" not in result or not isinstance(result["scores"], dict):
        raise ValueError("Missing scores")
    for label in ["A", "B"]:
        if label not in result["scores"]:
            raise ValueError(f"Missing scores for {label}")
        for criterion in CRITERIA:
            if normalize_score(result["scores"][label].get(criterion)) is None:
                raise ValueError(f"Invalid {label}.{criterion}")


def judge_pair(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    sleep_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful, impartial evaluator. Return strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            result = parse_json_response(content)
            validate_judge_result(result)
            usage = to_jsonable(response.usage) if getattr(response, "usage", None) is not None else {}
            return result, usage
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_error}")


def deblind_scores(judge: Dict[str, Any], label_to_method: Dict[str, str]) -> Dict[str, Any]:
    out = {"scores": {}, "winner_by_criterion": {}, "rationale": judge.get("rationale", {})}
    for label, method in label_to_method.items():
        out["scores"][method] = judge["scores"][label]
    winners = judge.get("winner_by_criterion", {})
    for criterion in CRITERIA:
        winner_label = winners.get(criterion, "tie")
        if winner_label in label_to_method:
            out["winner_by_criterion"][criterion] = label_to_method[winner_label]
        else:
            out["winner_by_criterion"][criterion] = "tie"
    return out


def completed_instructions(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("instruction") and not rec.get("error"):
                done.add(rec["instruction"])
    return done


def load_detail_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("error"):
                records.append(rec)
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n": len(records),
        "mean_scores": {"sapa": {}, "dpo": {}},
        "win_counts": {},
        "win_rates": {},
    }
    for method in ["sapa", "dpo"]:
        for criterion in CRITERIA:
            vals = []
            for rec in records:
                val = normalize_score(rec.get("deblinded", {}).get("scores", {}).get(method, {}).get(criterion))
                if val is not None:
                    vals.append(val)
            summary["mean_scores"][method][criterion] = sum(vals) / len(vals) if vals else None

    for criterion in CRITERIA:
        counts = {"sapa": 0, "dpo": 0, "tie": 0}
        for rec in records:
            winner = rec.get("deblinded", {}).get("winner_by_criterion", {}).get(criterion, "tie")
            if winner not in counts:
                winner = "tie"
            counts[winner] += 1
        summary["win_counts"][criterion] = counts
        denom = max(len(records), 1)
        summary["win_rates"][criterion] = {k: v / denom for k, v in counts.items()}
    return summary


def write_csvs(out_dir: Path, summary: Dict[str, Any]) -> None:
    score_path = out_dir / "mean_scores.csv"
    with open(score_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *CRITERIA])
        for method in ["sapa", "dpo"]:
            writer.writerow([method, *[summary["mean_scores"][method].get(c) for c in CRITERIA]])

    win_path = out_dir / "win_rates.csv"
    with open(win_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["criterion", "sapa_win_rate", "dpo_win_rate", "tie_rate", "sapa_wins", "dpo_wins", "ties"])
        for criterion in CRITERIA:
            rates = summary["win_rates"][criterion]
            counts = summary["win_counts"][criterion]
            writer.writerow(
                [
                    criterion,
                    rates["sapa"],
                    rates["dpo"],
                    rates["tie"],
                    counts["sapa"],
                    counts["dpo"],
                    counts["tie"],
                ]
            )


def build_records(
    tasks: List[Dict[str, Any]],
    reference_reviews: Dict[str, str],
    sapa_outputs: Dict[str, Any],
    dpo_outputs: Dict[str, Any],
    user_history: Dict[str, Any],
    selector: CentroidSelector,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    records = []
    missing_reference = 0
    missing_sapa = 0
    missing_dpo = 0
    for idx, task in enumerate(tasks):
        instruction = task.get("task", "")
        if instruction not in reference_reviews:
            missing_reference += 1
            continue
        if instruction not in sapa_outputs:
            missing_sapa += 1
            continue
        if instruction not in dpo_outputs:
            missing_dpo += 1
            continue

        sapa_candidates = normalize_generation(sapa_outputs[instruction])
        dpo_candidates = normalize_generation(dpo_outputs[instruction])
        if not sapa_candidates or not dpo_candidates:
            continue

        selected_sapa, selected_idx, centroid_scores = selector.select(sapa_candidates)
        records.append(
            {
                "task_index": idx,
                "instruction": instruction,
                "user_id": task.get("user_id"),
                "timestamp": task.get("timestamp"),
                "target_review": truncate_text(reference_reviews[instruction], args.max_reference_chars),
                "product_context": product_context(task, args.max_product_chars),
                "history_context": format_history(
                    task,
                    user_history,
                    args.history_k,
                    args.max_history_review_chars,
                ),
                "sapa_review": truncate_text(selected_sapa, args.max_review_chars),
                "sapa_selected_index": selected_idx,
                "sapa_num_candidates": len(sapa_candidates),
                "sapa_centroid_scores": centroid_scores,
                "dpo_review": truncate_text(dpo_candidates[0], args.max_review_chars),
                "dpo_num_candidates": len(dpo_candidates),
            }
        )

    print(
        f"Prepared {len(records)} paired review tasks "
        f"(missing_reference={missing_reference}, missing_sapa={missing_sapa}, missing_dpo={missing_dpo})."
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sapa-path",
        default="/tmp/mine_centroid.json",
    )
    parser.add_argument("--dpo-path", default="/tmp/param_test_res_dpo_review_rerun.json")
    parser.add_argument(
        "--reference-path",
        default="SAPA/data/pre_sft_data.json",
    )
    parser.add_argument(
        "--tasks-path",
        default="PersonalWAB/envs/pwab/data/user_instructions.json",
    )
    parser.add_argument(
        "--user-history-glob",
        default="PersonalWAB/envs/pwab/data/user_history_part_*.json",
    )
    parser.add_argument("--criteria-set", choices=sorted(CRITERIA_SETS), default="ref3")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--history-k", type=int, default=5)
    parser.add_argument("--max-review-chars", type=int, default=1800)
    parser.add_argument("--max-reference-chars", type=int, default=1800)
    parser.add_argument("--max-product-chars", type=int, default=900)
    parser.add_argument("--max-history-review-chars", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    set_criteria(args.criteria_set)

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir or f"SAPA/judge_outputs/review_sapa_vs_dpo_{args.criteria_set}")
    detail_path = out_dir / "details.jsonl"

    tasks = load_tasks(args.tasks_path)
    reference_reviews = load_reference_reviews(args.reference_path)
    sapa_outputs = load_json(args.sapa_path)
    dpo_outputs = load_json(args.dpo_path)
    user_history = merge_json_glob(args.user_history_glob)

    selector = CentroidSelector(args.embedding_model, device=device)
    records = build_records(tasks, reference_reviews, sapa_outputs, dpo_outputs, user_history, selector, args)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]

    if args.dry_run:
        print(f"Dry run: {len(records)} records would be judged.")
        if records:
            rec = records[0]
            prompt = build_judge_prompt(
                rec["instruction"],
                rec["product_context"],
                rec["target_review"],
                rec["history_context"],
                rec["sapa_review"],
                rec["dpo_review"],
            )
            print(prompt[:4000])
        return

    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY is required, or pass --api-key.")

    from openai import OpenAI

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    done = set() if args.no_resume else completed_instructions(detail_path)
    print(f"Judging {len(records)} records; already completed={len(done)}.")

    for i, rec in enumerate(records, 1):
        if rec["instruction"] in done:
            continue

        methods = [("sapa", rec["sapa_review"]), ("dpo", rec["dpo_review"])]
        rng.shuffle(methods)
        label_to_method = {"A": methods[0][0], "B": methods[1][0]}
        method_to_label = {v: k for k, v in label_to_method.items()}

        prompt = build_judge_prompt(
            rec["instruction"],
            rec["product_context"],
            rec["target_review"],
            rec["history_context"],
            methods[0][1],
            methods[1][1],
        )

        try:
            judge, usage = judge_pair(
                client,
                args.judge_model,
                prompt,
                args.temperature,
                args.max_retries,
                args.sleep,
            )
            out = {
                **rec,
                "label_to_method": label_to_method,
                "method_to_label": method_to_label,
                "judge": judge,
                "deblinded": deblind_scores(judge, label_to_method),
                "usage": usage,
            }
            append_jsonl(detail_path, out)
            print(
                f"[{i}/{len(records)}] ok "
                f"gt_winner={out['deblinded']['winner_by_criterion'].get('gt_faithfulness')}"
            )
        except Exception as exc:  # noqa: BLE001
            append_jsonl(detail_path, {"instruction": rec["instruction"], "error": str(exc)})
            print(f"[{i}/{len(records)}] error: {exc}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    detail_records = load_detail_records(detail_path)
    summary = summarize(detail_records)
    dump_json(out_dir / "summary.json", summary)
    write_csvs(out_dir, summary)
    print(f"Wrote {detail_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Evaluated records: {summary['n']}")


if __name__ == "__main__":
    main()
