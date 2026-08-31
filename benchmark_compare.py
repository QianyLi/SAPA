'Documentation.'

import os
import sys
import json
import time
import random
import argparse
import threading
import numpy as np
from collections import defaultdict
from datetime import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

THREAD_TIMER = threading.local()

def _reset_timer():
    THREAD_TIMER.llm = 0.0
    THREAD_TIMER.tool = 0.0
    THREAD_TIMER.bm25 = 0.0
    THREAD_TIMER.rerank = 0.0
    THREAD_TIMER.faiss = 0.0
    THREAD_TIMER.env_step = 0.0

def _snap_timer():
    return {
        "llm":      getattr(THREAD_TIMER, "llm", 0.0),
        "tool":     getattr(THREAD_TIMER, "tool", 0.0),
        "bm25":     getattr(THREAD_TIMER, "bm25", 0.0),
        "rerank":   getattr(THREAD_TIMER, "rerank", 0.0),
        "faiss":    getattr(THREAD_TIMER, "faiss", 0.0),
        "env_step": getattr(THREAD_TIMER, "env_step", 0.0),
    }


_USAGE_ALLOWED_KEYS = {"completion_tokens", "prompt_tokens", "total_tokens"}

def _sanitize_usage(usage_obj):
    if usage_obj is None: return usage_obj
    try: d = dict(usage_obj)
    except Exception: return usage_obj
    return {k: d.get(k, 0) for k in _USAGE_ALLOWED_KEYS}


@contextmanager
def patch_openai_and_env(env):
    from PersonalWAB.agents import gpt_function_calling_agent as fc_mod
    from PersonalWAB.agents import rise_agent as rise_mod

    restorers = []


    orig_step = env.step
    def timed_step(action):
        t0 = time.time()
        try:
            return orig_step(action)
        finally:
            dt = time.time() - t0
            THREAD_TIMER.env_step = getattr(THREAD_TIMER, "env_step", 0.0) + dt
            THREAD_TIMER.tool     = getattr(THREAD_TIMER, "tool",     0.0) + dt
    env.step = timed_step
    restorers.append(lambda: setattr(env, "step", orig_step))


    seen = set()
    for module in (fc_mod, rise_mod):
        c = getattr(module, "client", None)
        if c is None or id(c) in seen: continue
        seen.add(id(c))
        orig_create = c.chat.completions.create
        def make_timed(orig=orig_create):
            def _timed(*args, **kwargs):
                t0 = time.time()
                resp = orig(*args, **kwargs)
                THREAD_TIMER.llm = getattr(THREAD_TIMER, "llm", 0.0) + (time.time() - t0)
                try:
                    if getattr(resp, "usage", None) is not None:
                        resp.usage = _sanitize_usage(resp.usage)
                except Exception: pass
                return resp
            return _timed
        c.chat.completions.create = make_timed()
        cap = c
        restorers.append(lambda c=cap, o=orig_create: setattr(c.chat.completions, "create", o))


    def wrap_method(obj, attr, counter):
        orig = getattr(obj, attr, None)
        if orig is None: return
        def _w(*a, **kw):
            t0 = time.time()
            try:
                return orig(*a, **kw)
            finally:
                dt = time.time() - t0
                setattr(THREAD_TIMER, counter, getattr(THREAD_TIMER, counter, 0.0) + dt)
                THREAD_TIMER.tool = getattr(THREAD_TIMER, "tool", 0.0) + dt
        try:
            setattr(obj, attr, _w)
            restorers.append(lambda o=obj, a=attr, x=orig: setattr(o, a, x))
        except Exception: pass

    for name, counter in [("global_bm25_searcher", "bm25"),
                          ("global_reranker", "rerank"),
                          ("global_vector_index", "faiss")]:
        obj = getattr(rise_mod, name, None)
        if obj is not None:
            attr = "predict" if "reranker" in name else "search"
            wrap_method(obj, attr, counter)

    try:
        yield
    finally:
        for r in restorers:
            try: r()
            except Exception: pass


def build_args(strategy: str, log_dir: str, max_steps: int, user_mode: str):
    return argparse.Namespace(
        num_trials=1, env="pwab",
        model="gpt-4o-mini",
        user_mode=user_mode, user_model="gpt-4o-mini",
        agent_strategy=strategy,
        temperature=0.0, task_split="test",
        agent_memory="taskspe_rise" if strategy == "rise" else "taskspe",
        memory_length=100, max_steps=max_steps,
        start_index=0, end_index=-1,
        resume_from=None, verbose=False,
        log_dir=log_dir, num_gpus=None, max_concurrency=1,
        seed=2024, shuffle=0,
        interec_memory_file=None, tts_n=10,
        sapa_param_file=None,
        sapa_function_file="SAPA/output/res/function_test_res.json",
        sapa_generate=0,
        sapa_model_path="finetune/output/input/Llama-2-7b-chat-hf/",
        mem_token_length=768,
    )


def run_strategy(strategy, task_ids, out_path, max_steps, user_mode):
    from PersonalWAB.envs import get_env
    args = build_args(strategy, log_dir="benchmark_logs", max_steps=max_steps, user_mode=user_mode)
    env = get_env(args.env, user_mode=args.user_mode, user_model=args.user_model,
                  task_split=args.task_split, max_steps=args.max_steps)

    if strategy == "function_calling":
        from PersonalWAB.agents.gpt_function_calling_agent import (
            GPTFunctionCallingAgent, initialize_client)
        initialize_client(api_key=os.getenv("OPENAI_API_KEY"),
                          base_url="https://xiaoai.plus/v1")
        tools_info = [t for t in env.functions_info
                      if t["function"]["name"] != "get_product_details_by_asin"]
        agent = GPTFunctionCallingAgent(
            tools_info, env.sys_prompt, model=args.model,
            function_selection_file=args.sapa_function_file)
    elif strategy == "rise":
        from PersonalWAB.agents.rise_agent import RISEAgent
        from PersonalWAB.agents.gpt_function_calling_agent import initialize_client
        initialize_client(api_key=os.getenv("OPENAI_API_KEY"),
                          base_url="https://xiaoai.plus/v1")
        agent = RISEAgent(
            tools_info=env.functions_info, sys_prompt=env.sys_prompt,
            model=args.model,
            function_selection_file=args.sapa_function_file,
            memory_file=None, tts_n=args.tts_n)
    else:
        raise ValueError(strategy)

    results = []
    with patch_openai_and_env(env):
        for i, idx in enumerate(task_ids):
            _reset_timer()
            t_total = time.time()
            task_type = env.tasks[idx].get("type", "unknown")
            try:
                action_acc, res_acc, info = agent.act(
                    env, idx, verbose=False, temperature=args.temperature,
                    max_steps=env.max_steps, memory=args.agent_memory)
                total_dur = time.time() - t_total
                tm = _snap_timer()
                usage = info.get("usage", {})
                rec = {
                    "task_id": idx, "task_type": task_type, "strategy": strategy,
                    "total_latency": total_dur,
                    "llm_latency": tm["llm"],
                    "tool_latency": tm["tool"],
                    "env_step_latency": tm["env_step"],
                    "bm25_latency": tm["bm25"],
                    "rerank_latency": tm["rerank"],
                    "faiss_latency": tm["faiss"],
                    "other_latency": max(total_dur - tm["llm"] - tm["tool"], 0.0),
                    "action_acc": action_acc, "res_acc": res_acc,
                    "prompt_tokens": sum(usage.get("prompt_tokens", [0])),
                    "completion_tokens": sum(usage.get("completion_tokens", [0])),
                    "total_tokens": sum(usage.get("total_tokens", [0])),
                    "total_price": usage.get("total_price", 0),
                }
                print(f"[{strategy:>18s}] {i+1:3d}/{len(task_ids)} "
                      f"id={idx:4d} type={task_type:<9s} "
                      f"total={total_dur:6.2f}s llm={tm['llm']:6.2f}s "
                      f"tool={tm['tool']:5.2f}s (bm25={tm['bm25']:.2f} "
                      f"rerank={tm['rerank']:.2f} faiss={tm['faiss']:.2f} "
                      f"env={tm['env_step']:.2f}) "
                      f"tok={rec['total_tokens']:5d} "
                      f"acc={action_acc}/{res_acc}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = {"task_id": idx, "task_type": task_type,
                       "strategy": strategy, "error": str(e)}
            results.append(rec)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    return results


def _safe_arr(records, key):
    return np.array([r.get(key, 0) or 0 for r in records], dtype=float)

def _first_or_zero(x):
    return (x[0] if isinstance(x, list) and x else (x or 0))

def summarize_by_type(results):
    valid = [r for r in results if "error" not in r]
    if not valid: return {}
    by_type = defaultdict(list)
    for r in valid:
        by_type[r["task_type"]].append(r)
    by_type["ALL"] = valid
    out = {}
    for ttype, recs in by_type.items():
        n = len(recs)
        if n == 0: continue
        summary = {"n": n}
        for k in ["total_latency", "llm_latency", "tool_latency",
                  "env_step_latency", "bm25_latency", "rerank_latency",
                  "faiss_latency", "other_latency",
                  "prompt_tokens", "completion_tokens", "total_tokens"]:
            a = _safe_arr(recs, k)
            summary[f"mean_{k}"]   = float(a.mean())
            summary[f"median_{k}"] = float(np.median(a))
        summary["total_cost"] = float(_safe_arr(recs, "total_price").sum())
        summary["action_acc"] = float(np.mean([_first_or_zero(r.get("action_acc")) for r in recs]))
        summary["res_acc"]    = float(np.mean([_first_or_zero(r.get("res_acc"))    for r in recs]))
        out[ttype] = summary
    return out


def print_comparison_table(summaries, strategies):
    print("\n" + "=" * 120)
    header = f"{'metric':<26s}{'type':<10s}" + "".join(f"{s:>22s}" for s in strategies)
    print(header); print("-" * len(header))
    types = ["ALL", "search", "recommend", "review"]
    keys = [
        ("n", "int"),
        ("mean_total_latency", "s"),
        ("mean_llm_latency", "s"),
        ("mean_tool_latency", "s"),
        ("mean_bm25_latency", "s"),
        ("mean_rerank_latency", "s"),
        ("mean_faiss_latency", "s"),
        ("mean_env_step_latency", "s"),
        ("median_total_latency", "s"),
        ("mean_prompt_tokens", "f"),
        ("mean_completion_tokens", "f"),
        ("mean_total_tokens", "f"),
        ("total_cost", "f4"),
        ("action_acc", "f4"),
        ("res_acc", "f4"),
    ]
    for key, kind in keys:
        for ttype in types:
            row = f"{key:<26s}{ttype:<10s}"
            for s in strategies:
                v = summaries.get(s, {}).get(ttype, {}).get(key)
                if v is None: row += f"{'-':>22s}"
                elif kind == "int": row += f"{int(v):>22d}"
                elif kind == "s":   row += f"{v:>19.3f}s  "
                elif kind == "f4":  row += f"{v:>22.4f}"
                else:               row += f"{v:>22.2f}"
            print(row)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--strategies", nargs="+", default=["function_calling", "rise"])
    ap.add_argument("--out_dir", default="benchmark_logs")
    ap.add_argument("--seed", type=int, default=2024,
                    help="seed for random task sampling (same for both strategies)")
    ap.add_argument("--random", type=int, default=1,
                    help="1: random sample; 0: take first N tasks")
    ap.add_argument("--balanced", type=int, default=1,
                    help="1: stratified sample by task_type (n_tasks split evenly); "
                         "0: plain random")
    ap.add_argument("--max_steps", type=int, default=1,
                    help="1: single-step (user_mode=naive, enables RISE candidate fill); "
                         "-1: no user sim (user_mode=no); >1: multi-turn")
    ap.add_argument("--user_mode", default="naive",
                    choices=["naive", "no"])
    args = ap.parse_args()


    from PersonalWAB.envs import get_env
    env_tmp = get_env("pwab", user_mode="no", user_model="gpt-4o-mini",
                      task_split="test", max_steps=-1)
    all_ids = list(range(len(env_tmp.tasks)))

    rng = random.Random(args.seed)
    if args.random:
        if args.balanced:
            by_t = defaultdict(list)
            for i in all_ids:
                by_t[env_tmp.tasks[i].get("type")].append(i)
            types = sorted(by_t.keys())
            per = args.n_tasks // len(types)
            rem = args.n_tasks - per * len(types)
            task_ids = []
            for t in types:
                k = per + (1 if rem > 0 else 0)
                if rem > 0: rem -= 1
                task_ids.extend(rng.sample(by_t[t], min(k, len(by_t[t]))))
            rng.shuffle(task_ids)
        else:
            task_ids = rng.sample(all_ids, min(args.n_tasks, len(all_ids)))
    else:
        task_ids = all_ids[:args.n_tasks]
    del env_tmp

    print(f"selected {len(task_ids)} task ids, seed={args.seed}, "
          f"random={args.random}, balanced={args.balanced}")
    from collections import Counter
    from PersonalWAB.envs import get_env as _get_env
    _e = _get_env("pwab", user_mode="no", user_model="gpt-4o-mini",
                  task_split="test", max_steps=-1)
    type_ct = Counter(_e.tasks[i].get("type") for i in task_ids)
    print("type distribution:", dict(type_ct))
    del _e

    ts = datetime.now().strftime("%m%d%H%M")
    summaries = {}
    for s in args.strategies:
        out_path = os.path.join(args.out_dir, f"{s}_n{args.n_tasks}_ms{args.max_steps}_{ts}.json")
        print(f"\n=== running {s} (max_steps={args.max_steps}, user_mode={args.user_mode}, "
              f"{len(task_ids)} tasks) -> {out_path} ===", flush=True)
        res = run_strategy(s, task_ids, out_path, args.max_steps, args.user_mode)
        summaries[s] = summarize_by_type(res)

    print_comparison_table(summaries, args.strategies)

    meta = {"n_tasks": args.n_tasks, "seed": args.seed,
            "task_ids": task_ids, "summaries": summaries}
    sum_path = os.path.join(args.out_dir, f"summary_{ts}.json")
    with open(sum_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsummary -> {sum_path}")


if __name__ == "__main__":
    main()
