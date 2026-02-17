"""
Benchmark: BatchedDynamicBudgetTrimKVCache vs PagedDynamicBudgetTrimKVCache

Measures wall-clock generation time, tokens/sec, and peak GPU memory across
varying batch sizes, memory budgets, and generation lengths.

Usage:
    # Quick test (1 config, 1 trial)
    python examples/bench_batched_vs_paged.py --quick

    # Full sweep (default)
    python examples/bench_batched_vs_paged.py

    # Custom sweep
    python examples/bench_batched_vs_paged.py \
        --model ngocbh/TrimKV-Qwen3-4B-Math \
        --batch-sizes 1 2 4 \
        --memory-sizes 256 512 1024 \
        --max-new-tokens 512 2048 8192 \
        --trials 3 \
        --warmup 1 \
        --page-size 256 \
        --buffer-size 32 \
        --output results.json
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product

import torch
from transformers import AutoTokenizer

from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import (
    BatchedDynamicBudgetTrimKVCache,
    PagedDynamicBudgetTrimKVCache,
)


# ---------------------------------------------------------------------------
# Benchmark prompts – deliberately varied in length so the tokeniser produces
# unequal input sequences (exercises padding / variable-length handling).
# ---------------------------------------------------------------------------
PROMPTS = [
    "A train travels 120 kilometers in 3 hours at a constant speed. "
    "How many kilometers will it travel in 5 hours at the same speed?",

    "What is 2+2?",

    "Solve the following: if f(x) = 3x^2 - 2x + 1, find f(5).",

    "A box contains 5 red balls and 3 blue balls. If two balls are drawn "
    "without replacement, what is the probability that both are red?",
]


@dataclass
class TrialResult:
    implementation: str       # "batched" | "paged"
    batch_size: int
    memory_size: int
    max_new_tokens: int
    trial: int
    prefill_tokens: int       # total input tokens (sum across batch)
    generated_tokens: int     # total output tokens (sum across batch)
    wall_time_s: float        # end-to-end generation time
    tokens_per_sec: float     # generated_tokens / wall_time_s
    peak_memory_mb: float     # peak GPU memory during generation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_inputs(tokenizer, batch_size, device):
    """Prepare a batch of chat-formatted inputs."""
    prompts = PROMPTS[:batch_size]
    # Cycle prompts if batch_size > len(PROMPTS)
    while len(prompts) < batch_size:
        prompts = prompts + PROMPTS[:batch_size - len(prompts)]

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for m in messages_list
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    return model_inputs


def reset_gpu():
    """Aggressively free GPU memory between runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_generation(model, tokenizer, model_inputs, cache, max_new_tokens):
    """Run model.generate and return (output_ids, wall_time_s, peak_mem_mb)."""
    torch.cuda.synchronize()
    reset_gpu()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        past_key_values=cache,
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    wall_time = t1 - t0
    return output_ids, wall_time, peak_mem


def count_generated_tokens(model_inputs, output_ids):
    """Total new tokens across the batch."""
    total = 0
    for inp, out in zip(model_inputs.input_ids, output_ids):
        total += len(out) - len(inp)
    return total


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def bench_one(
    model,
    tokenizer,
    impl: str,
    batch_size: int,
    memory_size: int,
    max_new_tokens: int,
    buffer_size: int,
    page_size: int,
    trial: int,
) -> TrialResult:
    """Run a single benchmark trial for one (impl, config) combination."""

    # -- configure model for this implementation --
    if impl == "batched":
        model.config._attn_implementation = "db_attn_flash_batched"
        cache = BatchedDynamicBudgetTrimKVCache(device="cuda")
    elif impl == "paged":
        model.config._attn_implementation = "db_attn_flash_paged"
        cache = PagedDynamicBudgetTrimKVCache(page_size=page_size, device="cuda")
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    model.config.compress_memory = True
    model.config.memory_size = memory_size
    model.config.buffer_size = buffer_size

    model_inputs = build_inputs(tokenizer, batch_size, model.device)
    prefill_tokens = model_inputs.input_ids.numel()

    output_ids, wall_time, peak_mem = run_generation(
        model, tokenizer, model_inputs, cache, max_new_tokens,
    )
    gen_tokens = count_generated_tokens(model_inputs, output_ids)

    # clean up
    del cache, output_ids, model_inputs
    reset_gpu()

    return TrialResult(
        implementation=impl,
        batch_size=batch_size,
        memory_size=memory_size,
        max_new_tokens=max_new_tokens,
        trial=trial,
        prefill_tokens=prefill_tokens,
        generated_tokens=gen_tokens,
        wall_time_s=round(wall_time, 3),
        tokens_per_sec=round(gen_tokens / wall_time, 2) if wall_time > 0 else 0,
        peak_memory_mb=round(peak_mem, 1),
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def print_header():
    cols = (
        f"{'impl':<10} {'BS':>3} {'mem':>5} {'max_new':>8} "
        f"{'trial':>5} {'prefill':>8} {'gen_tok':>8} "
        f"{'time_s':>8} {'tok/s':>9} {'peak_MB':>9}"
    )
    print(cols)
    print("-" * len(cols))


def print_row(r: TrialResult):
    print(
        f"{r.implementation:<10} {r.batch_size:>3} {r.memory_size:>5} "
        f"{r.max_new_tokens:>8} {r.trial:>5} {r.prefill_tokens:>8} "
        f"{r.generated_tokens:>8} {r.wall_time_s:>8.3f} "
        f"{r.tokens_per_sec:>9.2f} {r.peak_memory_mb:>9.1f}"
    )


def print_summary(results):
    """Print a summary table: mean across trials, side-by-side batched vs paged."""
    from collections import defaultdict

    # group by (batch_size, memory_size, max_new_tokens, impl)
    groups = defaultdict(list)
    for r in results:
        key = (r.batch_size, r.memory_size, r.max_new_tokens, r.implementation)
        groups[key].append(r)

    # collect config keys
    config_keys = sorted(set(
        (r.batch_size, r.memory_size, r.max_new_tokens) for r in results
    ))

    print("\n" + "=" * 100)
    print("SUMMARY (mean over trials)")
    print("=" * 100)
    header = (
        f"{'BS':>3} {'mem':>5} {'max_new':>8} | "
        f"{'batched_tok/s':>14} {'batched_time':>13} {'batched_MB':>11} | "
        f"{'paged_tok/s':>14} {'paged_time':>13} {'paged_MB':>11} | "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for bs, mem, mnt in config_keys:
        batched_runs = groups.get((bs, mem, mnt, "batched"), [])
        paged_runs = groups.get((bs, mem, mnt, "paged"), [])

        def mean(lst, attr):
            vals = [getattr(r, attr) for r in lst]
            return sum(vals) / len(vals) if vals else float("nan")

        b_tps = mean(batched_runs, "tokens_per_sec")
        b_t = mean(batched_runs, "wall_time_s")
        b_m = mean(batched_runs, "peak_memory_mb")
        p_tps = mean(paged_runs, "tokens_per_sec")
        p_t = mean(paged_runs, "wall_time_s")
        p_m = mean(paged_runs, "peak_memory_mb")

        # speedup: paged tok/s  / batched tok/s  (>1 means paged is faster)
        speedup = p_tps / b_tps if b_tps > 0 else float("nan")

        print(
            f"{bs:>3} {mem:>5} {mnt:>8} | "
            f"{b_tps:>14.2f} {b_t:>12.3f}s {b_m:>10.1f}M | "
            f"{p_tps:>14.2f} {p_t:>12.3f}s {p_m:>10.1f}M | "
            f"{speedup:>7.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batched vs paged TrimKV cache"
    )
    parser.add_argument(
        "--model", default="ngocbh/TrimKV-Qwen3-4B-Math",
        help="Model path or HF id",
    )
    parser.add_argument(
        "--download-from", default="huggingface",
        choices=["huggingface", "wandb", "local"],
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
    )
    parser.add_argument(
        "--memory-sizes", type=int, nargs="+", default=[256, 512, 1024],
    )
    parser.add_argument(
        "--max-new-tokens", type=int, nargs="+", default=[512, 2048, 8192],
    )
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 1 config, 1 trial, no warmup",
    )
    parser.add_argument(
        "--impls", nargs="+", default=["batched", "paged"],
        choices=["batched", "paged"],
        help="Which implementations to benchmark",
    )
    args = parser.parse_args()

    if args.quick:
        args.batch_sizes = [2]
        args.memory_sizes = [256]
        args.max_new_tokens = [512]
        args.trials = 1
        args.warmup = 0

    # -- load model & tokenizer once --
    print(f"Loading model: {args.model}")
    model = TrimKVQwen3ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        load_trimkv_weights=True,
        download_from=args.download_from,
        use_cache=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_model, use_fast=True, padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded on {model.device}\n")

    # -- build sweep --
    configs = list(product(
        args.batch_sizes, args.memory_sizes, args.max_new_tokens,
    ))
    total_runs = (
        len(args.impls) * len(configs) * (args.warmup + args.trials)
    )
    print(
        f"Sweep: {len(args.impls)} impls x {len(configs)} configs "
        f"x ({args.warmup}W + {args.trials}T) = {total_runs} runs\n"
    )

    # -- warmup --
    if args.warmup > 0:
        print("--- Warmup ---")
        for impl in args.impls:
            for _ in range(args.warmup):
                bs, mem, mnt = configs[0]
                bench_one(
                    model, tokenizer, impl, bs, mem,
                    min(mnt, 64),  # short generation for warmup
                    args.buffer_size, args.page_size, trial=0,
                )
            print(f"  {impl}: {args.warmup} warmup run(s) done")
        print()

    # -- benchmark --
    results = []
    print_header()

    for bs, mem, mnt in configs:
        for impl in args.impls:
            for t in range(1, args.trials + 1):
                r = bench_one(
                    model, tokenizer, impl, bs, mem, mnt,
                    args.buffer_size, args.page_size, trial=t,
                )
                results.append(r)
                print_row(r)
                sys.stdout.flush()

    # -- summary --
    print_summary(results)

    # -- save --
    if args.output:
        data = {
            "args": vars(args),
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # -- GPU info --
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")


if __name__ == "__main__":
    main()
