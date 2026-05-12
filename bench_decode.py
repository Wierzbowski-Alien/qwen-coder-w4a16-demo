#!/usr/bin/env python3
"""Qwen 2.5 Coder W4A16 — Decode Benchmark

Measures pure decode throughput on RTX 3060 12 GB.
Same methodology as README: 200 steps, GPU auto-boost, Best5 mean.
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import deepseek_r1_w4_model as m

N_WARMUP = 10
N_STEPS = 200
N_RUNS = 5
PROMPT = "The capital of France is"
WEIGHTS = "qwen2.5_coder_7b_w4a16_bi.pt"
TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("=" * 60)
print(" Qwen 2.5 Coder W4A16 — Decode Benchmark")
print("=" * 60)
print(f" GPU: {torch.cuda.get_device_name(0)}")
print(f" Steps: {N_STEPS}, Runs: {N_RUNS}")
print(f" Weights: {WEIGHTS}")
print()

# Load model
print("Loading model...")
model = m.Decoder(weights_file=WEIGHTS, tokenizer_name=TOKENIZER, verbose=True)
print()

# Prefill
ids = model.tokenizer.encode(PROMPT, add_special_tokens=False)
model.prefill(ids)
last_tok = ids[-1]
pos = model._position

# Warmup (GPU auto-boost stabilizes during warmup)
print(f"Warmup ({N_WARMUP} steps)...")
for _ in range(N_WARMUP):
    model.step(last_tok)
    model._position = pos
torch.cuda.synchronize()
print("Warmup done.\n")

# Benchmark runs
results = []
for run in range(1, N_RUNS + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        model.step(last_tok)
        model._position = pos
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = N_STEPS / elapsed
    ms_per_step = elapsed / N_STEPS * 1000
    results.append((tok_s, ms_per_step, elapsed))
    print(f"  Run {run}: {elapsed:.3f}s = {tok_s:.1f} tok/s ({ms_per_step:.2f} ms/token)")

# Best5 mean (same methodology as README)
avg_tok_s = sum(r[0] for r in results) / len(results)
avg_ms = sum(r[1] for r in results) / len(results)
best_tok_s = max(r[0] for r in results)

print()
print("=" * 60)
print(f" Mean ({N_RUNS} runs):  {avg_tok_s:.1f} tok/s  ({avg_ms:.2f} ms/token)")
print(f" Best of {N_RUNS}:        {best_tok_s:.1f} tok/s")
print("=" * 60)

# VRAM
vram = torch.cuda.memory_allocated() / 1e9
print(f" VRAM: {vram:.1f} GB")
print(f" GPU clock: check with: nvidia-smi dmon -s pucm -c 1")
print()

# Cleanup
del model; gc.collect(); torch.cuda.empty_cache()
print("Benchmark complete.")
