# NFL Football Fine-Tuning — Full Report

A LoRA fine-tuning and multi-adapter serving study built on the from-scratch
[`llm_inference_engine`](https://github.com/ashwinvijayakumar24/llm_inference_engine) and
[`llm_serving_layer`](https://github.com/ashwinvijayakumar24/llm_serving_layer). Llama 3.2 1B
Instruct is taught to read raw NFL play descriptions (`"(3:12) 1-J.Hurts pass short right to
11-A.Brown to DAL 22 for 14 yards"`) and emit the fantasy-relevant stat credits as JSON. Then
the question an applied-AI team actually faces gets answered with numbers instead of
intuition: **regex, prompting, a frontier API model, LoRA, QLoRA, a full fine-tune, or
distillation — which one, and what does each cost?** The adapters are then served on the
owner's own engine and serving layer, many at once, and compared against vLLM.

Trained and benchmarked on NVIDIA A100 80GB, L40S, and H100 80GB (Georgia Tech PACE Phoenix),
with the frontier and teacher rungs on the OpenAI API. Ground truth is exact (nflverse's
official structured stats), so **every quality number is an exact match against official
data, with no LLM judge anywhere**. Every number below resolves to a committed artifact in
[`results/`](results/), and every claim's pass condition was written down before its run.

**Fifteen claims were scoped. Ten are earned, four are not, and one could not be measured —
and each says so.** The most useful result is a loss: on clean, fixed-format official text, a
careful one-day regex (99.79%) still edges out the best fine-tuned model (99.73%). The most
striking one is a win: the fine-tuned 1B model **beats gpt-5.5** on the same plays at
**~1/170th the cost per play**.

| | claim | result |
|---|---|---|
| **T1** | fine-tuning beats prompting at 1B | **0.0% → 99.54%** on the frozen 37,859-play test set |
| **T2** | a 1B adapter matches a frontier model | **beats gpt-5.5**: 98.53% vs 97.96%, paired **+0.57** [+0.17, +1.00], at ~$0.04 vs $6.40 per 1k plays |
| **T3** | LoRA beats the regex on hard plays | **NOT EARNED**: 99.54% vs 99.79%, paired −0.25 [−0.32, −0.19] |
| **T3b** | spot decomposition beats the regex *(pre-registered follow-up)* | **NOT EARNED**: 99.73% vs 99.79% / 99.90% — but **+0.18** over plain LoRA, and penalty plays 93.5% → **100%** |
| **T4** | QLoRA costs little quality | **−0.4 points** with a 4-bit base (limit 1.0); ~2% slower steps |
| **T5** | LoRA matches full fine-tuning | **LoRA beat it**: 98.53% vs 97.35%, paired +1.18 [+0.43, +2.12] |
| **T6** | rejection filtering recovers distillation quality | **NOT EARNED**: closes at most **31%** of the teacher-vs-truth gap; the filter drops exactly the hard plays |
| **T7** | fine-tuning causes little forgetting | **no MMLU drop**: 31.1% → 32.3% (limit −2.0) |
| **L1** | a merged adapter adds no latency | **−0.07%** decode vs base on an H100, inside base-vs-base noise |
| **L2** | merged == unmerged | **identical greedy tokens** on every path, laptop and H100 |
| **L3** | unmerged overhead is small at low rank | **batch 1: NOT EARNED** (~30%, launch-bound, flat in rank) · **batch 32: earned** for v1 (+11.3%) |
| **L4** | multi-LoRA scales with the number of adapters | **NOT MEASURABLE** on its goodput metric; saturation throughput holds **96–98%** from 1 → 256 adapters |
| **L5** | adapters beat one merged model per tenant | **~3,300 resident r=16 adapters vs 31 merged copies** per 80 GB, measured |
| **L6** | the prefix cache is adapter-safe | **0 cross-adapter hits**, with a fault-injection test that catches the bug |
| **L7** | our serving is competitive with vLLM | **NOT EARNED**: 9–13% of vLLM 0.31's throughput (≈60 vs 2.3 ms per token) |

---

## Why this exists

Two projects already sit under this one. The engine runs Llama 3.2 1B correctly, one request
at a time. The serving layer serves it: paged KV cache, continuous batching, preemption, radix
prefix caching. Neither can **specialize** the model, and neither can serve more than one
specialization of it.

That gap is a live production problem. Applied-AI teams increasingly ship one fine-tuned
adapter per customer or per task, and the naive way to serve 100 of them is 100 copies of the
model. The real way is one base model plus 100 small adapters batched together in the same
forward pass. Getting there means answering two questions honestly:

1. **When is fine-tuning even worth it?** Most teams answer this by intuition. The decision
   ladder here answers it for one concrete task, with quality, cost, and latency for every rung,
   including the rungs where fine-tuning loses.
2. **What does it take to serve adapters well?** Merged vs unmerged, batching requests for
   different adapters together, keeping the prefix cache correct, and how far a from-scratch
   stack sits from vLLM.

The task is chosen for one property above all: **exact ground truth for free.** nflverse ships
every NFL play's official text in the same row as its official structured stats, so every
answer is right or wrong with no judgment call. That made it possible to measure things that
are usually only estimated — including exactly what distillation costs, because the teacher's
labels can be scored against the truth.

---

## The task

**Input** (the minimal prompt the final models use; the line of scrimmage is added for T3b):

```
posteam: PHI
desc: (3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards
(21-T.Diggs). FUMBLES (21-T.Diggs), RECOVERED by DAL-55-D.Lawrence at DAL 20.
```

**Output** (fixed schema, ten stat types, names exactly as the text writes them):

```json
{"nullified": false,
 "credits": [{"player": "J.Hurts", "stat": "pass_yds", "value": 14},
             {"player": "A.Brown", "stat": "rec", "value": 1},
             {"player": "A.Brown", "stat": "rec_yds", "value": 14},
             {"player": "A.Brown", "stat": "fumble_lost", "value": 1}]}
```

A deterministic scorer turns credits into points under any scoring system (standard, half-PPR,
PPR, the owner's 14-team ESPN league). The model only reads; arithmetic on points stays in code.

**Data.** nflverse play-by-play, 2015–2024, scoped to pass, run, and no-play rows plus
two-point tries. Splits are by season so the model always parses a season it has never seen:

| split | seasons | plays | role |
|---|---|---|---|
| train | 2015–2022 | 294,016 | training, sweeps (50k fixed subset) |
| val | 2023 | 38,102 | early stopping, knob selection, all development |
| test | 2024 | 37,859 | **frozen** (sha256 `13d0d714…`), scored once per final system |
| eval_lite | 2024 | 2,795 | 18 whole games + a top-up so every bucket has ≥100 plays (lateral: all 29) |

**Validation of the ground truth itself.** Per-play labels were summed into per-player game
totals and compared with nflverse's official weekly stats: **99.79% of 50,602 player-games agree
on every stat, PPR fantasy-point MAE 0.0035**, and every remaining disagreement is explained
(multi-lateral plays, a fantasy-formula quirk, one fake punt). A 200-play stratified audit found
errors only in the lateral bucket. See [`docs/benchmarks/p1-crosscheck.md`](docs/benchmarks/p1-crosscheck.md).

**Buckets.** Every test play gets one tag (`normal`, `td`, `penalty_nullified`, `penalty_stands`,
`fumble`, `interception`, `challenge`, `two_point`, `lateral`) and **every metric is reported per
bucket**. 83% of plays are `normal`, and an average over them hides everything interesting.

---

## Results

> **Read the bucket columns, not just the overall number.** Every system here is perfect or
> near-perfect on the ~96% of plays that are routine. The differences live in four rare
> buckets — fumbles, laterals, challenges, and penalties where the play stands — where
> official yardage follows rules the text does not spell out.

### The decision ladder

Every rung on the same plays. Paired differences are game-clustered bootstrap CIs (plays in
one game are correlated, so games, not plays, are resampled).

**Frozen test set — all 37,859 plays from 2024** (scored once per final system):

| system | exact match [95% CI] | fumble | pen. stands | lateral | challenge | pen. nullified |
|---|---|---|---|---|---|---|
| R0: regex, 1-day budget | **99.79** [99.75, 99.84] | 92.5 | 100.0 | 62.1 | 97.9 | 99.5 |
| R0 + line of scrimmage (fairness arm) | **99.90** | 98.1 | 100.0 | 82.8 | 99.4 | 99.5 |
| R5: LoRA, all 294k plays | 99.54 [99.47, 99.61] | 83.2 | 93.5 | 44.8 | 97.0 | 99.9 |
| T3b: LoRA with spot outputs | 99.73 | 86.0 | **100.0** | 20.7 | 97.9 | **100.0** |

**eval_lite — 2,795 plays** (every rung, including the expensive ones):

| rung | system | exact match [95% CI] | cost per 1k plays |
|---|---|---|---|
| R0 | regex | 99.14 [98.6, 99.5] | ≈ $0 (CPU, ~74k plays/s) |
| T3b | LoRA, spot outputs | 98.68 [97.8, 99.2] | ≈ $0.04 |
| **R5** | **LoRA r=16, all 294k plays** | **98.53** [97.7, 99.0] | **≈ $0.04** (A100, 15 plays/s, unoptimized HF generate) |
| R6 | QLoRA (4-bit NF4 base) | 98.1 | ≈ $0.04 |
| R4 | **gpt-5.5**, 8-shot, reasoning low | 97.96 [97.0, 98.6] | **$6.40** (measured) |
| R7 | full fine-tune (fp32 masters) | 97.35 | ≈ $0.04 |
| R9 | LoRA on 100k filtered teacher labels | 96.78 | $1.40 to label + ≈ $0.04 |
| R8 | LoRA on 100k raw teacher labels | 96.46 | $1.40 to label + ≈ $0.04 |
| R4 | gpt-5.4-mini, 8-shot | 95.24 [93.8, 96.1] | $1.00 (measured) |
| R2 | Llama 1B, 8 fixed examples | 16.7 *(108-play slice)* | — |
| R1 | Llama 1B, zero-shot | **0.0** (24.7% valid JSON) | — |

The ladder's answer for this task: **prompting a 1B model does not work at all; fine-tuning it
beats a frontier model on quality and cost; and a hand-written regex is still the best tool for
clean, fixed-format official text.** The fine-tuned model's advantage over the regex is the one
not measured here — it needed no rules, which matters when the format is not fixed.

*Artifacts: [`results/test/`](results/test/), [`results/t3b/test_comparison.json`](results/t3b/test_comparison.json), [`results/eval_lite/`](results/eval_lite/) · R5 job `13906064` (A100), test eval `13908197`, T3b `13909293` / `13918365`.*

### T3 — does LoRA beat the regex? Not earned.

R5 trained on all 294k plays (early-stopped at step 7,000, best at 5,000) and was scored once
on the frozen test set: **99.54% vs the regex's 99.79%, paired −0.25 [−0.32, −0.19]**. Of 37,859
plays, R5 alone got 30 right and the regex alone got 126.

Where it loses is the finding. On fumbles, penalties where the play stands, and laterals, the
**official** yards are measured to a spot on the field — the fumble point, the foul spot — not
the "for N yards" the text states. Official yards disagree with the text's stated gain on 32–42%
of fumble plays. The regex was written with those rules in mind; the model has to infer them
from examples, and with 294k plays it learns most but not all.

### T3b — let code do the arithmetic. Pre-registered, not earned, and still informative.

Written into [`PRD.md`](PRD.md) §17 and committed **before** any code existed: if the model's
remaining errors are arithmetic, then having it output the **field spots** that bound each
yardage credit — and letting plain code subtract — should beat the regex. Because the new model
sees the line of scrimmage, the regex got the same field with a fixed two-hour budget (it used
20 minutes and jumped to 99.90%).

| paired difference on the frozen test set | estimate [95% CI] |
|---|---|
| T3b − R0 | −0.07 [−0.12, −0.02] |
| T3b − R0 + line of scrimmage | −0.18 [−0.23, −0.12] |
| **T3b − R5 (plain LoRA)** | **+0.18 [+0.12, +0.25]** |

The pass condition required beating both regex arms, so **T3b is not earned**. But the
decomposition did exactly what it was designed to do where the error *was* arithmetic: penalty
plays where the play stands went **93.5% → 100%**. What remains on fumbles is a reading
problem (choosing *which* printed spot is official), and laterals got worse because
multi-lateral plays need a spot nflverse never credits. A round-trip check confirmed the spot
format loses nothing: converting the spot labels back reproduces the original ground truth on
**100.0000%** of all 370k plays. See [`docs/benchmarks/t3b-test.md`](docs/benchmarks/t3b-test.md).

### What LoRA's knobs actually do (25 runs)

Every run is one knob changed from a fixed base (50k plays, r=16, α=32, all-linear projections,
dropout 0.05, lr 2e-4, minimal prompt, 2 epochs, early stop), scored on eval_lite.

| knob | values | eval_lite exact match |
|---|---|---|
| rank (α = 2r) | 2 · 4 · 8 · 16 · 64 | 97.5 · 97.6 · 97.6 · 97.7 · 97.7 |
| alpha | α=16 at r = 2 · 4 · 8 · 16 · 64 | 97.5 · 97.7 · 97.8 · 97.8 · 97.6 |
| target modules | q,v · q,k,v,o · q,v at matched params · q,k,v,o at matched params | 97.7 · 97.7 · 97.7 · 97.9 |
| learning rate | 5e-5 · 1e-4 · 2e-4 · 5e-4 · 1e-3 | 97.5 · 97.5 · 97.7 · 97.8 · 97.6 |
| dropout | 0 · 0.05 · 0.1 | 97.7 · 97.7 · 97.7 |
| prompt | minimal · full system prompt | 97.7 · 97.9 |
| **training plays** | **1k · 5k · 20k · 50k · 100k · 294k** | **84.6 · 96.3 · 97.7 · 97.7 · 98.1 · 98.5** |

**No LoRA knob matters on this task.** All 23 knob variants land within 0.4 points of each other,
inside each run's ±0.8-point CI. **Rank 2 (≈1.4M trainable parameters) scores the same as rank 64
(≈45M).** That is LoRA's core premise — task adaptation lives in a low-dimensional subspace —
measured on this task rather than quoted. **Data is the lever**, and the gain from 50k to 294k
comes almost entirely from the rare buckets, which simply appear more often in a bigger sample.
See [`docs/benchmarks/p3-sweep.md`](docs/benchmarks/p3-sweep.md).

### Training cost: why LoRA is cheap

| | trainable params | training state (weights + grads + AdamW) |
|---|---|---|
| full fine-tune | 1,235,814,400 | **18.42 GiB** |
| LoRA r=16, all 7 projections | 11,272,192 (**0.91%**) | **2.47 GiB** (2.30 of it is the frozen bf16 base) |
| LoRA r=4, q,v only | 425,984 | 2.31 GiB |

Analytic counts match the real injected model exactly. On an A100, LoRA trained at **~11.7k
tokens/s (0.62 s per 64-example step)**; R5 took ~1.5 hours. The hand-written LoRA layer
matches Hugging Face PEFT's logits exactly (max difference 0.0 on a tiny model, <1e-5 on the real
1B) and saves adapters in PEFT's on-disk format, which vLLM later loaded unchanged.

**The adapter does not need its system prompt.** 89% of training tokens were prompt, and the
186-token system prompt was most of it. Trained with no system message, the adapter matched the
full-prompt adapter (93.2% vs 93.7% at pilot scale, paired −0.5 [−1.5, +0.4]; 97.7% vs 97.9% at
50k) with **60% fewer tokens per example and 2.8× faster training steps**. Every GPU run used
the minimal prompt. See [`docs/benchmarks/p3-prompt-ablation.md`](docs/benchmarks/p3-prompt-ablation.md).

### QLoRA, full fine-tuning, and forgetting (T4, T5, T7)

| question | result | verdict |
|---|---|---|
| Does a 4-bit base hurt? (R6, NF4 + double quantization, bf16 compute) | 98.1% vs 98.53%; 11.6k vs 11.9k tok/s | **T4 earned** (−0.4, limit 1.0) |
| Does LoRA leave quality on the table? (R7, all 1.24B weights, fp32 masters, lr 2e-5) | 97.35% vs 98.53%, paired LoRA **+1.18** [+0.43, +2.12] | **T5: LoRA beat it** |
| Does fine-tuning erase general knowledge? (1,000 seeded MMLU questions, 0-shot) | base 31.1% [28.4, 34.0] → R5 32.3%, paired +1.2 [−1.8, +4.4] | **T7 earned** (limit −2.0) |

The full fine-tune's learning rate was a standard default, not tuned, and each arm is one run —
so T5 says LoRA did not leave quality on the table here, not that it always wins. The forgetting
threshold was committed in [`docs/phases/T7.md`](docs/phases/T7.md) before the scorer was written.
One detail worth noting: R5 changes the base model's answer on 41% of MMLU questions while
leaving accuracy unchanged.

### Frontier models and distillation (T2, T6)

**T2 — the 1B adapter beats gpt-5.5.** Same prompt and 8 fixed examples as R2, reasoning effort
low, prices verified against OpenAI's pricing page on the day of the run.

| | exact match | R5 − system | fumble | pen. stands | lateral | cost per 1k plays |
|---|---|---|---|---|---|---|
| **R5 (1B + LoRA)** | **98.53** | — | 85 | 94 | 45 | ≈ $0.04 |
| gpt-5.5 | 97.96 | **+0.57** [+0.17, +1.00] | 80 | 82 | 62 | $6.40 |
| gpt-5.4-mini | 95.24 | +3.29 [+2.68, +4.06] | 72 | 79 | 21 | $1.00 |

gpt-5.5 is better on laterals, which reward multi-step reasoning; the adapter is better on the
official-yardage buckets, whose conventions it absorbed from 294k examples and an 8-shot prompt
cannot convey. See [`docs/benchmarks/t2-frontier.md`](docs/benchmarks/t2-frontier.md).

**T6 — distillation is capped by the teacher's hard-case errors.** gpt-5.4-mini labeled 100,000
train plays with 3 samples each ($139.64). Students were trained on **exactly the plays** the
ground-truth data-size runs used, so the three curves differ only in where labels came from.
R8 keeps sample 0 if it parses; R9 keeps a play only if all 3 samples agree and every yardage
value and name appears literally in the text. Ground truth was never used by the filter.

| training plays | true labels | R8 (raw teacher) | R9 (filtered) | gap closed by filtering |
|---|---|---|---|---|
| 1k | 84.58 | 83.40 | 83.51 | 9% |
| 5k | 96.28 | 93.27 | 94.20 | 31% |
| 20k | 97.67 | 96.85 | 96.82 | −5% |
| 100k | 98.07 | 96.46 | 96.78 | 20% |

**T6 is not earned** — filtering never closes most of the gap. The mechanism is the useful part:

| bucket | teacher label error | share the filter drops |
|---|---|---|
| lateral | 66.7% | 40.0% |
| fumble | 22.0% | 16.0% |
| two_point | 12.3% | 20.2% |
| penalty_stands | 12.5% | 8.9% |
| normal | 2.1% | 4.1% |

The filter lifts label precision from 97.5% to 99.2%, but the teacher's errors sit in exactly the
buckets that separate good models from great ones, and the agreement check removes those plays.
The filtered student gets cleaner labels and fewer of the hard examples it most needs. A student
trained on 100k teacher labels (96.5–96.8%) lands below a student trained on 20k true labels
(97.7%): **a teacher's systematic errors cap the student, and more teacher data does not fix
them.** See [`docs/benchmarks/t6-distillation.md`](docs/benchmarks/t6-distillation.md).

### Serving: correct, scalable, not fast

**Correctness chain.** Every serving layer is verified against the one below it before anything
is timed:

```
batched mixed-adapter output  ==  each request run alone      v1 and v2 kernels, real 1B, CUDA
unmerged adapter (engine)     ==  merged adapter (engine)      identical greedy tokens
engine + adapter (fp32)       ==  HF + PEFT                    max logit diff 2.2e-4
hand-written LoRALinear       ==  PEFT                         max logit diff 0.0 (tiny) / <1e-5 (1B)
```

The batch-equals-solo gate also holds with a pool smaller than the adapter set (requests wait,
adapters are evicted and reloaded), chunked prefill, staggered arrivals, and both preemption
modes. Reversing the recorded adapter slots makes 9 gate tests fail, so the gate can actually
catch misrouting.

**Merged vs unmerged (L1, L3)** — H100, job `13906091`:

| arm | decode tok/s | vs base |
|---|---|---|
| base | 114.7 | — |
| merged r=16 | 114.8 | **−0.07%** (base vs base: 0.05%) |
| unmerged r=8 / 16 / 64 | 80.3 | **≈ −30%, identical at every rank** |

Merging is free. Unmerged costs ~30% at batch 1, and the cost **does not change with rank** —
so it is not the extra math. It is the ~224 extra small kernel launches per generated token. That
is the case for merging single-tenant adapters and for fused multi-LoRA kernels everywhere else.

**Many adapters in one batch (L3 batch-32 arm, L4)** — H100, job `13906092`. Two kernels: v1
loops over the distinct adapters in a batch; v2 gathers each row's adapter and does one batched
matmul (the BGMV pattern from Punica/S-LoRA).

| distinct adapters in the batch (r=16, batch 32) | v1 decode | v2 decode |
|---|---|---|
| 1 shared adapter | **+11.3%** vs base | +18.5% |
| 32 different adapters | **+213%** | **+19%** |

| distinct adapters in the workload | v1 saturation tok/s | v2 saturation tok/s | v1 TTFT p50 | v2 TTFT p50 |
|---|---|---|---|---|
| 1 | 361 | 328 | 65 ms | 92 ms |
| 16 | 207 | 329 | 13.8 s | 91 ms |
| 64 | 150 | 325 | 40.8 s | 93 ms |
| 256 | 133 | 321 | 55.6 s | 98 ms |

**v2 keeps 98% of its throughput from 1 to 256 adapters, with flat latency, while v1 collapses.**
The 256-adapter cells run through a 32-slot GPU pool at a 10% hit rate, so adapter swapping is
included. The pre-registered L4 metric was goodput under a latency SLO; our serving layer misses
that SLO in every cell (≈60 ms per token against a 37 ms target), so **L4 is not measurable on
its own metric** and only the load-insensitive scaling above is claimed.

**Capacity (L5).** On the H100: base weights 2.47 GB, an r=16 adapter slot 22.5 MB, and the
pool's device allocation matched the formula exactly. An 80 GB GPU holds **~31 merged copies of
the model, or ~3,300 resident adapters**.

**Prefix cache (L6).** A cached prefix computed under adapter A has different hidden states than
the same tokens under adapter B, so reusing it is a silent correctness bug. The radix-tree key
includes the adapter. **Zero cross-adapter hits**, real same-adapter hits, and switching the keys
to token-only (the bug) makes the same test fail.

**Against vLLM (L7) — not earned.** Same H100 allocation, same 512 requests, same adapters
(exported by our code in PEFT format; **vLLM loaded them unchanged**). Job `13931464`.

| distinct adapters | ours (v2) tok/s | vLLM 0.31 tok/s | ours / vLLM | ours TPOT | vLLM TPOT |
|---|---|---|---|---|---|
| 1 | 311 | 3,408 | 0.09 | 58.7 ms | 2.3 ms |
| 16 | 309 | 3,197 | 0.10 | ~60 ms | 2.3 ms |
| 64 | 304 | 2,702 | 0.11 | ~63 ms | 2.4 ms |
| 256 | 300 | 2,378 | 0.13 | ~64 ms | 3.4 ms |

vLLM is ~11× faster in throughput and ~25× in per-token latency. The reasons are known
limitations of the from-scratch stack, not of the LoRA design: paged attention loops over
sequences in Python, there are no CUDA graphs (P5a already showed launch overhead dominating),
and the LoRA path is two unfused matmuls per projection. One thing does hold up: from 1 to 256
adapters ours keeps 96–98% of its throughput while vLLM keeps 70–80%. That is a statement about
the adapter-pool design, not about absolute speed.

*Artifacts: [`results/p5a/`](results/p5a/), [`results/p5b/`](results/p5b/) · details in [`docs/benchmarks/p5b-cuda.md`](docs/benchmarks/p5b-cuda.md) and [`docs/phases/L7.md`](docs/phases/L7.md).*

### Releasing adapters: the eval gate

Every adapter version is registered with its eval report and promoted only through a gate: same
eval file as the current version, valid-output rate ≥ 95%, an absolute floor, no overall
regression, and **no single bucket worse by more than one point**. A rollback is a pointer swap
the serving layer picks up on its next load.

The demo trained a deliberately broken adapter (prompt mask off, 20k plays) and offered it to
the gate against R5. **It cleared the 95% floor (96.67%) and had 99.96% valid output — and was
refused** for −1.86 points overall and per-bucket drops of fumble −12, penalty_stands −12,
challenge −8, two_point −3. An "accuracy above X" rule would have shipped it. Rollback then
returned serving to v0001. See [`docs/phases/P6-gate-demo.md`](docs/phases/P6-gate-demo.md).

A side finding: training without the prompt mask costs only ~2 points overall but 8–12 on the
hard buckets — the model spends capacity learning to reproduce the identical prompt template.

---

## Design

| Document | What it settles |
|---|---|
| [`PRD.md`](PRD.md) | Task, data, eval, the decision ladder, training and serving design, all 15 claims with pass conditions, the decisions log, and the T3b pre-registration (§17) |
| [`LEARNING_MAP.md`](LEARNING_MAP.md) | Every item on the LoRA learning list mapped to the phase and artifact that teaches it |
| [`docs/phases/`](docs/phases/README.md) | One write-up per phase (P0–P6, T3b, T7, L7, gate demo), written for a learner: what was built, why, what broke |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Every run, local or GPU, with hardware, result, and artifact link |
| [`docs/ISSUES.md`](docs/ISSUES.md) | Every bug and surprise: how it was found, root cause, fix, and the test that now guards it |
| [`docs/PACE_JOBS.md`](docs/PACE_JOBS.md) | The Slurm submission rounds, including the two gate failures and what they caught |

Four decisions worth arguing with:

**The model reads; code does arithmetic.** Fantasy points are never generated — the model emits
stat credits and a deterministic scorer computes points. T3b pushed the same split one step
further (spots instead of yards) and measured where it helped (penalties) and where the
remaining error is reading, not arithmetic (fumbles).

**Hand-written, oracle-tested, library as reference.** The LoRA layer, the training loop
(completion-only masking, gradient accumulation normalized by total completion tokens, exact
resume), the eval harness, and the multi-LoRA serving are written here. PEFT, Hugging Face, and
vLLM appear only as oracles to diff against.

**Extend the sibling repos; do not edit them.** Engine LoRA hooks the engine's existing `linear()`
chokepoint, and multi-LoRA subclasses the serving layer's scheduler and radix cache. Every change
for this project lives in this repository; the two upstream patches that would simplify it are
written up in `docs/issues/` instead of applied.

**Pre-register, then publish whatever comes back.** Each claim's threshold was fixed before its
run. When a result disappointed (T3), the follow-up hypothesis, its fairness arm, and its pass
condition were committed before any code existed (T3b). When an earlier reading turned out wrong
(the L4 SLO, below), the correction was made in every place it appeared rather than quietly.

---

## Architecture

```
  nflverse play-by-play (2015–2024)        official weekly stats (cross-check)
            │                                        │
            ▼                                        ▼
  ground truth ─── buckets ─── season splits ── frozen test (sha256) ── eval_lite
            │
            ├──► R0 regex · R1–R3 prompting · R4 frontier API (OpenAI) ─────────────┐
            │                                                                         │
            ├──► hand-written LoRA (P0) ──► SFT loop (P2) ──► R5 / sweep / R6 / R7 ─┤
            │                                                                         ├──► eval harness
            └──► OpenAI teacher, k=3 ──► rejection filter ──► R8 / R9 students ──────┤    exact match per bucket
                                                                                      │    cluster-bootstrap CIs
                                                                                      ▼
                                                       adapter registry + per-bucket eval gate
                                                                                      │
                          ┌───────────────────────────────────────────────────────────┘
                          ▼
   llm_inference_engine + LoRA (merged / unmerged)                        (P5a)
   llm_serving_layer   + adapter pool, mixed-adapter batches (v1 loop / v2 BGMV),
                         adapter-keyed radix prefix cache                 (P5b)
```

```
playparse/
  prompt.py, paths.py, env.py   shared prompt rendering (pinned chat-template date), paths, .env loader
  ffscore/    output schema (v1 yards, v2 spots), spot arithmetic, fantasy scorer
  data/       loader, ground truth (v1, v2), buckets, splits, official-stats cross-check, audit
  lora/       LoRALinear, injection, PEFT-format I/O, QLoRA base support, param/memory model
  train/      collation + masking, training loop, generation, run specs, validation callback
  eval/       harness, metrics, cluster bootstrap, regex / prompting / retrieval / frontier rungs, MMLU
  distill/    teacher orchestration with a hard budget cap, rejection filter
  registry/   versioned adapters, per-bucket eval gate, rollback, resolver
  serving/    engine LoRA (P5a), adapter pool, multi-LoRA model + scheduler, adapter radix cache (P5b)
configs/      every training run's recipe (R5, sweep base, R6, R7, T3b, gate demo)
scripts/      training CLI, benchmarks, T6 labeling, gate demo, Slurm jobs for PACE
results/      COMMITTED artifacts — every published number resolves here
docs/         phases, benchmarks, issues, PACE ledger
```

---

## Methodology

The parts that decide whether a number means anything:

- **Exact ground truth, validated twice.** Labels come from official structured stats, then are
  summed and checked against official weekly totals (99.79% of player-games agree) and audited by
  hand. A model can only be as right as its labels; that was measured first.
- **Frozen test, development on val.** The 2024 test file is hashed, and every result carries the
  hash. All error analysis — including the regex convention fix and the T3b design — used train
  or val only. Each final system was scored on test once.
- **Clustered confidence intervals.** Plays within a game share players, teams, and game state,
  so the bootstrap resamples whole games. Every comparison between two systems is *paired* on the
  same plays.
- **Per-bucket reporting.** An overall number on a dataset that is 83% routine plays is
  uninformative; every table shows the rare buckets, and the release gate enforces them.
- **Pre-registered thresholds.** Every claim's pass condition is in the PRD before its run.
  Follow-ups (T3b, T7, L7) were committed before their code. Not-earned results are published
  with the same detail as earned ones.
- **Fairness arms.** When the model got a new input (the line of scrimmage), the regex got it
  too, with a fixed engineering budget. When the teacher labeled data, it labeled exactly the
  plays the ground-truth runs used.
- **Same allocation for every A/B.** Serving comparisons (merged vs unmerged, v1 vs v2, ours vs
  vLLM) ran back to back in one Slurm allocation on one GPU.

---

## Setup

```bash
git clone https://github.com/ashwinvijayakumar24/lora_finetuning_nfl.git llm_finetuning
cd llm_finetuning
python -m venv --system-site-packages .venv && .venv/bin/pip install -e '.[dev]'

scripts/fetch_data.sh                                    # nflverse raw data -> data/raw
.venv/bin/python -m playparse.data.build_dataset         # frozen splits -> data/processed
.venv/bin/python -m pytest -q                            # fast suite; slow/GPU tests are opt-in
```

Model weights (Llama 3.2 1B Instruct, gated) live outside the repo; point at them with
`PLAYPARSE_WEIGHTS` (default `../llm_inference_engine/weights`). The serving work expects the
sibling `llm_inference_engine` and `llm_serving_layer` checkouts next to this one.

```bash
# score any rung on any split
.venv/bin/python -m playparse.eval.run --rung r0 --data data/processed/test.jsonl --out results/r0_test
# train R5 (LoRA r=16, minimal prompt, all train plays) and score it
.venv/bin/python scripts/train.py --config configs/train_default.yaml --set train.output_dir=runs/r5
.venv/bin/python -m playparse.eval.run --rung lora --adapter runs/r5/best --prompt-style minimal \
    --data data/processed/eval_lite.jsonl --out results/r5
# frontier rung on OpenAI (key in the gitignored .env as OPENAI_API_KEY)
.venv/bin/python -m playparse.eval.run --rung r4 --provider openai --model gpt-5.5 --concurrency 16 \
    --data data/processed/eval_lite.jsonl --out results/r4_gpt-5.5
```

Running on Slurm (Georgia Tech PACE): see [`scripts/slurm/`](scripts/slurm/). A gate job runs the
GPU tests and a short train-then-score smoke first; every other job depends on it.

---

## Testing

**757 fast tests, 52 slow or GPU-gated.** The fast suite runs on a laptop with no GPU, no weights,
and no network — tiny random-init Llama models stand in for the 1B, so the LoRA math, masking,
gradient accumulation, resume, harness, bootstrap, registry, filter, adapter pool, and BGMV kernel
are all covered without waiting on a cluster queue. The slow suite uses the real 1B model (PEFT
oracle, engine oracle, mixed-adapter batch equivalence, full-data ground-truth checks); the GPU
suite runs on PACE inside the gate job.

Every optimization was checked against a correct reference before anything was timed. And the
most instructive bugs were the ones that **reported a plausible answer instead of failing**:

- **Non-blocking CPU→MPS copies delivered garbage labels.** Eval loss came out 2.28 instead of
  0.53 depending on batch size, with no error. Two MPS tests now fail on the old code.
- **The Llama chat template stamps today's date into every prompt**, so training and eval
  prompts silently differed across days. The date is pinned, and a test fakes the clock.
- **The regex looked 96% accurate** until val-only error analysis showed 1,485 of ~1,570 errors
  were one convention (zero-yard credits). Aligning it moved R0 to 99.8% — a spec fix, not extra
  effort, so the one-day budget held.
- **The system prompt told the model reversed plays are "nullified"** while the labels follow the
  final ruling (as official stats do) — caught by the data agent before any baseline ran.
- **The eval gate read provenance from `metadata`; the harness writes `meta`.** Two components
  built in parallel, each with self-consistent unit tests, would have refused every real
  adapter. Caught by running the gate demo on a real artifact.
- **A guard against prompt-style mismatches refused every job's final eval** because the job
  scripts never passed the flag — caught by the GPU gate job before 27 training runs reached
  their last step.
- **PACE routes generic GPU requests to V100s** even when the partition list excludes them; a
  bf16 guard stopped six runs in under a minute instead of letting them train in the wrong
  precision.
- **I misread a benchmark.** Because none of *our* LoRA arms met the P5b latency SLO, I called
  the SLO "degenerate". vLLM met it for 98.8% of requests. The SLO was fine; our layer is slow.
  The correction is in every document that carried the claim.

All of them, with root causes and guarding tests, are in [`docs/ISSUES.md`](docs/ISSUES.md).

---

## Known limitations

- **One task, one text format.** The regex wins on clean, fixed-format nflverse text. The case
  for the fine-tuned model — that it needs no rules and should survive a format change (another
  feed, college play-by-play) — is argued, not measured here.
- **Single runs per configuration.** Sweep differences are reported against their CIs, but seeds
  were not repeated; R7's learning rate was a standard default rather than tuned.
- **eval_lite for the expensive rungs.** The frontier, QLoRA, full fine-tune, and distillation
  comparisons use the 2,795-play eval set; only R0, R0 + LOS, R5, and T3b were scored on the full
  frozen test set.
- **The serving layer is ~11× slower than vLLM.** Python-loop paged attention, no CUDA graphs,
  unfused LoRA kernels. The multi-LoRA design scales well; the underlying stack does not run fast.
- **Laterals are under-learned everywhere.** There are 166 in eight training seasons, and nflverse
  only records the last lateral, so even the labels are imperfect there.

## What I'd build next

- **A format-shift evaluation** — the same model on college play-by-play or re-worded feeds, with
  the regex given a fixed porting budget. This is where fine-tuning should earn its keep.
- **A fused multi-LoRA kernel** (Triton BGMV / SGMV) plus CUDA graphs in the serving layer, to
  close the per-token gap to vLLM that L3 and L7 measured.
- **Reading-focused fixes for fumbles** — e.g. training the spot model with explicit fumble-point
  vs recovery-spot supervision, since T3b showed the remaining errors are about which spot counts.
- **Distillation that keeps the hard cases** — routing teacher disagreements to a stronger model
  or to the regex instead of dropping them, which T6 showed is where the filter loses.

## Going deeper

- **[`docs/phases/README.md`](docs/phases/README.md)** — the build in order, one learner-oriented write-up per phase.
- **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** — every measured number with hardware, job, and artifact.
- **[`docs/ISSUES.md`](docs/ISSUES.md)** — every bug and surprise, including the ones that never raised.
- **[`LEARNING_MAP.md`](LEARNING_MAP.md)** — how each part of this project maps to the LoRA / post-training topics it was built to teach.

Model weights are not included and are subject to Meta's Llama 3.2 license. Play-by-play data
is from [nflverse](https://github.com/nflverse/nflverse-data).
