# NFL Football Fine-Tuning: Turn Play-by-Play Text into Fantasy Stats with LoRA, and Serving Many Fine-Tuned Adapters at Once

I fine-tuned Llama 3.2 1B with a hand-written LoRA implementation to read raw NFL play
descriptions and output exact fantasy stat credits as JSON. I then measured every alternative
an applied-AI team would consider: a regex, prompting, a frontier API model, QLoRA, a full
fine-tune, and distillation. Finally, I served the adapters on my own
[inference engine](https://github.com/ashwinvijayakumar24/llm_inference_engine) and
[serving layer](https://github.com/ashwinvijayakumar24/llm_serving_layer), with many adapters
batched together, and benchmarked against vLLM.

Ground truth comes from official NFL stats (nflverse), so every quality number is an exact
match, with no LLM judge. Training and benchmarks ran on A100, L40S, and H100 GPUs (Georgia
Tech PACE). Every claim's pass condition was written down before its run, and losses are
published alongside wins.

## Results

| Question | Result |
|---|---|
| Does fine-tuning beat prompting? | **Yes.** 0% zero-shot → **99.54%** on 37,859 held-out 2024 plays |
| Does the 1B model beat a frontier model? | **Yes.** 98.5% vs **gpt-5.5**'s 98.0%, at **~1/170th the cost** per play |
| Does it beat a hand-written regex? | **No.** 99.54% vs 99.79%. A pre-registered fix got to 99.73%, still short |
| LoRA vs full fine-tune vs QLoRA | LoRA **98.5%**, full fine-tune 97.4%, 4-bit QLoRA 98.1% |
| Does fine-tuning cause forgetting? | **No.** MMLU 31.1% → 32.3% |
| Does distillation from a frontier teacher work? | **Partly.** Students lose 1–3 points vs true labels; filtering recovers ≤31% |
| Multi-adapter serving | **~3,300 adapters vs 31 model copies** per GPU; throughput flat from 1 → 256 adapters |
| vs vLLM | **~11× slower.** Correct and scalable, but the from-scratch stack isn't optimized |

## Key findings

- **LoRA settings barely matter; data does.** 25 one-setting runs land within 0.4 points of each
  other, and rank 2 (1.4M parameters) matches rank 64 (45M). Going from 1k to 294k training
  plays moves accuracy from 84.6% to 98.5%.
- **The model reads well but struggles with field-position arithmetic.** It loses to the regex
  only on fumbles, laterals, and penalties, where official yards come from a spot on the field.
  Having the model output spots and computing yards in code fixed penalties (93.5% → 100%).
- **A teacher's mistakes cap the student.** The frontier teacher is wrong mostly on the hard plays
  (67% of laterals), and rejection filtering removes exactly those plays. So cleaner labels didn't
  help where it counted.
- **Merged adapters are free; unmerged ones cost ~30%.** That overhead is kernel-launch overhead,
  not math: it's identical at every rank. That's why production systems merge or fuse.
- **Release gates must check every category.** A deliberately broken adapter passed a 95% overall
  threshold, but per-category checks caught 8–12 point drops on hard plays and blocked it.

## What I built

- **LoRA from scratch**, matching Hugging Face PEFT exactly, saving adapters in PEFT format (vLLM
  loads them unchanged), plus QLoRA support.
- **A hand-written training loop**: prompt masking, exact gradient accumulation, checkpoint resume.
- **An evaluation harness**: per-category exact match, confidence intervals over games, frozen test set.
- **Multi-LoRA serving** on my own engine: merged and unmerged adapters, a batched multi-adapter
  kernel, an adapter memory pool, and a prefix cache that never mixes adapters.
- **An adapter registry** with a per-category release gate and rollback.

**Stack:** Python, PyTorch, Hugging Face, PEFT (as reference), bitsandbytes, vLLM (baseline),
OpenAI API, Slurm. 757 tests.

## More detail

- [`REPORT.md`](REPORT.md): the full write-up, with every table, method, and artifact.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): every run with its hardware and result files.
- [`docs/ISSUES.md`](docs/ISSUES.md): every bug found, including the ones that failed silently.
- [`docs/phases/`](docs/phases/README.md): the build, phase by phase.
