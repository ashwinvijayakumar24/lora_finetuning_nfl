# T6: distillation — does rejection sampling recover the cost of teacher labels?

**Date:** 2026-10-10/11. **Teacher:** gpt-5.4-mini (8-shot R4 prompt, reasoning low), k = 3
samples per play, $139.64 for 100,000 train plays. **Students:** R5's recipe
(`configs/train_default.yaml`) trained on teacher labels for **exactly the plays** the
ground-truth data-size runs used (`nested_sample(train, N, seed 0)`), so the three curves
differ only in where labels came from. **R8** = sample 0 if it parses (unfiltered).
**R9** = rejection-filtered (schema, yardage literally in text, names literally in text,
all 3 samples agree). Ground truth was never used by the filter. Eval: eval_lite, paired
game-clustered bootstrap. Artifacts: `results/t6/`, `results/eval_lite/r8_n*/`, `r9_n*/`.

## Label quality (vs ground truth, never used to filter)

| Plays | R8 labels correct | R9 labels correct | R9 plays kept |
|---|---|---|---|
| 1k | 96.9% | 98.9% | 957 |
| 5k | 96.8% | 98.9% | 4,775 |
| 20k | 97.1% | 99.1% | 19,159 |
| 100k | 97.5% | 99.2% | 95,987 |

## Student accuracy (exact match on eval_lite)

| Plays | True labels | R8 | R9 | Truth − R8 | R9 − R8 | Gap closed |
|---|---|---|---|---|---|---|
| 1k | 84.58 | 83.40 | 83.51 | +1.18 [+0.57, +1.73] | +0.11 [−0.69, +1.11] | 9% |
| 5k | 96.28 | 93.27 | 94.20 | +3.01 [+2.11, +3.93] | **+0.93 [+0.09, +1.57]** | 31% |
| 20k | 97.67 | 96.85 | 96.82 | +0.82 [+0.47, +1.26] | −0.04 [−0.25, +0.22] | −5% |
| 100k | 98.07 | 96.46 | 96.78 | +1.61 [+1.07, +2.33] | **+0.32 [+0.03, +0.73]** | 20% |

## Verdict: T6 not earned

The pre-registered condition was that filtering "closes most of the R8 → ground-truth gap
at a stated data size". It closes at most 31% (5k), 20% at 100k, and nothing at 1k or 20k.
Filtering does help significantly at 5k and 100k, just not by "most".

## Why: the teacher is wrong where the task is hard, and the filter removes those plays

| Bucket | Teacher (R8) label error | Filter drops |
|---|---|---|
| lateral | 66.7% | 40.0% |
| fumble | 22.0% | 16.0% |
| two_point | 12.3% | 20.2% |
| penalty_stands | 12.5% | 8.9% |
| normal | 2.1% | 4.1% |
| challenge / interception / td / penalty_nullified | 0.2–3.4% | 0.0–3.2% |

The teacher's errors concentrate in exactly the buckets that separate good models from
great ones, and the agreement filter preferentially drops those plays. The filtered
student gets cleaner labels but fewer of the hard examples it most needs, so its accuracy
on the hard buckets cannot reach the ground-truth student's. The student trained on 100k
teacher labels (96.5–96.8%) also lands *below* the ground-truth student at 20k (97.7%):
a teacher's systematic errors cap the student, and more teacher data does not fix them.

## Lessons for the decision ladder

- Distillation from a cheaper frontier model is real but bounded by the teacher: here a
  95.2%-accurate teacher (gpt-5.4-mini 8-shot, see T2) produced students at ~96.5–96.8%,
  slightly *better* than the teacher itself, but ~1.3–1.6 points below training on truth.
- Rejection sampling buys label precision (97.5% → 99.2%) at the price of coverage, and
  that coverage loss is not uniform: it lands on the hard cases.
- When ground truth exists, use it. When it does not, measure where the filter drops
  data, not just how many labels it keeps.
