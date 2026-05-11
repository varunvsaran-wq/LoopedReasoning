# LoopedReasoning

Second experiment in a series on improving reasoning in small language models.

**Series context:**
- Part 1: [Black-Box Knowledge Distillation](https://github.com/varunvsaran-wq/BlackBoxDistillation) — single-pass CoT distillation from Claude Haiku 4.5 into Qwen2.5-1.5B. Best result: 30.02% → 46.02% GSM8K accuracy. Teacher ceiling: 64.75%.
- Part 2 (this repo): Looped Reasoning — can a structured two-pass draft-revise format close more of the remaining 18.73-point gap?

## Research Question

Does training a small model on two-pass draft-revise traces improve reasoning over single-pass CoT distillation at the same data scale and compute budget? And does the source of the correction signal matter — all examples vs. error-correcting examples only?

## Quick Start

```
pip install -r requirements.txt

# Generate draft passes (reuses existing CoT traces from Part 1)
python collect.py --config configs/collect.yaml

# Stitch traces and filter conditions
python stitch.py --config configs/stitch.yaml

# Train
python train.py --config configs/condition_e.yaml
python train.py --config configs/condition_f.yaml

# Evaluate
python evaluate.py --config configs/eval.yaml

# Judge
python judge.py --config configs/judge.yaml
```

## Results

| Condition | GSM8K | BBH (4-task) |
|---|---|---|
| Teacher (Haiku 4.5) | 64.75% | 69.85% |
| Baseline (untuned) | 30.02% | 12.08% |
| D: Single-pass CoT (Part 1) | 46.02% | 23.19% |
| E: Loop-All | TBD | TBD |
| F: Loop-Error-Only | TBD | TBD |

## Repo Structure

```
LoopedReasoning/
├── configs/           # YAML configs per condition
├── data/              # raw traces, stitched datasets, splits
├── design/            # experimental design documents
├── outputs/           # model checkpoints and eval results
├── collect.py         # generate draft passes via Haiku API
├── stitch.py          # combine drafts + existing traces into loop format
├── train.py           # QLoRA fine-tuning via TRL
├── evaluate.py        # GSM8K + BBH evaluation
└── judge.py           # LLM-as-judge qualitative eval
```

See [design/experiment.md](design/experiment.md) for full experimental design.
