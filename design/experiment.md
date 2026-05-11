# Experimental Design: Looped Reasoning via Two-Pass Draft-Revise Distillation

## Motivation

Part 1 of this series showed that full CoT + answer distillation from Claude Haiku 4.5 raises
Qwen2.5-1.5B GSM8K accuracy from 30.02% to 46.02%, while the teacher reaches 64.75%.
The 18.73-point gap suggests the single-pass CoT format may be a limiting factor — the student
gets one clean trace, but no signal about how to catch and fix errors mid-reasoning.

This experiment tests whether a structured two-pass trace — draft, check, revise — teaches
the model a more robust reasoning process at the same data scale and compute budget.

## Research Questions

1. Does two-pass loop training improve GSM8K accuracy over single-pass CoT distillation?
2. Does it matter whether the training traces contain genuine self-correction (wrong draft → right revision) vs. just a longer format (right draft → confirmed revision)?
3. Does the loop format transfer to BBH tasks the same way single-pass CoT did?

## Loop Format

The model generates a single output with two explicit reasoning passes before committing to an answer.

```
Problem: {question}

First attempt:
{draft reasoning — may contain errors}

Check: {one sentence noting what to verify or what was wrong}

Revised reasoning:
{corrected or confirmed step-by-step reasoning}

Answer: {value}
```

This is implemented entirely in the text domain. No architecture changes, no weight tying,
no RL. The student is fine-tuned on these traces via SFT, same as Part 1.

## Conditions

### Condition D — Single-Pass CoT Baseline
The Condition C checkpoint from Part 1 (46.02% GSM8K). No new training needed.
Serves as the primary comparison point for conditions E and F.

### Condition E — Loop-All
Fine-tune on two-pass traces for all examples where the teacher's final answer is correct,
regardless of whether the draft was wrong. This tests whether the two-pass *format*
alone helps, even when the teacher is just confirming a correct draft.

### Condition F — Loop-Error-Only
Fine-tune on two-pass traces filtered to cases where the teacher's draft answer was
wrong but the revised answer is correct. This is a smaller dataset but contains
only genuine self-correction signal. Tests whether correction-conditioned training
is more data-efficient than the full loop format.

| Condition | Data source | Filter | Expected N |
|---|---|---|---|
| D | Part 1 Condition C checkpoint | — | — |
| E | Stitched loop traces | Final answer correct | ~1,200–1,400 |
| F | Stitched loop traces | Draft wrong, final correct | ~200–280 |

## Data Strategy

Existing Part 1 data (3,000 filtered correct single-pass traces) is reused to avoid
regenerating full two-pass traces from scratch. This halves the API cost.

**What is reused:**
The existing CoT traces become the "Revised reasoning" block (Pass 2). They are
already filtered for correctness, so no additional quality check is needed for that block.

**What needs to be generated:**
Only the "First attempt" (Pass 2) block per example. This is a shorter generation
(~80–120 tokens) compared to a full trace (~200–300 tokens).

### Draft Generation Prompt

```
Solve the following math problem step by step. Give your first attempt at a solution
working through the arithmetic. Don't double-check your work yet — just solve it.

Problem: {problem}

First attempt:
```

Send to `claude-haiku-4-5`. Keep response as-is (no filtering at generation time).

### Stitching Logic (`stitch.py`)

For each existing record `{question, cot_trace, teacher_answer, gt_answer}`:

1. Send draft prompt to Haiku, get `draft_text`
2. Extract `draft_answer` from `draft_text` using same regex as evaluate.py
3. Build the full loop trace:
   ```
   Problem: {question}

   First attempt:
   {draft_text}

   Check: Let me verify my calculation and correct any errors.

   Revised reasoning:
   {cot_trace}

   Answer: {teacher_answer}
   ```
4. Write record with fields:
   `{question, draft_text, draft_answer, cot_trace, teacher_answer, gt_answer, loop_trace}`

### Filtering

- **Condition E**: keep all records where `teacher_answer == gt_answer` (already guaranteed by Part 1 filter)
- **Condition F**: additionally require `draft_answer != gt_answer` (draft was wrong)

If Condition F yields fewer than 150 examples, loosen to: draft answer is absent or
differs from teacher answer by any amount (captures rounding errors and format mismatches too).

## Training Configuration

All settings match Part 1 except sequence length and batch size.

| Hyperparameter | Part 1 | This experiment |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Training method | TRL SFTTrainer | TRL SFTTrainer |
| Epochs | 3 | 3 |
| Learning rate | 2e-4 | 2e-4 |
| Scheduler | Cosine | Cosine |
| Warmup ratio | 0.05 | 0.05 |
| Max sequence length | 512 | 1024 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Quantization | 4-bit NF4 | 4-bit NF4 |
| Random seed | 42 | 42 |

**Batch size by GPU:**

| GPU VRAM | per_device_batch_size | gradient_accumulation | effective batch |
|---|---|---|---|
| 24 GB (A10G) | 2 | 8 | 16 |
| 16 GB | 1 | 16 | 16 |

The sequence length increase from 512 to 1024 roughly doubles per-step compute.
Estimate ~1.5–2 hrs on A10G for Condition E (1,400 examples, 3 epochs).
Condition F trains faster due to smaller dataset.

## Evaluation Protocol

### GSM8K
Identical to Part 1: same CoT prompt, greedy decoding, `Answer:` tag extraction with
last-number fallback, exact-match accuracy on the 1,319-example test split.

At inference, prompt the model with the **two-pass prompt format** (so it has the
opportunity to use the loop). Also evaluate with the **single-pass prompt** to test
whether the loop is needed at inference or has been internalized.

### BBH
Same four tasks as Part 1 (excluding date understanding and tracking shuffled objects
due to protocol sensitivity):
- multistep arithmetic two
- logical deduction five objects
- word sorting
- causal judgement

### Judge Evaluation
100 randomly sampled GSM8K outputs per condition, scored by Claude Haiku 4.5.

**Dimensions (updated from Part 1):**
- `draft_correct` (0/1): was the first attempt already correct?
- `revision_improves` (0/1): does the revised reasoning differ meaningfully from the draft?
- `reasoning_quality` (1–5): logical coherence of the full output
- `reasoning_present` (0/1): does the output show step-by-step reasoning?

The `draft_correct` and `revision_improves` dimensions distinguish "the model learned
the two-pass format" from "the model learned to actually self-correct."

## Compute Budget

| Step | Location | Estimated time | API cost |
|---|---|---|---|
| Generate ~3,000 draft passes | Local | ~1 hr | ~$0.50–1.00 |
| Stitch + filter datasets | Local | 30 min | — |
| Train Condition E | RunPod | ~2 hrs | — |
| Train Condition F | RunPod | ~45 min | — |
| Evaluate all conditions | RunPod or local | ~1 hr | — |
| Judge eval | Local | ~30 min | ~$0.30–0.50 |
| **Total** | | **~5–6 hrs RunPod** | **~$1.50** |

## Expected Findings and Failure Modes

**If E > D**: The two-pass format helps even when the teacher isn't self-correcting.
The model benefits from seeing a longer, more deliberate reasoning structure.

**If F > E** (despite less data): Genuine self-correction signal is more data-efficient
than format imitation. This is the most interesting result for the series narrative.

**If E ≈ D**: Format alone doesn't help — the bottleneck is model capacity or
the quality of the reasoning content, not structure.

**If F < E**: The error-correcting subset is too small to generalize, or the stitched
traces are incoherent (draft and revision were generated independently and don't
flow naturally). Mitigation: inspect 20–30 stitched examples before training.

**Key risk — incoherent stitching**: The draft and revision are generated in separate
API calls, so they may not reference each other naturally. The "Check:" sentence is
a static string bridging them. Before training, manually read 20–30 stitched examples
to confirm they read as plausible self-correction rather than two disconnected attempts.
If they look broken, add a stitching prompt step: send the draft to Haiku and ask it
to write a one-sentence check and a revised solution (generating only Pass 2 and the
check sentence fresh, while keeping the existing CoT trace is dropped).

## Connection to Part 1

| Finding from Part 1 | How this experiment extends it |
|---|---|
| Format matters: CoT+Answer > Reasoning-Only | Tests whether *more structured* format (two-pass) > standard CoT |
| Reasoning-only (no answer anchor) hurt accuracy | Loop format preserves answer anchor; isolates format effect |
| Teacher gap is 18.73 points | Measures how much the loop format closes vs. single-pass |
| BBH transfer was uneven | Re-evaluates same four BBH tasks to track transfer pattern |
