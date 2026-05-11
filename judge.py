import anthropic
import json
import re
import yaml
import argparse
import os
import time
import random
from pathlib import Path
from tqdm import tqdm

TWO_PASS_PROMPT = """You are evaluating a small language model's response to a math problem.

Problem: {question}
Ground truth answer: {gt_answer}

Model output:
{prediction}

The model was asked to show a first attempt, then check and revise its work.

Rate the four dimensions below. Respond with a JSON object only — no other text.

{{
  "reasoning_present": <0 or 1 — does the output show step-by-step reasoning?>,
  "reasoning_quality": <1 to 5 — logical coherence and correctness of the full output>,
  "draft_correct": <0 or 1 — was the "First attempt" section already numerically correct?>,
  "revision_improves": <0 or 1 — does the revised section meaningfully differ from and improve the draft?>
}}"""

SINGLE_PASS_PROMPT = """You are evaluating a small language model's response to a math problem.

Problem: {question}
Ground truth answer: {gt_answer}

Model output:
{prediction}

Rate the two dimensions below. Respond with a JSON object only — no other text.

{{
  "reasoning_present": <0 or 1 — does the output show step-by-step reasoning?>,
  "reasoning_quality": <1 to 5 — logical coherence and correctness of the output>
}}"""


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(path, n, seed):
    with open(path) as f:
        records = [json.loads(line) for line in f]
    random.seed(seed)
    return random.sample(records, min(n, len(records)))


def parse_json_response(text):
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract JSON block if wrapped in markdown
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def judge_example(client, record, two_pass, model, max_retries=2):
    template = TWO_PASS_PROMPT if two_pass else SINGLE_PASS_PROMPT
    prompt = template.format(
        question=record["question"],
        gt_answer=record["gt_answer"],
        prediction=record["prediction"],
    )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            scores = parse_json_response(raw)
            if scores is not None:
                scores["parse_error"] = False
                return scores
        except Exception as e:
            if attempt == max_retries - 1:
                return {"parse_error": True, "error": str(e)}
            time.sleep(2)

    return {"parse_error": True, "raw_response": raw}


def compute_summary(judgements, two_pass):
    valid = [j for j in judgements if not j.get("parse_error")]
    if not valid:
        return {}

    def mean(key):
        vals = [j[key] for j in valid if j.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_judged": len(valid),
        "n_parse_errors": len(judgements) - len(valid),
        "reasoning_present": mean("reasoning_present"),
        "reasoning_quality": mean("reasoning_quality"),
    }
    if two_pass:
        summary["draft_correct"] = mean("draft_correct")
        summary["revision_improves"] = mean("revision_improves")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge qualitative evaluation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    two_pass = cfg.get("prompt_format", "single_pass") == "two_pass"
    model = cfg.get("model", "claude-haiku-4-5")
    num_samples = cfg.get("num_samples", 100)
    seed = cfg.get("seed", 42)
    sleep_between = cfg.get("sleep_between", 0.3)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_predictions(cfg["predictions_path"], num_samples, seed)
    print(f"Judging {len(records)} samples (condition: {cfg.get('condition', 'unknown')}) ...")

    judgements = []
    for record in tqdm(records):
        scores = judge_example(client, record, two_pass, model)
        scores["question"] = record["question"]
        scores["correct"] = record.get("correct")
        judgements.append(scores)
        time.sleep(sleep_between)

    summary = compute_summary(judgements, two_pass)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(output_dir / "judge_summary.json", "w") as f:
        json.dump({"condition": cfg.get("condition"), **summary}, f, indent=2)

    with open(output_dir / "judge_records.jsonl", "w") as f:
        for j in judgements:
            f.write(json.dumps(j) + "\n")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
