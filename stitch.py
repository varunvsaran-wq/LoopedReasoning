import json
import re
import yaml
import argparse
from pathlib import Path

CHECK_SENTENCE = "Let me verify my calculation and correct any errors."

LOOP_TEMPLATE = (
    "Problem: {question}\n\n"
    "First attempt:\n{draft_text}\n\n"
    "Check: {check}\n\n"
    "Revised reasoning:\n{cot_trace}\n\n"
    "Answer: {teacher_answer}"
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_answer(text):
    match = re.search(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(",", "").lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def build_loop_trace(record, cfg):
    question = record[cfg["question_field"]]
    draft_text = record["draft_text"]
    cot_trace = record[cfg["cot_field"]]
    teacher_answer = record[cfg["teacher_answer_field"]]

    return LOOP_TEMPLATE.format(
        question=question,
        draft_text=draft_text,
        check=CHECK_SENTENCE,
        cot_trace=cot_trace,
        teacher_answer=teacher_answer,
    )


def main():
    parser = argparse.ArgumentParser(description="Stitch drafts + CoT traces into loop format.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(cfg["input_path"]) as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} records.")

    condition_e = []
    condition_f = []

    for record in records:
        gt = str(record[cfg["gt_answer_field"]]).replace(",", "").lower()
        draft_answer = extract_answer(record.get("draft_text", ""))

        loop_trace = build_loop_trace(record, cfg)

        out = {
            "question": record[cfg["question_field"]],
            "loop_trace": loop_trace,
            "draft_answer": draft_answer,
            "teacher_answer": record[cfg["teacher_answer_field"]],
            "gt_answer": gt,
            "draft_correct": draft_answer == gt,
        }

        condition_e.append(out)
        if draft_answer != gt:
            condition_f.append(out)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    path_e = Path(cfg["output_dir"]) / "condition_e.jsonl"
    path_f = Path(cfg["output_dir"]) / "condition_f.jsonl"

    with open(path_e, "w") as f:
        for r in condition_e:
            f.write(json.dumps(r) + "\n")

    with open(path_f, "w") as f:
        for r in condition_f:
            f.write(json.dumps(r) + "\n")

    print(f"Condition E (all):         {len(condition_e)} examples → {path_e}")
    print(f"Condition F (error-only):  {len(condition_f)} examples → {path_f}")
    print(f"Draft error rate: {len(condition_f) / len(condition_e):.1%}")

    if len(condition_f) < 150:
        print(
            "WARNING: Condition F has fewer than 150 examples. "
            "Consider loosening the filter in stitch.yaml (set loose_filter: true)."
        )


if __name__ == "__main__":
    main()
