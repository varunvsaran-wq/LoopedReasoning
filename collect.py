import anthropic
import json
import yaml
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

DRAFT_PROMPT = (
    "Solve the following math problem step by step. Give your first attempt at a "
    "solution working through the arithmetic. Don't double-check your work yet — "
    "just solve it.\n\n"
    "Problem: {problem}\n\n"
    "First attempt:"
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_done(output_path, question_field):
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                done.add(record[question_field])
    return done


def generate_draft(client, problem, model, max_tokens):
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": DRAFT_PROMPT.format(problem=problem)}],
    )
    return response.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate draft passes via Haiku API.")
    parser.add_argument("--config", required=True, help="Path to collect YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    input_path = cfg["input_path"]
    output_path = cfg["output_path"]
    model = cfg.get("model", "claude-haiku-4-5")
    max_tokens = cfg.get("max_tokens", 200)
    sleep_between = cfg.get("sleep_between", 0.3)
    question_field = cfg.get("question_field", "question")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    done = load_done(output_path, question_field)

    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    to_process = [r for r in records if r[question_field] not in done]
    print(f"Total: {len(records)} | Already done: {len(done)} | Remaining: {len(to_process)}")

    errors = 0
    with open(output_path, "a") as out_f:
        for record in tqdm(to_process):
            problem = record[question_field]
            try:
                draft_text = generate_draft(client, problem, model, max_tokens)
                record["draft_text"] = draft_text
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                time.sleep(sleep_between)
            except Exception as e:
                errors += 1
                print(f"\nError ({errors}): {problem[:60]}... — {e}")
                time.sleep(2)

    print(f"\nDone. Errors: {errors}")


if __name__ == "__main__":
    main()
