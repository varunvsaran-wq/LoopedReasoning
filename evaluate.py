import json
import re
import yaml
import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

SINGLE_PASS_SYSTEM = (
    "Solve the following problem step by step. Show your reasoning clearly, "
    "then state your final answer on a new line as: Answer: value."
)

TWO_PASS_SYSTEM = (
    "Solve the following math problem step by step. Show a first attempt, "
    "then check your work and give a revised final answer as: Answer: value."
)

BBH_SYSTEM = (
    "Answer the following question. Show your reasoning step by step, "
    "then state your final answer on a new line as: Answer: value."
)

BBH_TASKS = [
    "multistep_arithmetic_two",
    "logical_deduction_five_objects",
    "word_sorting",
    "causal_judgement",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    adapter_path = cfg.get("adapter_path")
    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def make_prompt(tokenizer, system, question):
    messages = [{"role": "user", "content": f"{system}\n\nProblem: {question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_bbh_prompt(tokenizer, question):
    messages = [{"role": "user", "content": f"{BBH_SYSTEM}\n\nQuestion: {question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batched(model, tokenizer, prompts, max_new_tokens, batch_size):
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens (same padded length for whole batch)
        prompt_len = inputs["input_ids"].shape[1]
        for ids in out_ids:
            outputs.append(tokenizer.decode(ids[prompt_len:], skip_special_tokens=True))
    return outputs


def extract_answer(text, two_pass=False):
    # For two-pass outputs take the LAST Answer: tag (after the revision)
    if two_pass:
        matches = list(re.finditer(r"Answer:\s*([^\n]+)", text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip().replace(",", "").lower()
    else:
        match = re.search(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace(",", "").lower()
    # Fallback: last number in output
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def extract_gsm8k_gt(answer_str):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if match:
        return match.group(1).replace(",", "").lower()
    return answer_str.strip().lower()


def evaluate_gsm8k(model, tokenizer, cfg):
    print("\n--- GSM8K ---")
    gsm_cfg = cfg.get("gsm8k", {})
    max_new_tokens = gsm_cfg.get("max_new_tokens", 512)
    batch_size = cfg.get("batch_size", 4)
    two_pass = cfg.get("prompt_format", "single_pass") == "two_pass"
    system = TWO_PASS_SYSTEM if two_pass else SINGLE_PASS_SYSTEM
    num_examples = gsm_cfg.get("num_examples", None)

    dataset = load_dataset("gsm8k", "main", split="test")
    if num_examples:
        dataset = dataset.select(range(num_examples))

    prompts = [make_prompt(tokenizer, system, ex["question"]) for ex in dataset]
    predictions = generate_batched(model, tokenizer, prompts, max_new_tokens, batch_size)

    records = []
    correct = 0
    for ex, pred_text in zip(dataset, predictions):
        pred = extract_answer(pred_text, two_pass=two_pass)
        gt = extract_gsm8k_gt(ex["answer"])
        is_correct = pred == gt
        correct += int(is_correct)
        records.append({
            "question": ex["question"],
            "prediction": pred_text,
            "pred_answer": pred,
            "gt_answer": gt,
            "correct": is_correct,
        })

    accuracy = correct / len(records)
    print(f"GSM8K accuracy: {accuracy:.4f} ({correct}/{len(records)})")
    return {"accuracy": accuracy, "n": len(records), "correct": correct}, records


def evaluate_bbh(model, tokenizer, cfg):
    print("\n--- BBH ---")
    bbh_cfg = cfg.get("bbh", {})
    max_new_tokens = bbh_cfg.get("max_new_tokens", 256)
    batch_size = cfg.get("batch_size", 4)
    tasks = bbh_cfg.get("tasks", BBH_TASKS)

    task_results = {}
    for task in tasks:
        print(f"  Task: {task}")
        dataset = load_dataset("lukaemon/bbh", task, split="test", trust_remote_code=True)

        prompts = [make_bbh_prompt(tokenizer, ex["input"]) for ex in dataset]
        predictions = generate_batched(model, tokenizer, prompts, max_new_tokens, batch_size)

        correct = 0
        for ex, pred_text in zip(dataset, predictions):
            pred = extract_answer(pred_text)
            gt = str(ex["target"]).strip().lower()
            # Normalize: strip parens common in BBH targets like "(A)"
            pred_norm = re.sub(r"[().]", "", pred or "").strip()
            gt_norm = re.sub(r"[().]", "", gt).strip()
            is_correct = pred_norm == gt_norm
            correct += int(is_correct)

        acc = correct / len(dataset)
        print(f"    {task}: {acc:.4f} ({correct}/{len(dataset)})")
        task_results[task] = {"accuracy": acc, "n": len(dataset), "correct": correct}

    macro_avg = sum(v["accuracy"] for v in task_results.values()) / len(task_results)
    print(f"  BBH macro-average ({len(tasks)} tasks): {macro_avg:.4f}")
    task_results["macro_avg"] = macro_avg
    return task_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a condition on GSM8K and BBH.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg)

    gsm8k_summary, gsm8k_records = evaluate_gsm8k(model, tokenizer, cfg)
    bbh_summary = evaluate_bbh(model, tokenizer, cfg)

    summary = {
        "condition": cfg.get("condition", "unknown"),
        "model_id": cfg["model_id"],
        "adapter_path": cfg.get("adapter_path"),
        "prompt_format": cfg.get("prompt_format", "single_pass"),
        "gsm8k": gsm8k_summary,
        "bbh": bbh_summary,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "gsm8k_predictions.jsonl", "w") as f:
        for r in gsm8k_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
