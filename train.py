import json
import yaml
import argparse
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. Show a first attempt, "
    "then check your work and give a revised final answer."
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_records(records, tokenizer, question_field, loop_trace_field):
    texts = []
    for r in records:
        question = r[question_field]
        loop_trace = r[loop_trace_field]
        # loop_trace starts with "Problem: {question}\n\n" — strip it for the assistant turn
        prefix = f"Problem: {question}\n\n"
        response_body = loop_trace[len(prefix):] if loop_trace.startswith(prefix) else loop_trace
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nProblem: {question}"},
            {"role": "assistant", "content": response_body},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for loop trace conditions.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    records = load_jsonl(cfg["data_path"])
    texts = format_records(
        records,
        tokenizer,
        cfg.get("question_field", "question"),
        cfg.get("loop_trace_field", "loop_trace"),
    )
    dataset = Dataset.from_dict({"text": texts})
    print(f"Training examples: {len(dataset)}")

    train_cfg = cfg.get("training", {})
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_seq_length=train_cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        packing=False,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
