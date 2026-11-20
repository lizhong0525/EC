"""Opt-in QLoRA for reviewed structured extraction. Test data is never trained on."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("training/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--revision", help="Verified immutable base-model commit")
    args = parser.parse_args()
    if not (args.data / "train.jsonl").is_file() or not (args.data / "validation.jsonl").is_file():
        raise SystemExit("Run prepare_qwen_training.py first; no ad-hoc chat input is accepted.")
    # Heavy dependencies are imported only for an explicitly requested training run.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU; no silent CPU fallback is supported.")
    if (args.output / "RUN.json").exists() or (
        args.output.is_dir() and get_last_checkpoint(str(args.output))
    ):
        raise SystemExit("Choose a new output directory; existing training records are preserved.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        completion = row["messages"][-1]["content"] + tokenizer.eos_token
        first = tokenizer(prompt, add_special_tokens=False)
        second = tokenizer(completion, add_special_tokens=False)
        ids = (first["input_ids"] + second["input_ids"])[: args.max_length]
        labels = ([-100] * len(first["input_ids"]) + second["input_ids"])[: args.max_length]
        if not any(label != -100 for label in labels):
            raise ValueError(f"{row['example_id']}: no target tokens remain after truncation")
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data / "train.jsonl"),
            "validation": str(args.data / "validation.jsonl"),
        },
    )
    encoded = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    def collate(rows: list[dict]) -> dict:
        longest = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [
                    row["input_ids"] + [tokenizer.pad_token_id] * (longest - len(row["input_ids"]))
                    for row in rows
                ]
            ),
            "labels": torch.tensor(
                [row["labels"] + [-100] * (longest - len(row["input_ids"])) for row in rows]
            ),
            "attention_mask": torch.tensor(
                [row["attention_mask"] + [0] * (longest - len(row["input_ids"])) for row in rows]
            ),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            logging_steps=args.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            report_to=[],
            seed=42,
        ),
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=collate,
    )
    training = trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))

    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    adapter_hashes = {
        path.name: file_hash(path)
        for path in (args.output / "adapter").iterdir()
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "revision": args.revision or "unfixed-explicitly-not-for-reproducible-release",
        "data": str(args.data),
        "data_sha256": {
            name: file_hash(args.data / name)
            for name in ("train.jsonl", "validation.jsonl", "groups.json")
        },
        "test_data_used": False,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "seed": 42,
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "train_metrics": training.metrics,
        "validation_metrics": evaluation,
        "adapter_sha256": adapter_hashes,
        "status": "adapter_trained_not_clinically_validated",
        "next": "Merge/quantize only after held-out baseline comparison and safety regression.",
    }
    (args.output / "RUN.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
