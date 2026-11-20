"""Compare frozen validation targets using the base model and the trained adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecocompanion.pet.evidence import QwenAssessment


class Row(BaseModel):
    messages: list[dict[str, str]]
    example_id: str
    group_id: str


def decode(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    results = []
    split_path = args.data / f"{args.split}.jsonl"
    for line in split_path.read_text(encoding="utf-8").splitlines():
        row = Row.model_validate_json(line)
        prompt = tokenizer.apply_chat_template(
            row.messages[:-1], tokenize=False, add_generation_prompt=True
        )
        record = {"example_id": row.example_id, "group_id": row.group_id}
        raw = ""
        try:
            raw = decode(model, tokenizer, prompt)
            actual = QwenAssessment.model_validate_json(raw)
            expected = QwenAssessment.model_validate_json(row.messages[-1]["content"])
            current_message = json.loads(row.messages[-2]["content"])["current_message"]
            grounded = all(
                quote and quote in current_message for quote in actual.evidence_quotes
            )
            record.update(
                valid=True,
                grounded=grounded,
                exact_schema_match=actual == expected,
                safety_match=actual.safety_signal == expected.safety_signal,
                actual=actual.model_dump(),
            )
        except (ValueError, ValidationError) as exc:
            record.update(valid=False, error=type(exc).__name__, raw=raw[:4000])
        results.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "ecocompanion-adapter-development-validation-v1",
                "scope": f"small synthetic {args.split} engineering fixture; not clinical validation",
                "split": args.split,
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "adapter": str(args.adapter) if args.adapter else None,
                "total": len(results),
                "valid": sum(item["valid"] for item in results),
                "grounded": sum(item.get("grounded", False) for item in results),
                "safety_match": sum(item.get("safety_match", False) for item in results),
                "exact_schema_match": sum(
                    item.get("exact_schema_match", False) for item in results
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
