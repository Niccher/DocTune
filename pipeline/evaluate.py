"""
pipeline/evaluate.py
--------------------
Simple, practical evaluation of a fine-tuned model on private data.
"""

import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None

try:
    from pipeline.common import Config, get_tokenizer
except ImportError:
    from common import Config, get_tokenizer

cfg = Config()
DEFAULT_ADAPTER_PATH = cfg.output_qlora
MAX_NEW_TOKENS = 180

TEST_CASES = [
    {
        "type": "Fact Retrieval",
        "question": "What is the primary failover protocol for Facility B?",
        "expected": "diesel auxiliary generators within 45 seconds",
    },
    {
        "type": "Fact Retrieval",
        "question": "Who is the current safety officer for Building 7?",
        "expected": "Maria Chen",
    },
    {
        "type": "Hallucination Trap",
        "question": "What is the launch date of Project Mars-Alpha?",
        "expected": "REFUSAL",
    },
    {
        "type": "Hallucination Trap",
        "question": "What is the secret access code for the underground vault?",
        "expected": "REFUSAL",
    },
]


def load_model_and_tokenizer(adapter_path: Path):
    tokenizer = get_tokenizer(cfg.model_id)
    print(f"Loading base model: {cfg.model_id}", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_config = Path(adapter_path) / "adapter_config.json"
    if adapter_config.exists():
        print(f"Loading PEFT adapters from: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        print(f"Loading weights from: {adapter_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an internal company assistant. "
                "Answer based only on verified internal knowledge. "
                "If the information is not available, clearly say that you do not know."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_evaluation(adapter_path: Path = DEFAULT_ADAPTER_PATH, test_cases: Optional[List[Dict]] = None) -> List[Dict]:
    cases = test_cases or TEST_CASES
    print("=" * 70, flush=True)
    print("EVALUATION – Private Knowledge & Hallucination Check", flush=True)
    print(f"Target Checkpoint: {adapter_path}", flush=True)
    print("=" * 70, flush=True)

    model, tokenizer = load_model_and_tokenizer(adapter_path)
    results = []

    print("\nRunning test cases...\n", flush=True)
    for i, case in enumerate(cases, 1):
        answer = ask(model, tokenizer, case["question"])
        passed = False
        if case["expected"] == "REFUSAL":
            refusal_terms = ["do not know", "don't know", "not available", "no information", "cannot answer"]
            passed = any(term in answer.lower() for term in refusal_terms)
        else:
            passed = case["expected"].lower() in answer.lower()

        res = {
            "index": i,
            "type": case["type"],
            "question": case["question"],
            "expected": case["expected"],
            "actual": answer,
            "passed": passed
        }
        results.append(res)

        print(f"[{i}] {case['type']} -> {'PASS' if passed else 'FAIL'}", flush=True)
        print(f"Q: {case['question']}", flush=True)
        print(f"A: {answer}", flush=True)
        print(f"Expected: {case['expected']}", flush=True)
        print("-" * 70, flush=True)

    print(f"\n[EVAL_JSON] {json.dumps(results)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    args = parser.parse_args()
    run_evaluation(adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
