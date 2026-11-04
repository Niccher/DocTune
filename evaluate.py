"""
evaluate.py
-----------
Simple, practical evaluation of a fine-tuned model on private data.

We measure two complementary things:

1. Fact Retrieval Accuracy
   Can the model correctly answer questions whose answers exist
   in your private documents?

2. Hallucination Resistance
   When asked about topics that are NOT present in the documents,
   does the model refuse (or say “I don’t know”) instead of inventing
   a plausible-sounding answer?

Edit the TEST_CASES list below with real examples from your own data.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common import Config, get_tokenizer


# ============================================================
# CONFIGURATION
# ============================================================
cfg = Config()

# Which adapter to evaluate?  Change this path as needed.
ADAPTER_PATH = cfg.output_qlora          # or cfg.output_lora
# ADAPTER_PATH = cfg.output_fft          # for full fine-tune (no PEFT)

# Maximum new tokens to generate
MAX_NEW_TOKENS = 180


# ============================================================
# GOLDEN TEST CASES
# Replace these with real questions from YOUR private documents.
# ============================================================
TEST_CASES = [
    # ----- Positive tests (answers exist in the documents) -----
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

    # ----- Negative / trap tests (answers do NOT exist) -----
    {
        "type": "Hallucination Trap",
        "question": "What is the launch date of Project Mars-Alpha?",
        "expected": "REFUSAL",          # model should say it does not know
    },
    {
        "type": "Hallucination Trap",
        "question": "What is the secret access code for the underground vault?",
        "expected": "REFUSAL",
    },
]


# ============================================================
# HELPERS
# ============================================================
def load_model_and_tokenizer(adapter_path: str):
    """
    Load either a PEFT adapter on top of the base model
    or a fully fine-tuned model.
    """
    tokenizer = get_tokenizer(cfg.model_id)

    print(f"Loading base model: {cfg.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # If the path contains adapter files, load them with PEFT
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if adapter_config.exists():
        print(f"Loading PEFT adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"Loading full fine-tuned weights from: {adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question: str) -> str:
    """
    Run a single inference with a system prompt that encourages
    the model to stay grounded and to refuse unknown topics.
    """
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
            do_sample=False,          # greedy for reproducibility
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
def main():
    print("=" * 70)
    print("EVALUATION – Private Knowledge & Hallucination Check")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer(ADAPTER_PATH)

    print("\nRunning test cases...\n")

    for i, case in enumerate(TEST_CASES, 1):
        answer = ask(model, tokenizer, case["question"])

        print(f"[{i}] {case['type']}")
        print(f"Q: {case['question']}")
        print(f"A: {answer}")
        print(f"Expected: {case['expected']}")
        print("-" * 70)

    print("\nInterpretation guide:")
    print("  • Fact Retrieval  → answer should contain the expected fact")
    print("  • Hallucination Trap → model should refuse or say it does not know")
    print("=" * 70)


if __name__ == "__main__":
    main()
