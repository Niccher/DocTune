"""
pipeline/eval/harness.py
------------------------
Evaluation harness executing domain validation and 100-sample regression suites.
Uses llama.cpp (or active InferenceManager) for local execution under 3 minutes.
Computes multi-metric scores: EM, BLEU, ROUGE, refusal rate, and baseline accuracy delta.
"""

import re
import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable

from pipeline.eval.scorers import (
    exact_match_score,
    bleu_score,
    rouge_scores,
    refusal_score,
    evaluate_rubric,
)
from pipeline.eval.dataset_100 import BENCHMARK_100_SAMPLES
from web.services.inference.manager import inference_manager

logger = logging.getLogger("doctune.eval.harness")


def extract_choice_answer(raw_text: str) -> str:
    """Extract multiple-choice letter (A, B, C, D) from raw model output."""
    clean = raw_text.strip()
    # Match direct starting letter: "B", "B.", "Answer: B", "(B)"
    m = re.search(r"\b([A-D])\b", clean)
    if m:
        return m.group(1).upper()
    return clean[:1].upper() if clean else ""


class EvaluationHarness:
    """Orchestrates test suites against fine-tuned checkpoints and base models."""

    def __init__(self, concurrency: int = 4):
        self.concurrency = concurrency

    async def _query_model(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        """Helper to invoke active inference backend."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = await inference_manager.chat(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    async def run_regression_100(
        self,
        target_model: str,
        baseline_model: Optional[str] = "base",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the 100-sample balanced ARC & MMLU regression suite in < 3 minutes.
        Compares target model accuracy against baseline model accuracy.
        """
        start_time = time.time()
        total = len(BENCHMARK_100_SAMPLES)
        correct_target = 0
        correct_baseline = 0
        detailed_results = []

        system_instruction = (
            "You are a helpful and precise assistant. Answer the multiple-choice question. "
            "Output ONLY the single letter (A, B, C, or D) corresponding to the correct answer."
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_sample(item: Dict[str, Any], index: int):
            nonlocal correct_target, correct_baseline
            choices_text = "\n".join(f"{k}. {v}" for k, v in item["choices"].items())
            prompt = f"Question: {item['question']}\n\nChoices:\n{choices_text}\n\nAnswer:"

            async with semaphore:
                # 1. Query target model
                try:
                    resp_target = await self._query_model(prompt, target_model, system_instruction, max_tokens=16)
                except Exception as e:
                    resp_target = f"[Error: {str(e)}]"

                pred_target = extract_choice_answer(resp_target)
                is_correct_target = pred_target == item["answer"]
                if is_correct_target:
                    correct_target += 1

                # 2. Query baseline model if requested
                pred_baseline = None
                is_correct_baseline = None
                if baseline_model:
                    try:
                        resp_base = await self._query_model(prompt, baseline_model, system_instruction, max_tokens=16)
                        pred_baseline = extract_choice_answer(resp_base)
                        is_correct_baseline = pred_baseline == item["answer"]
                        if is_correct_baseline:
                            correct_baseline += 1
                    except Exception:
                        pass

                detail = {
                    "id": item["id"],
                    "benchmark": item["benchmark"],
                    "category": item["category"],
                    "question": item["question"],
                    "expected": item["answer"],
                    "target_pred": pred_target,
                    "target_correct": is_correct_target,
                    "baseline_pred": pred_baseline,
                    "baseline_correct": is_correct_baseline,
                }
                detailed_results.append(detail)

                if progress_callback:
                    completed = len(detailed_results)
                    progress_callback({
                        "completed": completed,
                        "total": total,
                        "target_accuracy": round(correct_target / completed, 4),
                        "baseline_accuracy": round(correct_baseline / completed, 4) if baseline_model else None,
                        "elapsed_s": round(time.time() - start_time, 1),
                    })

        tasks = [evaluate_sample(item, i) for i, item in enumerate(BENCHMARK_100_SAMPLES)]
        await asyncio.gather(*tasks)

        duration_s = round(time.time() - start_time, 2)
        target_acc = round(correct_target / total, 4)
        base_acc = round(correct_baseline / total, 4) if baseline_model else None
        delta = round(target_acc - base_acc, 4) if base_acc is not None else 0.0

        return {
            "suite": "regression_100",
            "total_samples": total,
            "target_model": target_model,
            "baseline_model": baseline_model,
            "target_accuracy": target_acc,
            "baseline_accuracy": base_acc,
            "accuracy_delta": delta,
            "regression_detected": delta < -0.05,  # Regression warning if >5% accuracy drop
            "duration_seconds": duration_s,
            "details": detailed_results,
        }

    async def run_domain_evaluation(
        self,
        target_model: str,
        test_cases: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on domain test cases (fact retrieval and hallucination traps).
        Computes BLEU, ROUGE-1/2/L, Exact Match, and Refusal rates.
        """
        start_time = time.time()
        total = len(test_cases)
        results = []

        total_bleu = 0.0
        total_r1 = 0.0
        total_r2 = 0.0
        total_rl = 0.0
        total_em = 0.0
        total_rubric = 0.0
        refusal_tests = 0
        refusals_passed = 0

        for i, tc in enumerate(test_cases, 1):
            question = tc.get("question") or tc.get("instruction") or ""
            expected = tc.get("expected") or tc.get("output") or tc.get("ground_truth") or ""
            is_trap = tc.get("type") == "Hallucination Trap" or expected == "REFUSAL"

            resp = await self._query_model(question, target_model, max_tokens=128)

            if is_trap:
                refusal_tests += 1
                passed = refusal_score(resp)
                if passed:
                    refusals_passed += 1
                item_metrics = {"refusal_passed": passed}
            else:
                em = exact_match_score(resp, expected)
                b = bleu_score(resp, expected)
                r = rouge_scores(resp, expected)
                rub = evaluate_rubric(resp, expected, question)

                total_em += em
                total_bleu += b
                total_r1 += r["rouge_1"]
                total_r2 += r["rouge_2"]
                total_rl += r["rouge_l"]
                total_rubric += rub["score"]

                item_metrics = {
                    "em": em,
                    "bleu": b,
                    "rouge": r,
                    "rubric_score": rub["score"],
                    "rationale": rub["rationale"],
                }

            results.append({
                "index": i,
                "question": question,
                "expected": expected,
                "prediction": resp,
                "is_trap": is_trap,
                "metrics": item_metrics,
            })

            if progress_callback:
                progress_callback({
                    "completed": i,
                    "total": total,
                    "elapsed_s": round(time.time() - start_time, 1),
                })

        non_trap_count = max(1, total - refusal_tests)
        avg_bleu = round(total_bleu / non_trap_count, 4)
        avg_r1 = round(total_r1 / non_trap_count, 4)
        avg_r2 = round(total_r2 / non_trap_count, 4)
        avg_rl = round(total_rl / non_trap_count, 4)
        avg_em = round(total_em / non_trap_count, 4)
        avg_rubric = round(total_rubric / non_trap_count, 2)
        refusal_acc = round(refusals_passed / max(1, refusal_tests), 4) if refusal_tests > 0 else 1.0

        return {
            "suite": "domain_eval",
            "total_cases": total,
            "target_model": target_model,
            "metrics": {
                "bleu": avg_bleu,
                "rouge_1": avg_r1,
                "rouge_2": avg_r2,
                "rouge_l": avg_rl,
                "exact_match": avg_em,
                "avg_rubric": avg_rubric,
                "refusal_accuracy": refusal_acc,
            },
            "duration_seconds": round(time.time() - start_time, 2),
            "details": results,
        }


evaluation_harness = EvaluationHarness()
