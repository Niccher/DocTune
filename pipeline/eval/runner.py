"""
pipeline/eval/runner.py
-----------------------
Command-line runner and background process entrypoint for evaluation jobs.
Supports:
--suite regression_100 | domain
--target-model <id_or_path>
--baseline-model base | none
--output-file <path>
--save-db
"""

import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from pipeline.eval.harness import evaluation_harness
from web.db.session import SessionLocal
from web.db.models import EvaluationBenchmark, Checkpoint


async def run_evaluation_suite_async(
    suite: str = "regression_100",
    target_model: str = "default",
    baseline_model: Optional[str] = "base",
    test_cases: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute evaluation suite asynchronously (regression_100 or domain)."""
    if suite == "regression_100":
        return await evaluation_harness.run_regression_100(
            target_model=target_model,
            baseline_model=baseline_model if baseline_model != "none" else None,
            progress_callback=progress_callback,
        )
    else:
        cases = test_cases
        if not cases:
            from pipeline.evaluate import TEST_CASES
            cases = TEST_CASES
        return await evaluation_harness.run_domain_evaluation(
            target_model=target_model,
            test_cases=cases,
            progress_callback=progress_callback,
        )


def run_evaluation_suite(
    suite: str = "regression_100",
    target_model: str = "default",
    baseline_model: Optional[str] = "base",
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Synchronous entrypoint for running evaluation suites."""
    return asyncio.run(
        run_evaluation_suite_async(
            suite=suite,
            target_model=target_model,
            baseline_model=baseline_model,
            test_cases=test_cases,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="DocTune Model Evaluation Runner")
    parser.add_argument("--suite", type=str, default="regression_100", choices=["regression_100", "domain"])
    parser.add_argument("--target-model", type=str, default="default", help="Model ID, checkpoint ID, or path")
    parser.add_argument("--baseline-model", type=str, default="base", help="Baseline model for comparison")
    parser.add_argument("--test-cases-file", type=Path, default=None, help="JSON file with domain test cases")
    parser.add_argument("--output-file", type=Path, default=None, help="Where to save eval results JSON")
    parser.add_argument("--save-db", action="store_true", help="Persist evaluation results to PostgreSQL/SQLite")
    parser.add_argument("--checkpoint-id", type=str, default=None, help="Associated checkpoint ID")
    parser.add_argument("--run-id", type=str, default=None, help="Associated training run ID")
    args = parser.parse_args()

    print(f"Starting DocTune Evaluation: suite={args.suite}, target={args.target_model}", flush=True)

    def on_progress(p):
        print(f"[EVAL_PROGRESS] {json.dumps(p)}", flush=True)

    async def execute():
        if args.suite == "regression_100":
            return await evaluation_harness.run_regression_100(
                target_model=args.target_model,
                baseline_model=args.baseline_model if args.baseline_model != "none" else None,
                progress_callback=on_progress,
            )
        else:
            cases = []
            if args.test_cases_file and args.test_cases_file.exists():
                with open(args.test_cases_file, "r", encoding="utf-8") as f:
                    cases = json.load(f)
            else:
                # Default domain test cases
                from pipeline.evaluate import TEST_CASES
                cases = TEST_CASES

            return await evaluation_harness.run_domain_evaluation(
                target_model=args.target_model,
                test_cases=cases,
                progress_callback=on_progress,
            )

    result = asyncio.run(execute())

    print("\n" + "=" * 60, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print(f"Suite: {args.suite}", flush=True)
    if args.suite == "regression_100":
        print(f"Target Accuracy: {result['target_accuracy']*100:.1f}%", flush=True)
        if result.get("baseline_accuracy") is not None:
            print(f"Baseline Accuracy: {result['baseline_accuracy']*100:.1f}%", flush=True)
            print(f"Delta: {result['accuracy_delta']*100:+.1f}%", flush=True)
            if result.get("regression_detected"):
                print("WARNING: Regression detected! (>5% drop in benchmark performance)", flush=True)
    else:
        m = result.get("metrics", {})
        print(f"Exact Match: {m.get('exact_match', 0)*100:.1f}%", flush=True)
        print(f"BLEU: {m.get('bleu', 0):.4f}", flush=True)
        print(f"ROUGE-L: {m.get('rouge_l', 0):.4f}", flush=True)
        print(f"Refusal Accuracy: {m.get('refusal_accuracy', 0)*100:.1f}%", flush=True)

    print(f"Duration: {result.get('duration_seconds', 0)}s", flush=True)
    print("=" * 60, flush=True)

    print(f"\n[EVAL_JSON] {json.dumps(result)}", flush=True)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.save_db and SessionLocal:
        try:
            db = SessionLocal()
            bench = EvaluationBenchmark(
                id=f"eval_{int(time.time()*1000)}",
                checkpoint_id=args.checkpoint_id,
                run_id=args.run_id,
                test_suite_name=args.suite,
                bleu_score=result.get("metrics", {}).get("bleu") or result.get("target_accuracy"),
                rouge_l=result.get("metrics", {}).get("rouge_l"),
                detailed_results=result,
            )
            db.add(bench)
            db.commit()
            print("Evaluation results persisted to database.", flush=True)
            db.close()
        except Exception as e:
            print(f"Database persist notice: {e}", flush=True)


if __name__ == "__main__":
    main()
