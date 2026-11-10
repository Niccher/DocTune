"""
pipeline/eval/__init__.py
-------------------------
Multi-tier evaluation subsystem:
- 100-sample balanced ARC & MMLU regression suite (<3 min execution)
- Domain factuality, hallucination refusal, and multi-metric scoring (BLEU, ROUGE, EM, Rubric)
"""

from pipeline.eval.scorers import exact_match_score, bleu_score, rouge_scores, refusal_score, evaluate_rubric
from pipeline.eval.dataset_100 import BENCHMARK_100_SAMPLES
from pipeline.eval.harness import evaluation_harness, EvaluationHarness

__all__ = [
    "exact_match_score",
    "bleu_score",
    "rouge_scores",
    "refusal_score",
    "evaluate_rubric",
    "BENCHMARK_100_SAMPLES",
    "EvaluationHarness",
    "evaluation_harness",
]
