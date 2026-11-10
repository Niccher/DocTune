"""
web/services/data/validator.py
------------------------------
Comprehensive dataset validation engine for LLM training datasets.
Computes:
- Total, valid, empty, and duplicate sample counts
- Message role alternation correctness (e.g. user -> assistant)
- Token length distribution (min, max, mean, median, p95, p99)
- Truncation risk at common context windows (512, 1024, 2048, 4096, 8192)
- Structured issue logs and actionable diagnostic recommendations
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import statistics

logger = logging.getLogger("doctune.data.validator")

ValidationTuple = Tuple[bool, str, int]


class DatasetValidator:
    """Validates SFT/DPO datasets and calculates token distributions and truncation risks."""

    def __init__(self, tokenizer=None, tokenizer_name: Optional[str] = None):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        if self.tokenizer is None and tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = None
        self._init_pii_redactor()

    def _init_pii_redactor(self):
        try:
            from pipeline.pii_redactor import PIIRedactor
            self.pii_redactor = PIIRedactor()
        except Exception:
            self.pii_redactor = None

    @property
    def tokenizer_description(self) -> str:
        if self.tokenizer_name:
            return f"pretrained ({self.tokenizer_name})"
        if self.tokenizer is not None:
            return f"custom ({type(self.tokenizer).__name__})"
        return "heuristic (~4 chars/token)"

    def count_tokens(self, text: str) -> int:
        """Estimate or count tokens for text."""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Reliable heuristic fallback: ~4 characters per token for English/code
        return max(1, len(text) // 4)

    def validate_messages(self, messages: List[Dict[str, Any]]) -> ValidationTuple:
        """
        Validate OpenAI chat message structure and role alternation.
        Returns (is_valid, error_reason, total_chars)
        """
        if not messages or not isinstance(messages, list):
            return False, "Messages list is empty or invalid", 0

        valid_roles = {"system", "user", "assistant"}
        expected_next = {"user", "system"}
        has_user = False
        has_assistant = False
        total_chars = 0

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message #{i} is not a valid JSON object", total_chars

            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "")

            if role not in valid_roles:
                return False, f"Message #{i} has unsupported role: '{role}'", total_chars

            if not isinstance(content, str) or not content.strip():
                return False, f"Message #{i} ({role}) has empty content", total_chars

            total_chars += len(content)

            if i == 0:
                if role == "system":
                    expected_next = {"user"}
                elif role == "user":
                    has_user = True
                    expected_next = {"assistant"}
                else:
                    return False, "First message must be 'system' or 'user'", total_chars
            else:
                if role not in expected_next:
                    return (
                        False,
                        f"Unexpected role '{role}' at message #{i}; expected one of {expected_next}",
                        total_chars,
                    )

                if role == "user":
                    has_user = True
                    expected_next = {"assistant"}
                elif role == "assistant":
                    has_assistant = True
                    expected_next = {"user"}

        if not has_user:
            return False, "Conversation missing 'user' message", total_chars
        if not has_assistant:
            return False, "Conversation missing 'assistant' message", total_chars

        return True, "", total_chars

    def validate_file(
        self,
        file_path: Union[str, Path],
        max_preview_samples: int = 5,
        target_seq_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Scan a JSONL file and produce a comprehensive validation report.
        """
        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "error": f"File not found: {file_path}"}

        if target_seq_lengths is None:
            target_seq_lengths = [512, 1024, 2048, 4096, 8192]

        total_samples = 0
        valid_samples = 0
        empty_samples = 0
        role_errors = 0
        seen_hashes = set()
        duplicate_count = 0

        token_lengths: List[int] = []
        issues: List[str] = []
        previews: List[Dict[str, Any]] = []
        pii_hits: Dict[str, int] = {}

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    empty_samples += 1
                    continue

                total_samples += 1

                # Exact duplicate detection via line hash
                line_hash = hashlib.sha256(clean_line.encode("utf-8")).hexdigest()
                if line_hash in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(line_hash)

                try:
                    record = json.loads(clean_line)
                except Exception as e:
                    if len(issues) < 20:
                        issues.append(f"Line {line_idx}: Invalid JSON syntax ({str(e)})")
                    continue

                # Support canonical OpenAI chat format or DPO pairs
                if "messages" in record:
                    is_val, err, chars = self.validate_messages(record["messages"])
                    if not is_val:
                        role_errors += 1
                        if len(issues) < 20:
                            issues.append(f"Line {line_idx}: {err}")
                    else:
                        valid_samples += 1
                        # Estimate total sample tokens
                        tokens = sum(self.count_tokens(m.get("content", "")) for m in record["messages"])
                        token_lengths.append(tokens)

                        # Scan for PII
                        if self.pii_redactor:
                            for m in record["messages"]:
                                content_str = m.get("content", "")
                                if content_str and isinstance(content_str, str):
                                    for hit in self.pii_redactor.analyze(content_str):
                                        etype = hit.get("entity", "UNKNOWN")
                                        pii_hits[etype] = pii_hits.get(etype, 0) + 1

                        if len(previews) < max_preview_samples:
                            previews.append({
                                "line": line_idx,
                                "type": "chat",
                                "message_count": len(record["messages"]),
                                "tokens_est": tokens,
                                "preview": record["messages"][:2],
                            })

                elif "chosen" in record and "rejected" in record:
                    # DPO pair validation
                    prompt_txt = str(record.get("prompt", "") or record.get("question", ""))
                    chosen_txt = str(record.get("chosen", ""))
                    rejected_txt = str(record.get("rejected", ""))

                    if not prompt_txt.strip() or not chosen_txt.strip() or not rejected_txt.strip():
                        empty_samples += 1
                        if len(issues) < 20:
                            issues.append(f"Line {line_idx}: DPO sample missing prompt, chosen, or rejected text")
                    else:
                        valid_samples += 1
                        tokens = self.count_tokens(prompt_txt) + max(
                            self.count_tokens(chosen_txt), self.count_tokens(rejected_txt)
                        )
                        token_lengths.append(tokens)

                        # Scan for PII
                        if self.pii_redactor:
                            for text_field in (prompt_txt, chosen_txt, rejected_txt):
                                if text_field:
                                    for hit in self.pii_redactor.analyze(text_field):
                                        etype = hit.get("entity", "UNKNOWN")
                                        pii_hits[etype] = pii_hits.get(etype, 0) + 1

                        if len(previews) < max_preview_samples:
                            previews.append({
                                "line": line_idx,
                                "type": "dpo",
                                "tokens_est": tokens,
                                "prompt": prompt_txt[:120],
                            })
                else:
                    if len(issues) < 20:
                        issues.append(f"Line {line_idx}: Unknown format (missing 'messages' or 'chosen'/'rejected')")

        # Distribution statistics
        stats = {
            "min_tokens": 0,
            "max_tokens": 0,
            "mean_tokens": 0,
            "median_tokens": 0,
            "p95_tokens": 0,
            "p99_tokens": 0,
        }

        if token_lengths:
            token_lengths.sort()
            stats["min_tokens"] = token_lengths[0]
            stats["max_tokens"] = token_lengths[-1]
            stats["mean_tokens"] = round(statistics.mean(token_lengths), 1)
            stats["median_tokens"] = round(statistics.median(token_lengths), 1)
            p95_idx = min(int(len(token_lengths) * 0.95), len(token_lengths) - 1)
            p99_idx = min(int(len(token_lengths) * 0.99), len(token_lengths) - 1)
            stats["p95_tokens"] = token_lengths[p95_idx]
            stats["p99_tokens"] = token_lengths[p99_idx]

        # Truncation risks
        truncation_risks = {}
        for seq_len in target_seq_lengths:
            exceeding = sum(1 for t in token_lengths if t > seq_len)
            pct = round((exceeding / total_samples * 100), 2) if total_samples > 0 else 0.0
            truncation_risks[str(seq_len)] = {
                "count_exceeding": exceeding,
                "percent_exceeding": pct,
                "risk_level": "HIGH" if pct > 15.0 else ("MEDIUM" if pct > 5.0 else "LOW"),
            }

        # Recommendations
        recommendations = []
        if duplicate_count > 0:
            recommendations.append(
                f"Found {duplicate_count} duplicate samples ({round(duplicate_count/max(1, total_samples)*100, 1)}%). Deduping is recommended to avoid memorization."
            )
        if role_errors > 0:
            recommendations.append(
                f"Found {role_errors} role alternation errors. Standard SFT templates require user -> assistant sequencing."
            )
        if pii_hits:
            total_pii = sum(pii_hits.values())
            breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(pii_hits.items()))
            recommendations.append(
                f"Detected {total_pii} potential PII occurrences ({breakdown}). Run PII redaction/anonymization before training."
            )
        if truncation_risks.get("2048", {}).get("risk_level") == "HIGH":
            recommendations.append(
                f"Over {truncation_risks['2048']['percent_exceeding']}% of samples exceed 2048 tokens. Consider increasing max_seq_length or filtering long samples."
            )
        if valid_samples < 50:
            recommendations.append(
                "Dataset contains fewer than 50 valid samples. Model fine-tuning typically benefits from at least 100-500 high-quality samples."
            )

        is_valid = total_samples > 0 and valid_samples >= (total_samples * 0.8) and role_errors == 0

        return {
            "valid": is_valid,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "empty_samples": empty_samples,
            "duplicate_count": duplicate_count,
            "role_errors": role_errors,
            "stats": stats,
            "truncation_risks": truncation_risks,
            "tokenizer_used": self.tokenizer_description,
            "pii_hits": pii_hits,
            "issues": issues,
            "recommendations": recommendations,
            "previews": previews,
        }


dataset_validator = DatasetValidator()
