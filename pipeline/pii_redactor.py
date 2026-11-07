"""
pipeline/pii_redactor.py
-------------------------
Enterprise PII Redaction and Anonymization Module.
Provides dual-engine protection:
  1. Microsoft Presidio Analyzer + Anonymizer (when installed)
  2. High-speed compiled regex & pattern fallback (zero external dependencies)
Protects private documents from leaking names, emails, phone numbers,
government IDs, financial data, and credentials during synthetic QA and LLM training.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional, Any

# Try importing Microsoft Presidio
HAS_PRESIDIO = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    HAS_PRESIDIO = True
except Exception:
    _analyzer = None
    _anonymizer = None
    HAS_PRESIDIO = False


# Supported default entities
ALL_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "API_KEY",
]


# Comprehensive Regex Patterns for fallback & credential scrubbing
REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
    ),
    "US_SSN": re.compile(
        r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11})\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "API_KEY": re.compile(
        r"(?i)\b(?:sk-[a-zA-Z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[0-9a-zA-Z]{10,48}|ey[A-Za-z0-9_-]{15,}\.ey[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,})\b"
    ),
    "PERSON": re.compile(
        r"(?i)(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    ),
}


class PIIRedactor:
    """
    Sanitizes enterprise documents to prevent memorization of PII during fine-tuning.
    """

    def __init__(self, entities: Optional[List[str]] = None, mode: str = "placeholder"):
        self.entities = set(entities or ALL_ENTITIES)
        self.mode = mode.lower()  # placeholder, mask, or synthetic_hash
        self.has_presidio = HAS_PRESIDIO

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Identify sensitive PII locations, values, and entity types."""
        results: List[Dict[str, Any]] = []

        # 1. Run Presidio if available
        if self.has_presidio and _analyzer:
            try:
                # Map entity names to Presidio types
                presidio_entities = [e for e in self.entities if e in [
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"
                ]]
                p_results = _analyzer.analyze(text=text, entities=presidio_entities, language="en")
                for r in p_results:
                    results.append({
                        "entity": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score,
                        "text": text[r.start:r.end],
                    })
            except Exception:
                pass

        # 2. Run Regex fallback / supplement (especially for API_KEY and when Presidio is absent or missed)
        for entity_type, pattern in REGEX_PATTERNS.items():
            if entity_type not in self.entities:
                continue
            # If presidio already ran for standard types with results, only supplement API_KEY or missing items
            if self.has_presidio and entity_type != "API_KEY" and results:
                continue

            for match in pattern.finditer(text):
                m_start, m_end = match.start(), match.end()
                matched_text = match.group(0).strip()
                if not matched_text:
                    continue

                # Avoid duplicate / overlapping spans
                if any(r["start"] <= m_start < r["end"] or r["start"] < m_end <= r["end"] for r in results):
                    continue

                results.append({
                    "entity": entity_type,
                    "start": m_start,
                    "end": m_end,
                    "score": 0.95,
                    "text": matched_text,
                })

        # Sort matches by starting offset
        results.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        # Guarantee strictly non-overlapping spans
        non_overlapping: List[Dict[str, Any]] = []
        last_end = -1
        for m in results:
            if m["start"] >= last_end:
                non_overlapping.append(m)
                last_end = m["end"]

        return non_overlapping

    def _format_replacement(self, entity_type: str, raw_val: str) -> str:
        """Format the anonymized token based on the selected mode."""
        if self.mode == "mask":
            if "@" in raw_val:
                parts = raw_val.split("@", 1)
                user = parts[0]
                masked_user = (user[0] + "***") if len(user) > 1 else "***"
                return f"{masked_user}@{parts[1]}"
            if len(raw_val) > 4:
                return raw_val[:2] + ("*" * (len(raw_val) - 4)) + raw_val[-2:]
            return "*" * len(raw_val)

        elif self.mode == "synthetic_hash":
            h = hashlib.sha256(raw_val.encode("utf-8")).hexdigest()[:6]
            return f"[{entity_type}_{h}]"

        # Default: placeholder
        return f"[{entity_type}]"

    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Scrub PII from text.
        Returns:
            Tuple of (anonymized_text, entity_counts_dict)
        """
        if not text:
            return "", {}

        matches = self.analyze(text)
        if not matches:
            return text, {}

        counts: Dict[str, int] = {}
        # Reconstruct string in reverse order to preserve char offsets
        pieces = []
        last_idx = len(text)

        for match in reversed(matches):
            ent = match["entity"]
            start = match["start"]
            end = match["end"]
            raw_val = match["text"]

            replacement = self._format_replacement(ent, raw_val)
            pieces.append(text[end:last_idx])
            pieces.append(replacement)
            last_idx = start
            counts[ent] = counts.get(ent, 0) + 1

        pieces.append(text[:last_idx])
        pieces.reverse()
        anonymized_text = "".join(pieces)

        return anonymized_text, counts


# Global singleton instance for quick use
pii_redactor = PIIRedactor()
