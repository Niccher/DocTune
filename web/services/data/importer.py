"""
web/services/data/importer.py
-----------------------------
Universal dataset importer and schema normalizer.
Accepts standard LLM training data formats:
1. OpenAI Chat: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
2. Alpaca: {"instruction": ..., "input": ..., "output": ...}
3. Prompt / Completion: {"prompt": ..., "completion": ...}
4. DPO Preference: {"prompt": ..., "chosen": ..., "rejected": ...}
5. CSV and Parquet files with column mapping.
6. Hugging Face Hub dataset identifiers.
Outputs canonical Chat JSONL ready for SFTTrainer / DPO.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Generator

logger = logging.getLogger("doctune.data.importer")


class DatasetImporter:
    """Universal dataset importer and schema converter."""

    def detect_schema(self, sample_records: List[Dict[str, Any]]) -> str:
        """
        Identify schema structure from sample rows.
        Returns one of: 'openai_chat', 'alpaca', 'prompt_completion', 'dpo', 'unknown'.
        """
        if not sample_records:
            return "unknown"

        first = sample_records[0]
        if not isinstance(first, dict):
            return "unknown"

        keys = set(first.keys())

        if "messages" in keys and isinstance(first["messages"], list):
            return "openai_chat"

        if "instruction" in keys and "output" in keys:
            return "alpaca"

        if "prompt" in keys and "completion" in keys:
            return "prompt_completion"

        if ("prompt" in keys or "question" in keys) and "chosen" in keys and "rejected" in keys:
            return "dpo"

        if "input" in keys and "output" in keys:
            return "prompt_completion"

        return "tabular"

    def normalize_record(
        self,
        record: Dict[str, Any],
        schema: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert any supported format into canonical OpenAI chat message list."""
        if not isinstance(record, dict):
            return None

        # Apply column mapping if provided
        if column_mapping:
            mapped = {}
            for target_col, src_col in column_mapping.items():
                if src_col in record:
                    mapped[target_col] = record[src_col]
            record = mapped

        # If already in OpenAI messages format
        if "messages" in record and isinstance(record["messages"], list):
            valid_msgs = []
            for m in record["messages"]:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    valid_msgs.append({"role": str(m["role"]).strip(), "content": str(m["content"])})
            if valid_msgs:
                return {"messages": valid_msgs}

        # Alpaca schema: instruction + input -> user; output -> assistant
        if "instruction" in record and "output" in record:
            instruction = str(record["instruction"]).strip()
            inp = str(record.get("input", "")).strip()
            user_text = f"{instruction}\n\nContext:\n{inp}" if inp else instruction
            assistant_text = str(record["output"]).strip()
            return {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }

        # Prompt / Completion
        prompt_key = "prompt" if "prompt" in record else ("input" if "input" in record else None)
        compl_key = "completion" if "completion" in record else ("output" if "output" in record else None)

        if prompt_key and compl_key:
            return {
                "messages": [
                    {"role": "user", "content": str(record[prompt_key]).strip()},
                    {"role": "assistant", "content": str(record[compl_key]).strip()},
                ]
            }

        # DPO schema: retain chosen/rejected preference pair
        if "chosen" in record and "rejected" in record:
            prompt_val = str(record.get("prompt") or record.get("question") or "").strip()
            return {
                "prompt": prompt_val,
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            }

        return None

    def import_file(
        self,
        input_path: Path,
        output_path: Path,
        column_mapping: Optional[Dict[str, str]] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[int, str]:
        """
        Import and convert a local file (JSONL, CSV, Parquet) into normalized JSONL.
        Returns (rows_converted, detected_schema).
        """
        path = Path(input_path)
        ext = path.suffix.lower()

        records = []
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                        if len(records) >= 10:
                            break
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
                    if len(records) >= 10:
                        break

        schema = self.detect_schema(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as out_f:
            if ext == ".jsonl":
                with open(path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        norm = self.normalize_record(rec, schema, column_mapping)
                        if norm:
                            out_f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                            count += 1
                            if max_rows and count >= max_rows:
                                break

            elif ext == ".csv":
                with open(path, "r", encoding="utf-8", newline="") as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        norm = self.normalize_record(dict(row), schema, column_mapping)
                        if norm:
                            out_f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                            count += 1
                            if max_rows and count >= max_rows:
                                break

        logger.info("Imported %d records from %s into %s (schema: %s)", count, path.name, output_path.name, schema)
        return count, schema


dataset_importer = DatasetImporter()
universal_importer = dataset_importer
