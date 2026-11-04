"""
prepare_data.py
---------------
Turn a folder of private documents (PDF, DOCX, CSV) into a clean
instruction-tuning JSONL file that can be used by the training scripts.

Pipeline:
  1. Walk the documents folder
  2. Extract text according to file type
  3. Split text into overlapping chunks
  4. Ask a strong "teacher" LLM to generate grounded Q&A pairs
  5. Write each pair as one line of JSONL

Output format (one object per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer grounded in the document"}
  ]
}
"""

import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from pypdf import PdfReader
from docx import Document
from openai import OpenAI

from common import DATA_DIR, DEFAULT_JSONL


# ============================================================
# CONFIGURATION – edit these values
# ============================================================
DOCS_DIR = Path("./raw_documents")          # Folder that contains your PDFs / DOCX / CSVs
OUTPUT_JSONL = DEFAULT_JSONL                # Final training file (data/private_docs.jsonl)

# Teacher model (local vLLM, Ollama, LM Studio, or real OpenAI-compatible endpoint)
TEACHER_BASE_URL = "http://localhost:8000/v1"
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Change to whatever strong model you run
TEACHER_API_KEY = "not-needed"              # Only needed for real OpenAI / paid endpoints

MAX_CHUNKS_PER_DOC = 3                      # Keep total pairs reasonable (~20-30k)
CHUNK_SIZE_WORDS = 1000
CHUNK_OVERLAP_WORDS = 150
MAX_WORKERS = 12                            # Parallel file processing


# ============================================================
# CLIENT
# ============================================================
client = OpenAI(base_url=TEACHER_BASE_URL, api_key=TEACHER_API_KEY)


# ============================================================
# TEXT EXTRACTION
# ============================================================
def extract_text(file_path: Path) -> str:
    """
    Extract clean plain text from PDF, DOCX or CSV.
    Returns empty string on failure so the pipeline can continue.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        elif ext == ".docx":
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)

        elif ext == ".csv":
            # Convert each row into a readable key-value style string
            df = pl.read_csv(str(file_path))
            rows = [str(row) for row in df.iter_rows(named=True)]
            return "\n".join(rows)

    except Exception as e:
        print(f"[WARN] Failed to read {file_path.name}: {e}")

    return ""


# ============================================================
# CHUNKING
# ============================================================
def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Simple overlapping word-based chunker.
    Returns only chunks that contain meaningful content.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 80:                     # Skip near-empty fragments
            chunks.append(chunk)

    return chunks


# ============================================================
# SYNTHETIC Q&A GENERATION
# ============================================================
def synthesize_qa(chunk: str) -> list[dict]:
    """
    Ask the teacher model to create one high-quality, grounded
    question-answer pair from the given text chunk.

    Returns a list containing zero or one record in the
    messages format expected by SFTTrainer / TRL.
    """
    system_prompt = (
        "You are an internal company assistant. "
        "Answer questions using ONLY information present in the provided document excerpt. "
        "Be precise, factual and concise."
    )

    user_prompt = f"""
Given the document excerpt below, generate exactly ONE realistic user question
and a direct, accurate answer that can be found in the text.

Return STRICT JSON with the keys "question" and "answer".

Document excerpt:
\"\"\"{chunk}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.15,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        content = response.choices[0].message.content
        data = json.loads(content)

        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            return []

        return [{
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }]

    except Exception as e:
        print(f"[WARN] QA generation failed: {e}")
        return []


# ============================================================
# PER-FILE PROCESSING
# ============================================================
def process_one_file(file_path: Path) -> int:
    """
    Full pipeline for a single document:
    extract → chunk → generate Q&A → append to JSONL.
    Returns the number of pairs written.
    """
    text = extract_text(file_path)
    if not text:
        return 0

    chunks = chunk_text(text)[:MAX_CHUNKS_PER_DOC]
    written = 0

    for chunk in chunks:
        records = synthesize_qa(chunk)
        if records:
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(records)

    return written


# ============================================================
# MAIN
# ============================================================
def main():
    # Make sure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start fresh
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
        print(f"Removed previous {OUTPUT_JSONL}")

    # Collect files
    patterns = ["*.pdf", "*.docx", "*.csv", "*.PDF", "*.DOCX", "*.CSV"]
    all_files = []
    for pattern in patterns:
        all_files.extend(DOCS_DIR.glob(pattern))

    all_files = sorted(set(all_files))
    print(f"Found {len(all_files)} documents in {DOCS_DIR}")

    if not all_files:
        print("No documents found. Put PDF / DOCX / CSV files into the raw_documents folder.")
        return

    total_pairs = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, f): f for f in all_files}
        for future in as_completed(futures):
            pairs = future.result()
            total_pairs += pairs
            print(f"  +{pairs} pairs  (running total: {total_pairs})")

    print("\n" + "=" * 60)
    print(f"Finished. Wrote {total_pairs} instruction pairs → {OUTPUT_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
