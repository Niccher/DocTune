"""
pipeline/prepare_data.py
------------------------
Turn a folder of private documents (PDF, DOCX, CSV) into a clean
instruction-tuning JSONL file that can be used by the training scripts.
"""

import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import polars as pl
    from pypdf import PdfReader
    from docx import Document
    from openai import OpenAI
except ImportError:
    pl = None
    PdfReader = None
    Document = None
    OpenAI = None

try:
    from pipeline.common import DATA_DIR, RAW_DOCS_DIR, DEFAULT_JSONL, CONFIG_FILE
except ImportError:
    from common import DATA_DIR, RAW_DOCS_DIR, DEFAULT_JSONL, CONFIG_FILE


def load_settings_defaults() -> dict:
    """Read defaults from config/settings.json if available."""
    defaults = {
        "teacher_url": "http://localhost:8000/v1",
        "teacher_model": "Qwen/Qwen2.5-72B-Instruct",
        "teacher_api_key": "not-needed",
        "max_chunks": 3,
        "chunk_size": 1000,
        "overlap": 150,
        "workers": 12,
    }
    try:
        cfg_path = Path(CONFIG_FILE)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dp = data.get("data_prep", {})
            qa = dp.get("qa_generator", {})
            if qa.get("base_url"):
                defaults["teacher_url"] = qa["base_url"]
            if qa.get("model"):
                defaults["teacher_model"] = qa["model"]
            if qa.get("api_key"):
                defaults["teacher_api_key"] = qa["api_key"]
            if dp.get("chunk_size"):
                defaults["chunk_size"] = dp["chunk_size"]
            if dp.get("chunk_overlap"):
                defaults["overlap"] = dp["chunk_overlap"]
    except Exception:
        pass
    return defaults


def resolve_teacher_url(url: str) -> str:
    """Map localhost/127.0.0.1 to host.docker.internal when executing inside Docker."""
    if os.path.exists("/.dockerenv") or os.environ.get("IS_DOCKER"):
        if "localhost" in url:
            return url.replace("localhost", "host.docker.internal")
        if "127.0.0.1" in url:
            return url.replace("127.0.0.1", "host.docker.internal")
    return url


# ============================================================
# DEFAULTS
# ============================================================
_INITIAL_DEFAULTS = load_settings_defaults()
DEFAULT_TEACHER_BASE_URL = _INITIAL_DEFAULTS["teacher_url"]
DEFAULT_TEACHER_MODEL = _INITIAL_DEFAULTS["teacher_model"]
DEFAULT_TEACHER_API_KEY = _INITIAL_DEFAULTS["teacher_api_key"]

DEFAULT_MAX_CHUNKS = _INITIAL_DEFAULTS["max_chunks"]
DEFAULT_CHUNK_SIZE = _INITIAL_DEFAULTS["chunk_size"]
DEFAULT_CHUNK_OVERLAP = _INITIAL_DEFAULTS["overlap"]
DEFAULT_MAX_WORKERS = _INITIAL_DEFAULTS["workers"]



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
            df = pl.read_csv(str(file_path))
            rows = [str(row) for row in df.iter_rows(named=True)]
            return "\n".join(rows)

    except Exception as e:
        print(f"[WARN] Failed to read {file_path.name}: {e}")

    return ""


# ============================================================
# CHUNKING
# ============================================================
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple overlapping word-based chunker.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 80:
            chunks.append(chunk)

    return chunks


def extract_json_object(content: str) -> dict:
    """Safely extract and parse JSON object even if wrapped in markdown code blocks or preamble."""
    if not content:
        return {}
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                pass
    return {}


def synthesize_qa(client: OpenAI, model: str, chunk: str, max_retries: int = 3) -> list[dict]:
    """
    Ask the teacher model to create one grounded question-answer pair.
    Includes exponential backoff retries for rate-limited cloud APIs (Gemini, Grok, OpenAI),
    fallback for engines without json_object schema support, and resilient JSON extraction.
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

    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.15,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
            except Exception as param_err:
                if "response_format" in str(param_err).lower():
                    # Fallback for llama.cpp / engines that reject response_format parameter
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": user_prompt}],
                        temperature=0.15,
                        max_tokens=512,
                    )
                else:
                    raise

            content = response.choices[0].message.content
            data = extract_json_object(content)

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
            err_str = str(e).lower()
            if attempt < max_retries - 1 and (
                "rate_limit" in err_str
                or "429" in err_str
                or "resource_exhausted" in err_str
                or "overloaded" in err_str
            ):
                sleep_sec = 2 ** (attempt + 1)
                print(f"[WARN] Teacher API throttled/rate-limited ({e}). Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
                continue
            print(f"[WARN] QA generation failed on attempt {attempt + 1}: {e}")
            return []


    return []


# ============================================================
# PER-FILE PROCESSING
# ============================================================
def process_one_file(file_path: Path, client: OpenAI, model: str,
                     output_file: Path, max_chunks: int,
                     chunk_size: int, overlap: int) -> int:
    text = extract_text(file_path)
    if not text:
        return 0

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)[:max_chunks]
    written = 0

    for chunk in chunks:
        records = synthesize_qa(client, model, chunk)
        if records:
            with open(output_file, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(records)

    return written


# ============================================================
# MAIN
# ============================================================
def run_preparation(docs_dir: Path = RAW_DOCS_DIR,
                    output_file: Path = DEFAULT_JSONL,
                    teacher_url: str = DEFAULT_TEACHER_BASE_URL,
                    teacher_model: str = DEFAULT_TEACHER_MODEL,
                    teacher_api_key: str = DEFAULT_TEACHER_API_KEY,
                    max_chunks: int = DEFAULT_MAX_CHUNKS,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    overlap: int = DEFAULT_CHUNK_OVERLAP,
                    max_workers: int = DEFAULT_MAX_WORKERS):
    if any(pkg is None for pkg in [pl, PdfReader, Document, OpenAI]):
        raise ImportError(
            "Data ingestion packages (polars, pypdf, python-docx, openai) are not installed. "
            "Please run: pip install -r requirements.txt"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        output_file.unlink()
        print(f"Removed previous {output_file}")

    patterns = ["*.pdf", "*.docx", "*.csv", "*.PDF", "*.DOCX", "*.CSV"]
    all_files = []
    for pattern in patterns:
        all_files.extend(docs_dir.glob(pattern))

    all_files = sorted(set(all_files))
    print(f"Found {len(all_files)} documents in {docs_dir}")

    if not all_files:
        print(f"No documents found. Put PDF / DOCX / CSV files into: {docs_dir}")
        return 0

    effective_url = resolve_teacher_url(teacher_url)
    effective_key = teacher_api_key.strip() if teacher_api_key and teacher_api_key.strip() else "not-needed"
    print(f"Connecting to Teacher LLM gateway at: {effective_url} (Model: {teacher_model})")
    client = OpenAI(base_url=effective_url, api_key=effective_key)



    total_pairs = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_file, f, client, teacher_model, output_file,
                max_chunks, chunk_size, overlap
            ): f for f in all_files
        }
        for future in as_completed(futures):
            pairs = future.result()
            total_pairs += pairs
            print(f"  +{pairs} pairs  (running total: {total_pairs})")

    print("\n" + "=" * 60)
    print(f"Finished. Wrote {total_pairs} instruction pairs → {output_file}")
    print("=" * 60)
    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare document instruction dataset.")
    parser.add_argument("--docs-dir", type=Path, default=RAW_DOCS_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--teacher-url", type=str, default=DEFAULT_TEACHER_BASE_URL)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default=DEFAULT_TEACHER_API_KEY)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)

    args = parser.parse_args()
    run_preparation(
        docs_dir=args.docs_dir,
        output_file=args.output_jsonl,
        teacher_url=args.teacher_url,
        teacher_model=args.teacher_model,
        teacher_api_key=args.teacher_api_key,
        max_chunks=args.max_chunks,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
