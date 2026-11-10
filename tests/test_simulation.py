import sys
import json
import uuid
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from web.core.ids import generate_ulid, generate_run_id, generate_job_id
from web.schemas.job_spec import JobSpec, JobType, JobState
from web.services.data.importer import dataset_importer
from web.services.data.validator import dataset_validator
from pipeline.hardware import detect_hardware
from pipeline.estimator import estimate_vram_and_feasibility
from pipeline.errors import classify_training_error
from pipeline.eval.scorers import exact_match_score, bleu_score, rouge_scores, refusal_score, evaluate_rubric
from pipeline.eval.dataset_100 import BENCHMARK_100_SAMPLES, get_100_eval_samples
from web.db.session import SessionLocal, init_db
from web.db.models import User, Project, ProjectMember, UserSession, ApiKey, Job, Run, MetricPoint
from web.services.auth_service import auth_service
from web.services.queue_manager import queue_manager

def run_simulation():
    print("=== STARTING REALISTIC END-TO-END SIMULATION ===")
    init_db()

    # 1. REALISTIC DATA INGESTION & VALIDATION
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_alpaca = tmp_path / "alpaca.jsonl"
        with open(raw_alpaca, "w", encoding="utf-8") as f:
            for i in range(75):
                f.write(json.dumps({
                    "instruction": f"Explain the architectural trade-offs of microservices vs monolith for workload #{i}.",
                    "input": "Workload involves financial transaction processing.",
                    "output": f"For financial transactions #{i}, a modular monolith provides ACID transactions without distributed 2PC overhead, while microservices isolate blast radiuses."
                }) + "\n")
        
        canonical_chat = tmp_path / "canonical.jsonl"
        count, schema = dataset_importer.import_file(raw_alpaca, canonical_chat)
        assert count == 75, f"Expected 75 records, got {count}"
        assert schema == "alpaca", f"Expected alpaca schema, got {schema}"
        print(f"[OK] Ingestion: Imported {count} records from Alpaca format to canonical chat format.")

        # Validate the generated canonical chat file
        val_res = dataset_validator.validate_file(canonical_chat)
        assert val_res["valid"] is True, f"Validation failed: {val_res}"
        assert val_res["total_samples"] == 75
        assert val_res["duplicate_count"] == 0
        assert val_res["role_errors"] == 0
        stats = val_res["stats"]
        assert stats["min_tokens"] > 0
        assert stats["mean_tokens"] >= stats["min_tokens"]
        print(f"[OK] Validation: Token percentiles min={stats['min_tokens']}, median={stats['median_tokens']}, p95={stats['p95_tokens']}, p99={stats['p99_tokens']}")
        trunc_risks = val_res["truncation_risks"]
        print(f"[OK] Truncation risk analysis: {trunc_risks}")

    # 2. HARDWARE PROFILING & VRAM ESTIMATION
    hw = detect_hardware()
    print(f"[OK] Hardware profile: Device={hw.device_type}, RAM={hw.total_ram_gb:.1f}GB, GPUs={hw.gpu_count}, CUDA={hw.cuda_available}")

    # Estimate 8B QLoRA vs LoRA
    vram_qlora = estimate_vram_and_feasibility(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        method="QLORA",
        per_device_batch_size=2,
        max_seq_length=2048,
        lora_r=16,
        gradient_checkpointing=True,
        total_vram_gb=16.0,
    )
    print(f"[OK] VRAM Estimation (8B QLoRA on 16GB VRAM): weights={vram_qlora['breakdown']['weights_gb']}GB, total={vram_qlora['breakdown']['total_estimated_gb']}GB, feasible={vram_qlora['feasible']}")
    assert vram_qlora["breakdown"]["weights_gb"] < 6.0, "4-bit 8B weights should be ~5GB"

    vram_lora = estimate_vram_and_feasibility(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        method="LORA",
        per_device_batch_size=2,
        max_seq_length=2048,
        lora_r=16,
        gradient_checkpointing=True,
        total_vram_gb=8.0,
    )
    print(f"[OK] VRAM Estimation (8B LoRA on 8GB VRAM): feasible={vram_lora['feasible']} (expected False for 16-bit 8B on 8GB VRAM)")
    assert vram_lora["feasible"] is False, "16-bit 8B model should not be feasible on 8GB VRAM"

    # 3. ERROR DIAGNOSTICS TAXONOMY
    oom_err = RuntimeError("CUDA out of memory. Tried to allocate 3.12 GiB (GPU 0; 15.78 GiB total capacity; 14.20 GiB already allocated)")
    diag_oom = classify_training_error(oom_err, {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"})
    assert diag_oom["code"] == "CUDA_OOM"
    assert "3.12 GiB" in diag_oom["message"]
    print(f"[OK] Diagnostic OOM classification: {diag_oom['title']} -> {diag_oom['hint'][:70]}...")

    gated_err = RuntimeError("401 Client Error: Repository meta-llama/Meta-Llama-3-8B-Instruct is gated. Must accept terms.")
    diag_gated = classify_training_error(gated_err, {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"})
    assert diag_gated["code"] == "AUTH_GATED_REPO"
    print(f"[OK] Diagnostic Gated classification: {diag_gated['title']} -> {diag_gated['hint'][:70]}...")

    # 4. EVALUATION SCORERS & 100-SAMPLE BENCHMARK
    samples = get_100_eval_samples()
    assert len(samples) == 100, f"Expected exactly 100 samples, got {len(samples)}"
    arc_count = sum(1 for s in samples if s["benchmark"] == "ARC")
    mmlu_count = sum(1 for s in samples if s["benchmark"] == "MMLU")
    assert arc_count == 50, f"Expected 50 ARC, got {arc_count}"
    assert mmlu_count == 50, f"Expected 50 MMLU, got {mmlu_count}"
    for s in samples:
        assert s["answer"] in ("A", "B", "C", "D"), f"Invalid answer key in sample {s}"
        assert len(s["choices"]) == 4, f"Invalid choices in sample {s}"
    print(f"[OK] Evaluation Suite: Exactly 100 verified samples (50 ARC + 50 MMLU).")

    # Test scorers
    em = exact_match_score("Chloroplast.", "chloroplast")
    assert em == 1.0
    b_score = bleu_score("The mitochondria is the powerhouse of the cell.", "Mitochondria is powerhouse of the cell")
    assert b_score > 0.3
    r_scores = rouge_scores("High heat and pressure form metamorphic rocks.", "Metamorphic rocks are formed through high heat and pressure.")
    assert r_scores["rouge_l"] >= 0.5
    rubric = evaluate_rubric(
        prediction="Metamorphic rocks are formed under intense geological heat and pressure.",
        ground_truth="High heat and pressure form metamorphic rocks.",
        instruction="How do metamorphic rocks form?",
    )
    assert rubric["score"] in (4, 5)
    print(f"[OK] Scorers: EM={em}, BLEU={b_score:.3f}, ROUGE-L={r_scores['rouge_l']:.3f}, Rubric={rubric['score']}/5 ({rubric['rationale']})")

    # 5. AUTH, SESSIONS, API KEYS & MULTI-TENANCY
    db = SessionLocal()
    try:
        test_user_id = 99901
        user = db.query(User).filter(User.id == test_user_id).first()
        if not user:
            user = User(
                id=test_user_id,
                email="engineer@enterprise.internal",
                username="test_engineer",
                hashed_password=auth_service.hash_password("SuperSecret123!"),
                role="admin",
            )
            db.add(user)
            db.commit()

        # Session creation & validation
        token_str = auth_service.create_session_token(
            user_id=user.id,
            ip_address="192.168.1.50",
            user_agent="Mozilla/5.0 Enterprise Client",
            db=db,
        )
        assert token_str.startswith("sess_")
        validated_user_id = auth_service.validate_session_token(token_str, db)
        assert validated_user_id == user.id

        # CSRF token generation & validation
        csrf_token = auth_service.generate_csrf_token(token_str)
        assert auth_service.validate_csrf_token(csrf_token, token_str) is True
        assert auth_service.validate_csrf_token("invalid_csrf_token", token_str) is False

        # API Key creation & validation
        raw_key, api_key_record = auth_service.create_api_key(
            user_id=user.id,
            project_id="proj_default",
            name="CI/CD Runner Key",
            db=db,
        )
        assert raw_key.startswith("dt_live_")
        key_user, key_project = auth_service.validate_api_key(raw_key, db)
        assert key_user.id == user.id
        assert key_project == "proj_default"

        # Revocation on logout
        auth_service.revoke_session_token(token_str, db)
        revoked_user = auth_service.validate_session_token(token_str, db)
        assert revoked_user is None, "Revoked session must return None!"
        print(f"[OK] Auth & Sessions: Session token generation, CSRF validation, API key auth, and instant revocation verified.")
    finally:
        db.close()

    # 6. ULID GENERATION & JOB SPEC ORCHESTRATION
    ulids = [generate_ulid() for _ in range(10)]
    assert len(set(ulids)) == 10, "ULIDs must be strictly unique"
    assert ulids == sorted(ulids), "ULIDs must be monotonically sortable"
    run_id = generate_run_id("qlora")
    job_id = generate_job_id("train")
    assert run_id.startswith("run_qlora_")
    assert job_id.startswith("job_train_")

    job_spec = JobSpec(
        job_id=job_id,
        run_id=run_id,
        display_name="Enterprise Llama-3 Fine-Tune Run",
        project_id="proj_default",
        method="sft-qlora",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        epochs=3,
        batch_size=2,
        learning_rate=2e-4,
    )
    enqueued = queue_manager.enqueue(job_spec)
    assert enqueued["run_id"] == run_id
    status = queue_manager.get_status(run_id)
    assert status["state"] in ("QUEUED", "RUNNING")
    queue_manager.mark_completed(run_id, {"final_loss": 1.234, "duration_sec": 450})
    finished = queue_manager.get_status(run_id)
    assert finished["state"] == "COMPLETED"
    print(f"[OK] Job Orchestration: ULIDs {run_id} enqueued, tracked, and completed successfully.")

    print("\n=== ALL REALISTIC END-TO-END SIMULATIONS PASSED CLEANLY! ===")

if __name__ == "__main__":
    run_simulation()
