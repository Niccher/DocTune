"""
web/services/export_service.py
------------------------------
One-click model export service for DocTune Studio.
Automates merging LoRA/QLoRA adapters with base causal language models,
quantizing to GGUF format, generating custom Ollama Modelfiles, and registering
the export in the platform checkpoint registry.
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.common import PROJECT_ROOT, Config
from pipeline.export_gguf import run_conversion, detect_chat_template, generate_ollama_modelfile

logger = logging.getLogger("doctune.export_service")


class ExportService:
    """Manages one-click GGUF conversion and Ollama packaging."""

    def resolve_adapter_and_base(
        self,
        run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        model_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Locate adapter weights, base model identifier, and determine whether merging is needed.
        """
        cfg = Config.load()
        source_dir = None
        base_model_id = cfg.model_id
        is_adapter = False

        # 0. Check explicit model_dir if provided
        if model_dir:
            p = Path(model_dir) if Path(model_dir).is_absolute() else PROJECT_ROOT / model_dir
            if p.exists():
                source_dir = p
                if (p / "adapter_config.json").exists():
                    is_adapter = True
                elif any(p.glob("*.safetensors")) or any(p.glob("*.bin")):
                    is_adapter = False

        # 1. Check if run_id provided
        if not source_dir and run_id:
            candidate_dirs = [
                PROJECT_ROOT / "artifacts" / "runs" / run_id / "adapter",
                PROJECT_ROOT / "artifacts" / "runs" / run_id,
                PROJECT_ROOT / "output_qlora",
                PROJECT_ROOT / "output_lora",
            ]
            for c in candidate_dirs:
                if c.exists() and (c / "adapter_config.json").exists():
                    source_dir = c
                    is_adapter = True
                    break

        # 2. Check if checkpoint_id provided
        if not source_dir and checkpoint_id:
            try:
                from web.db.session import SessionLocal
                from web.db.models import Checkpoint
                if SessionLocal:
                    db = SessionLocal()
                    ckpt = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
                    if ckpt and ckpt.disk_path:
                        p = Path(ckpt.disk_path)
                        if p.exists():
                            source_dir = p
                            if (p / "adapter_config.json").exists():
                                is_adapter = True
                            if ckpt.base_model:
                                base_model_id = ckpt.base_model
                    db.close()
            except Exception as e:
                logger.debug("Database checkpoint lookup failed: %s", e)

        # 3. Fallback to default output directories
        if not source_dir:
            if (PROJECT_ROOT / "output_merged").exists() and any((PROJECT_ROOT / "output_merged").glob("*.safetensors")):
                source_dir = PROJECT_ROOT / "output_merged"
                is_adapter = False
            elif (PROJECT_ROOT / "output_qlora" / "adapter_config.json").exists():
                source_dir = PROJECT_ROOT / "output_qlora"
                is_adapter = True
            elif (PROJECT_ROOT / "output_lora" / "adapter_config.json").exists():
                source_dir = PROJECT_ROOT / "output_lora"
                is_adapter = True
            elif (PROJECT_ROOT / "output_fft").exists():
                source_dir = PROJECT_ROOT / "output_fft"
                is_adapter = False

        # Read base model from adapter_config.json if available
        if source_dir and (source_dir / "adapter_config.json").exists():
            try:
                with open(source_dir / "adapter_config.json", "r", encoding="utf-8") as f:
                    cfg_json = json.load(f)
                    base_model_id = cfg_json.get("base_model_name_or_path") or base_model_id
            except Exception as e:
                logger.debug("Could not read base model from adapter_config.json: %s", e)

        return {
            "source_dir": source_dir,
            "base_model_id": base_model_id,
            "is_adapter": is_adapter,
        }

    def export_one_click(
        self,
        run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        quantization: str = "q4_k_m",
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full automated export:
        1. Auto-detect adapter or merged source.
        2. Merge adapter with base model if necessary.
        3. Quantize to GGUF and generate Ollama Modelfile.
        4. Register the export in the checkpoint database.
        """
        res = self.resolve_adapter_and_base(run_id=run_id, checkpoint_id=checkpoint_id, model_dir=model_dir)
        source_dir = res["source_dir"]
        base_model_id = res["base_model_id"]
        is_adapter = res["is_adapter"]

        if not source_dir or not source_dir.exists():
            raise FileNotFoundError("No trained checkpoint or adapter directory found to export.")

        safe_name = (model_name or run_id or "doctune-model").lower().replace("_", "-").replace(" ", "-")
        # Ensure name is valid for Ollama tags
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "."))

        # Target export directory
        if run_id:
            export_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id / "gguf"
        else:
            export_dir = PROJECT_ROOT / "output_gguf" / safe_name
        export_dir.mkdir(parents=True, exist_ok=True)

        merged_dir = source_dir
        # If it's an unmerged adapter, perform merge first
        if is_adapter:
            logger.info("Unmerged adapter detected at %s. Merging with base model %s...", source_dir, base_model_id)
            if run_id:
                merged_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id / "merged"
            else:
                merged_dir = PROJECT_ROOT / "output_merged" / safe_name
            merged_dir.mkdir(parents=True, exist_ok=True)

            try:
                from pipeline.merge_adapters import run_merge
                run_merge(source_dir, merged_dir, base_model_id)
            except Exception as e:
                logger.warning("Dynamic PEFT merge encountered an issue: %s. Using standalone fallback packaging.", e)
                # Standalone fallback: copy adapter and tokenizer files
                for item in source_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, merged_dir / item.name)

        # Run GGUF conversion & Modelfile generation
        conversion_result = run_conversion(
            model_dir=merged_dir,
            export_dir=export_dir,
            model_name=safe_name,
            quantization=quantization,
        )

        gguf_path = Path(conversion_result["gguf_path"])
        modelfile_path = Path(conversion_result["modelfile_path"])
        run_script_path = Path(conversion_result["run_script_path"])

        # Calculate file size
        size_mb = 0.0
        if gguf_path.exists():
            size_mb = round(gguf_path.stat().st_size / (1024 * 1024), 2)

        # Register in database checkpoint registry
        try:
            from web.db.session import SessionLocal
            from web.db.models import Checkpoint
            if SessionLocal:
                db = SessionLocal()
                ckpt_id = f"ckpt_gguf_{safe_name}_{int(time.time())}"
                new_ckpt = Checkpoint(
                    id=ckpt_id,
                    project_id="proj_default",
                    run_id=run_id,
                    name=f"{safe_name} ({quantization.upper()})",
                    adapter_type="gguf",
                    base_model=base_model_id,
                    disk_path=str(gguf_path),
                    size_mb=size_mb,
                )
                db.add(new_ckpt)
                db.commit()
                db.close()
        except Exception as ex:
            logger.debug("Database checkpoint registration notice: %s", ex)

        # Read Modelfile content for quick preview
        modelfile_content = ""
        if modelfile_path.exists():
            with open(modelfile_path, "r", encoding="utf-8") as f:
                modelfile_content = f.read()

        return {
            "status": "success",
            "run_id": run_id,
            "model_name": safe_name,
            "quantization": quantization,
            "gguf_filename": gguf_path.name,
            "gguf_path": str(gguf_path),
            "modelfile_path": str(modelfile_path),
            "run_script_path": str(run_script_path),
            "modelfile_content": modelfile_content,
            "size_mb": size_mb,
            "ollama_command": f"ollama create {safe_name} -f Modelfile && ollama run {safe_name}",
            "llamacpp_command": f"llama-server -m ./{gguf_path.name} --port 8080 -c 4096",
            "download_gguf_url": f"/api/v1/models/download-file?path={gguf_path.name}&run_id={run_id or ''}&model_name={safe_name}",
            "download_modelfile_url": f"/api/v1/models/download-file?path=Modelfile&run_id={run_id or ''}&model_name={safe_name}",
        }

    def get_existing_export(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Check if an export already exists for a run and return its details."""
        if not run_id:
            return None
        export_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id / "gguf"
        if not export_dir.exists():
            return None

        gguf_files = list(export_dir.glob("*.gguf"))
        modelfile = export_dir / "Modelfile"

        if not gguf_files or not modelfile.exists():
            return None

        primary_gguf = gguf_files[0]
        size_mb = round(primary_gguf.stat().st_size / (1024 * 1024), 2)
        safe_name = primary_gguf.stem

        with open(modelfile, "r", encoding="utf-8") as f:
            modelfile_content = f.read()

        return {
            "status": "success",
            "run_id": run_id,
            "model_name": safe_name,
            "gguf_filename": primary_gguf.name,
            "gguf_path": str(primary_gguf),
            "modelfile_path": str(modelfile),
            "modelfile_content": modelfile_content,
            "size_mb": size_mb,
            "ollama_command": f"ollama create {safe_name} -f Modelfile && ollama run {safe_name}",
            "llamacpp_command": f"llama-server -m ./{primary_gguf.name} --port 8080 -c 4096",
            "download_gguf_url": f"/api/v1/models/download-file?path={primary_gguf.name}&run_id={run_id}&model_name={safe_name}",
            "download_modelfile_url": f"/api/v1/models/download-file?path=Modelfile&run_id={run_id}&model_name={safe_name}",
        }


export_service = ExportService()
