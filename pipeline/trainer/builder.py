"""
pipeline/trainer/builder.py
---------------------------
Modular builder components for fine-tuning pipelines:
model instantiation, PEFT adapter configuration, dataset splitting,
loss masking data collators, and hardware-optimized training arguments.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

from pipeline.common import Config, get_tokenizer, print_trainable_parameters
from pipeline.hardware import HardwareProfile, get_hardware_profile, configure_training_hardware

logger = logging.getLogger("doctune.trainer.builder")

try:
    import torch
    from datasets import load_dataset, Dataset
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTConfig, DataCollatorForCompletionOnlyLM
    HAS_ML = True
except ImportError:
    torch = None
    load_dataset = None
    Dataset = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    LoraConfig = None
    prepare_model_for_kbit_training = None
    get_peft_model = None
    SFTConfig = None
    DataCollatorForCompletionOnlyLM = None
    HAS_ML = False


def build_model_and_peft(
    cfg: Config,
    hw: HardwareProfile,
    method: str = "qlora",
) -> Tuple[Any, Optional[Any]]:
    """
    Build and configure base causal language model and PEFT adapter wrappers.
    Adapts seamlessly across QLoRA (4-bit), LoRA (16-bit), and Full Fine-Tuning.
    """
    if not HAS_ML:
        raise ImportError("Machine learning libraries (torch, transformers, peft) are required.")

    hw_opts = configure_training_hardware(method=method, hw=hw)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(hw_opts["torch_dtype"], torch.float32)

    is_qlora = method.lower() in ("qlora", "sft-qlora")
    is_lora = method.lower() in ("lora", "sft-lora")
    is_fft = method.lower() in ("fft", "sft-full")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if hw.device == "cuda":
        model_kwargs["device_map"] = "auto"

    # Attention implementation
    if hw_opts.get("attn_implementation") == "flash_attention_2":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif hw_opts.get("attn_implementation") == "sdpa":
        model_kwargs["attn_implementation"] = "sdpa"

    # 1. QLoRA 4-bit Quantization Setup
    if is_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config

    logger.info("Instantiating base model: %s (precision: %s)", cfg.model_id, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)

    # 2. Prepare k-bit training if quantized
    if is_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # 3. PEFT Adapter Setup
    peft_config = None
    if is_qlora or is_lora:
        # Default to all-linear target modules if unspecified or set to default (PEFT >= 0.10)
        target_mods = list(cfg.target_modules) if cfg.target_modules else "all-linear"
        if target_mods == ["all-linear"] or target_mods == ("all-linear",):
            target_mods = "all-linear"

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_mods,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

    return model, peft_config


def build_dataset_and_eval(
    cfg: Config,
    tokenizer: Any,
) -> Tuple[Any, Optional[Any]]:
    """
    Load dataset from JSONL, perform train/validation split (min 20 samples),
    and return (train_dataset, eval_dataset).
    """
    if not HAS_ML:
        raise ImportError("datasets package required.")

    ds_path = Path(cfg.dataset_path)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found at {ds_path}")

    raw_ds = load_dataset("json", data_files=str(ds_path), split="train")
    total_samples = len(raw_ds)
    logger.info("Loaded %d dataset examples from %s", total_samples, ds_path)

    val_split = getattr(cfg, "val_split", 0.1)
    if val_split and val_split > 0 and total_samples >= 20:
        split_dict = raw_ds.train_test_split(test_size=val_split, seed=cfg.seed)
        train_ds = split_dict["train"]
        eval_ds = split_dict["test"]
        logger.info("Split dataset into %d training and %d validation samples", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    return raw_ds, None


class CompletionOnlyDataCollator:
    """
    Robust completion-only loss masking collator.
    Masks all prompt tokens up to the end of the response template with -100,
    ensuring gradients are only calculated on generated assistant tokens.
    """
    def __init__(self, response_template: str, tokenizer: Any):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, examples: list) -> Dict[str, Any]:
        import torch
        from transformers import default_data_collator
        batch = default_data_collator(examples)
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        resp_len = len(self.response_token_ids)

        if resp_len > 0:
            for i in range(len(input_ids)):
                seq = input_ids[i].tolist()
                found_idx = -1
                for j in range(len(seq) - resp_len + 1):
                    if seq[j:j + resp_len] == self.response_token_ids:
                        found_idx = j + resp_len
                        break
                if found_idx != -1:
                    labels[i, :found_idx] = -100

        return batch


def build_data_collator(
    cfg: Config,
    tokenizer: Any,
) -> Optional[Any]:
    """
    Build response-only / completion-only loss masking collator when loss_on == 'assistant'.
    Prevents model from wasting capacity memorizing prompt tokens.
    """
    loss_on = getattr(cfg, "loss_on", "assistant")
    if loss_on != "assistant" or not HAS_ML:
        return None

    # Detect assistant response delimiters from tokenizer chat template
    response_template = None
    chat_template = getattr(tokenizer, "chat_template", "") or ""

    if "<|start_header_id|>assistant<|end_header_id|>" in chat_template or "llama-3" in cfg.model_id.lower():
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "<|im_start|>assistant" in chat_template or "qwen" in cfg.model_id.lower():
        response_template = "<|im_start|>assistant\n"
    elif "<start_of_turn>model" in chat_template or "gemma" in cfg.model_id.lower():
        response_template = "<start_of_turn>model\n"
    elif "[/INST]" in chat_template or "mistral" in cfg.model_id.lower():
        response_template = "[/INST]"
    else:
        # Default fallback delimiter for chat formats
        response_template = "assistant\n"

    # Try TRL DataCollatorForCompletionOnlyLM first if available
    if DataCollatorForCompletionOnlyLM is not None:
        try:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            logger.info("Enabled TRL completion-only loss masking with delimiter: %r", response_template)
            return collator
        except Exception as ex:
            logger.debug("TRL DataCollatorForCompletionOnlyLM initialization failed: %s, falling back to built-in collator", ex)

    # Fallback to robust built-in completion collator
    try:
        collator = CompletionOnlyDataCollator(
            response_template=response_template,
            tokenizer=tokenizer,
        )
        logger.info("Enabled built-in completion-only loss masking with delimiter: %r", response_template)
        return collator
    except Exception as ex:
        logger.warning("Could not initialize completion loss masking collator: %s", ex)
        return None


def build_training_arguments(
    cfg: Config,
    hw: HardwareProfile,
    has_eval: bool = False,
    output_dir: Optional[Path] = None,
) -> Any:
    """
    Construct optimized Hugging Face / TRL SFTConfig with cosine learning rate
    schedules, early stopping, and step-based validation.
    """
    if not HAS_ML:
        raise ImportError("trl package required.")

    hw_opts = configure_training_hardware(method="qlora", hw=hw)
    out_dir = str(output_dir or cfg.output_dir or cfg.output_qlora)

    eval_strategy = "steps" if has_eval else "no"
    eval_steps = cfg.logging_steps * 5 if has_eval else None
    load_best = True if has_eval else False
    metric_best = "eval_loss" if has_eval else None

    return SFTConfig(
        output_dir=out_dir,
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate_lora,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy="steps" if has_eval else "epoch",
        save_steps=eval_steps if has_eval else None,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_best,
        bf16=hw_opts["bf16"],
        fp16=hw_opts["fp16"],
        optim=hw_opts["optim"],
        report_to="none",
        seed=cfg.seed,
    )
