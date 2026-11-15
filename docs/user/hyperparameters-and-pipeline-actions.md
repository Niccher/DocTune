# Machine Learning Hyperparameters & Pipeline Actions Guide

This guide provides deep technical documentation for operators, ML engineers, and developers running fine-tuning and alignment jobs on DocTune Studio.

---

## 1. Machine Learning Hyperparameters

When fine-tuning large language models on private domain documents, hyperparameters govern model stability, gradient convergence, memory usage, and catastrophic forgetting resistance.

### Parameter Reference & Formulas

| Parameter | Mathematical Formulation | Role & Practical Behavior | Recommended Default | Safe Range |
| :--- | :--- | :--- | :--- | :--- |
| **LoRA** | $W = W_0 + \Delta W$ where $\Delta W = \frac{\alpha}{r} (B \cdot A)$ | Decomposes dense weight updates into low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. Freezes $W_0$, slashing trainable parameters by $>99\%$. | Enabled (`qlora` 4-bit) | `qlora`, `lora`, `fft` |
| **LoRA Rank ($r$)** | $\text{rank} \le r \ll \min(d, k)$ | Inner bottleneck dimension. Higher values give the adapter more expressiveness and memory capacity for domain facts, but increase VRAM and risk overfitting. | **16** | 4 to 128 |
| **LoRA Alpha ($\alpha$)** | $\text{Scaling} = \frac{\alpha}{r}$ | Update scaling constant. Multiplies the output of $BA$ before adding to $W_0$. Balances newly learned adapter weights against base model pre-training. | **32** (Rule: $\alpha = 2r$) | 8 to 256 |
| **Epochs** | $\text{Steps} = \frac{N \times \text{Epochs}}{\text{Batch Size} \times \text{Grad Accum}}$ | Number of complete training cycles over the instruction dataset. Too few leads to underfitting; too many leads to verbatim memorization and hallucinations. | **3** | 1 to 10 |
| **Learning Rate** | $\theta \leftarrow \theta - \eta \nabla L(\theta)$ | Optimization step size. LoRA adapters require higher learning rates ($2\times 10^{-4}$) than Full Fine-Tuning ($1\times 10^{-5}$) because only small low-rank layers are adapted. | **0.0002** ($2\times 10^{-4}$) | $1\times 10^{-5}$ to $5\times 10^{-4}$ |
| **Batch Size** | Physical batch per forward pass | Number of prompt-completion sequences processed concurrently per GPU. Primary driver of activation memory. | **2** | 1 to 8 |
| **Grad Accum Steps** | $\text{EffBatch} = \text{Batch} \times \text{Accum} \times \text{GPUs}$ | Accumulates gradients across micro-batches before executing `optimizer.step()`. Simulates large batches with zero extra VRAM allocation. | **8** ($\text{Effective} = 16$) | 2 to 32 |
| **Max Sequence Length** | Attention memory $\sim \mathcal{O}(T^2)$ | Maximum token limit for prompt + completion. Attention tensor memory scales quadratically with length. | **1024** | 512 to 4096 |

---

### Hardware & VRAM Sizing Guide

Tested allocations using 4-bit QLoRA (`bitsandbytes` NormalFloat4 quantization):

| Model Architecture | Parameters | Recommended VRAM | Recommended Settings | Max Supported Context |
| :--- | :--- | :--- | :--- | :--- |
| **SmolLM2 / Qwen 2.5** | 1.5B – 3B | **6 GB – 8 GB** | Batch 4, Accum 4, Rank 16 | 2,048 tokens |
| **Llama 3 / Mistral** | 7B – 8B | **12 GB – 16 GB** | Batch 2, Accum 8, Rank 16 | 2,048 tokens |
| **Qwen 2.5** | 14B | **20 GB – 24 GB** | Batch 1, Accum 16, Rank 16 | 2,048 tokens |
| **Llama 3.1 / Qwen** | 70B | **48 GB – 80 GB** | Batch 1, Accum 32, Rank 32 | 4,096 tokens |

---

## 2. Human Feedback & Direct Preference Optimization (DPO)

### What is DPO?
Direct Preference Optimization (Rafailov et al., 2023) aligns language models to human preferences without training a separate reinforcement learning reward model (PPO).

### Mathematical Objective
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Where:
- $x$: Domain prompt / query.
- $y_w$: Preferred (chosen) completion (e.g. verified grounded answer).
- $y_l$: Dispreferred (rejected) completion (e.g. ungrounded hallucination or refusal failure).
- $\pi_\theta$: The model policy currently undergoing alignment training.
- $\pi_{\text{ref}}$: Frozen reference model (the initial SFT checkpoint).
- $\beta$: Regularization coefficient (typically `0.1`), penalizing divergence from $\pi_{\text{ref}}$.

### Data Collection in DocTune Studio
1. Operators query models in `/playground`.
2. Clicking **Thumbs Up** logs the fine-tuned output as $y_w$.
3. Clicking **Thumbs Down** opens a correction modal where the operator inputs the ground truth $y_w$, while the model's failed response becomes $y_l$.
4. Clicking **Export Dataset** compiles all preference triples into `data/preference_pairs.jsonl`.
5. Running `python -m pipeline.train_dpo` trains an aligned adapter checkpoint in `output_dpo/`.

---

## 3. Platform Pipeline Actions Reference

### Action 1: Document Ingestion & PII Scrubbing
- **Route**: `POST /api/v1/data-prep/upload`
- **Module**: `pipeline.prepare_data`, `pipeline.pii_redactor`
- **Details**:
  - Extracts raw text from `.pdf`, `.docx`, `.csv`, `.txt`, and `.md` files.
  - Regex PII masking filters out social security numbers, credit cards, email addresses, and API credentials.
  - Chunking applies a sliding token window (default 1,000 characters with 200 character overlap).
  - Embeddings are generated using `BAAI/bge-small-en-v1.5` and stored in embedded ChromaDB (`data/vectorstore/`).

### Action 2: Synthetic Instruction Synthesis (Teacher LLM)
- **Route**: `POST /api/v1/data-prep/start`
- **Module**: `pipeline.prepare_data`
- **Details**:
  - Prompts the configured Teacher LLM (Ollama, llama.cpp, vLLM, or cloud API).
  - Converts text excerpts into conversational QA instruction pairs.
  - Formats output into JSON Lines (`data/private_docs.jsonl`) with train/val splits.

### Action 3: QLoRA / LoRA Training Execution
- **Route**: `POST /api/v1/training/start`
- **Module**: `pipeline.train_qlora`, `pipeline.train_lora`, `pipeline.train_fft`
- **Details**:
  - Quantizes the base model into 4-bit NormalFloat (NF4).
  - Injects low-rank adapters into attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).
  - Broadcasts loss points over WebSocket (`/ws/stream`) to render real-time telemetry curves.
  - Writes adapter weights to `output_qlora/adapter_model.safetensors`.

### Action 4: Weight Merging & 1-Click GGUF Export
- **Route**: `POST /api/v1/models/merge`, `POST /api/v1/models/export-one-click`
- **Module**: `pipeline.merge_adapters`, `pipeline.export_gguf`
- **Details**:
  - Folds adapter weights into the base model ($W_0 + \frac{\alpha}{r}BA$).
  - Quantizes merged weights into GGUF (`q4_k_m`, `q8_0`) via `llama.cpp`.
  - Packages an Ollama `Modelfile` for instant native deployment.

### Action 5: Empirical 4-Way Evaluation & Regression Testing
- **Route**: `POST /api/v1/experiments/rag-benchmark`, `POST /api/v1/evaluate/run`
- **Module**: `pipeline.rag_benchmark`, `pipeline.eval.runner`
- **Details**:
  - Evaluates domain questions across 4 configurations: Base Zero-Shot, Base + RAG, Fine-Tuned, and Hybrid (FT + RAG).
  - Executes the 100-sample balanced ARC/MMLU benchmark (`eval/dataset_100.py`) to verify zero general reasoning degradation.
  - Persists metrics to PostgreSQL/SQLite `eval_benchmarks` table.
