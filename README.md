# gemma4-unsloth

A Jupyter notebook for fine-tuning Google's **Gemma 4 31B** multimodal LLM using [Unsloth](https://github.com/unslothai/unsloth) — 2x faster training with significantly less VRAM.

---

## What This Notebook Does

### 1. Installation
Installs PyTorch, Unsloth, Transformers, and related dependencies.

### 2. Model Loading
Loads `unsloth/gemma-4-31B-it` with:
- **4-bit quantization** to reduce memory usage
- **8192 token max sequence length**
- Balanced across 2x Tesla T4 GPUs

### 3. Multimodal Inference Demo
Shows off Gemma 4's capabilities before fine-tuning:
- **Vision**: Sends an image and asks questions about it
- **Text**: Generates poems and answers general questions

### 4. LoRA Adapter Setup
Adds LoRA adapters with `r=8`, targeting language, attention, and MLP modules.
Only **0.20% of parameters** (61M out of 31B) are trainable — vision layers are disabled for this text-only fine-tune.

### 5. Data Preparation
- Uses [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k), a high-quality instruction dataset (first 3000 rows)
- Applies Gemma-4's `gemma-4-thinking` chat template
- Uses `train_on_responses_only` — the model only learns from assistant replies, not user prompts

### 6. Training
- **SFTTrainer** (Supervised Fine-Tuning) from the `trl` library
- 60 steps (quick demo), batch size 1 with gradient accumulation of 4
- Learning rate: `2e-4`, optimizer: `adamw_8bit`
- Took ~44 minutes, using ~10.9 GB of 14.5 GB GPU memory

### 7. Post-Training Inference
Tests the fine-tuned model on sample prompts using streaming output.

### 8. Saving
Options to save the model as:
- **LoRA adapters only** (lightweight)
- **float16 merged** (for vLLM deployment)
- **GGUF format** (for llama.cpp / Ollama)
- Push to Hugging Face Hub

---

## Key Technologies

| Tool | Purpose |
|------|---------|
| Unsloth | 2x faster training, memory optimization |
| LoRA (PEFT) | Fine-tune only ~0.2% of parameters |
| 4-bit quantization | Fit 31B model on consumer GPUs |
| TRL SFTTrainer | Supervised fine-tuning framework |
| FineTome-100k | High-quality instruction dataset |

---

## Requirements

- GPU with 15+ GB VRAM (e.g. Tesla T4, A100) — recommended on Google Colab
- Python 3.10+
- See the notebook's installation cell for full dependencies
