# ONNX Exporter Plugin

## Overview
The **ONNX Exporter** converts a 🤗 Transformers model to the **ONNX** format so you can run efficient inference with ONNX Runtime on CPUs, GPUs (CUDA & AMD), or Apple Silicon (MLX).  
It can also apply **dynamic INT8 quantization** in one step, giving smaller models and faster inference with minimal impact on accuracy.

---

## Supported Model Architectures
- CohereForCausalLM
- FalconForCausalLM
- LlamaForCausalLM
- Gemma / Gemma2ForCausalLM
- GPT‑J ForCausalLM
- MistralForCausalLM
- MixtralForCausalLM
- Phi / Phi3ForCausalLM
- Qwen2 / Qwen3 (+ Qwen3Moe) ForCausalLM  
*(and any other models compatible with Optimum’s ONNX exporter)*

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **output_model_id** | string | *none* | Folder name (or HF repo name) where the exported ONNX model will be saved. |
| **opset** | integer | **17** | ONNX opset version. Keep the default unless you need an older opset for a specific runtime. |
| **quantize** | boolean | **false** | Apply dynamic INT8 quantization after export via Optimum. Produces a smaller `.onnx` file and can speed up inference. |

*(The plugin also uses the model selected in your experiment via `model_name`.)*

---
