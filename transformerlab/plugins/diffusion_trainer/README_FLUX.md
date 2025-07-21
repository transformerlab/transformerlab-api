# FLUX Trainer - Memory-Efficient Training for Consumer GPUs

This document describes the FLUX trainer implementation that enables training FLUX models on consumer GPUs with limited VRAM (~24GB).

## Overview

The FLUX trainer (`flux_trainer.py`) is automatically used when the model architecture is detected as "FluxPipeline". It implements memory-efficient training techniques inspired by Kohya's scripts to handle the large memory requirements of FLUX models.

## Key Features

### Memory Optimizations
- **Component-wise loading**: Loads model components individually to control memory usage
- **CPU offloading**: Moves unused components to CPU when not needed
- **Block swapping**: Swaps transformer blocks between CPU and GPU during training
- **Gradient checkpointing**: Reduces memory usage during backward pass
- **Frequent cleanup**: Aggressive garbage collection and CUDA cache clearing

### FLUX-Specific Adaptations
- **Flow matching training**: Uses flow matching loss instead of standard diffusion loss
- **Dual text encoders**: Handles CLIP-L and T5XXL text encoders properly
- **Transformer architecture**: Targets appropriate modules for LoRA adaptation
- **Packed latents**: Handles FLUX's packed latent representation

### Training Parameters
- **Small batch sizes**: Defaults to batch size 1 for memory efficiency
- **LoRA target modules**: Optimized for FLUX transformer blocks:
  - `to_q`, `to_k`, `to_v`, `to_out.0` (attention layers)
  - `ff.net.0.proj`, `ff.net.2` (feed-forward layers)

## Automatic Routing

The main trainer (`main.py`) automatically detects FLUX models and routes to the FLUX trainer:

```python
if "FluxPipeline" in model_architecture:
    print("***** Detected FLUX model - using memory-efficient FLUX trainer *****")
    from . import flux_trainer
    return flux_trainer.train_flux_lora()
```

## Configuration

### Required Parameters
- `model_architecture`: Must be "FluxPipeline"
- `model_name` or `model_path`: Path to FLUX model
- `train_batch_size`: Recommended 1-2 for consumer GPUs

### Optional Memory Optimization Parameters
- `blocks_to_swap`: Number of transformer blocks to swap (default: 0)
- `gradient_checkpointing`: Enable gradient checkpointing (default: True)
- `enable_xformers_memory_efficient_attention`: Use xFormers if available
- `mixed_precision`: "fp16" or "bf16" for reduced memory usage

### Disabled Features
- **Evaluation**: Disabled by default due to memory constraints
- **EMA**: Not implemented to save memory
- **Large batch training**: Limited by memory constraints

## Memory Requirements

### Minimum Requirements
- **VRAM**: 16GB (with aggressive optimizations)
- **Recommended**: 24GB for stable training
- **System RAM**: 32GB+ recommended for CPU offloading

### Memory Saving Tips
1. Use `train_batch_size=1`
2. Enable `gradient_checkpointing=True`
3. Set `mixed_precision="fp16"`
4. Use `blocks_to_swap` if available
5. Reduce `resolution` if needed

## Limitations

### Current Limitations
- Evaluation disabled due to memory constraints
- Simplified flow matching implementation
- Limited batch sizes
- No support for advanced features like EMA

### Future Improvements
- Full Kohya library integration for better efficiency
- Advanced block swapping implementation
- CPU offloading for text encoders
- Memory usage monitoring and optimization

## Usage Example

```python
args = {
    "model_architecture": "FluxPipeline",
    "model_name": "black-forest-labs/FLUX.1-dev",
    "train_batch_size": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "lora_r": 4,
    "lora_alpha": 4,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
}
```

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size to 1
2. Enable gradient checkpointing
3. Use fp16 mixed precision
4. Reduce image resolution
5. Enable block swapping if available

### Performance Issues
1. Ensure CUDA is available
2. Use appropriate mixed precision
3. Enable xFormers if installed
4. Monitor GPU memory usage

## Integration with Kohya Scripts

This implementation is inspired by and compatible with Kohya's training scripts. For production use, consider integrating the full Kohya library for maximum efficiency:

- `library.flux_utils`: FLUX-specific utilities
- `library.flux_train_utils`: Training utilities
- `library.strategy_flux`: Caching and optimization strategies
- Flow matching scheduler implementation

## Contributing

When contributing to the FLUX trainer:

1. Test with different GPU memory configurations
2. Monitor memory usage throughout training
3. Maintain compatibility with the main trainer interface
4. Document any new memory optimization techniques
