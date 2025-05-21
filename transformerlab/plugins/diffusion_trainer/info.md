# Stable Diffusion LoRA Trainer

## Introduction

The Stable Diffusion LoRA Trainer plugin enables fine-tuning of Stable Diffusion models using LoRA adapters. This trainer is designed for efficient and scalable customization of diffusion models on your own datasets.

## Features

- Fine-tune any HuggingFace-compatible Stable Diffusion model with LoRA adapters
- Flexible data preprocessing and augmentation options
- Customizable optimizer, learning rate scheduler, and training parameters
- Integrated logging with Weights & Biases (W&B)

## Parameters

Key parameters include:

- **model_name_manual**: HuggingFace model hub name or local path for the base Stable Diffusion model (required)
- **train_batch_size**: Number of images per batch (default: 16)
- **num_train_epochs**: Number of training epochs (default: 10)
- **learning_rate**: Learning rate for optimizer (default: 1e-4)
- **lora_r**: LoRA rank (default: 4)
- **lora_alpha**: LoRA scaling factor (default: 4)
- **resolution**: Image resolution for training (default: 512)
- **center_crop**: Use center crop for images (default: false)
- **random_flip**: Apply random horizontal flip (default: false)
- **log_to_wandb**: Log training to Weights & Biases (default: true)

See the plugin config panel for the full list and descriptions.

## Usage

1. Prepare your dataset with image and caption columns.
2. Configure the plugin parameters as needed.
3. Launch training via the TransformerLab interface or job system.
4. After training, LoRA adapter weights will be saved for use with Stable Diffusion pipelines.

## Output

- LoRA adapter weights saved to the specified output directory.
- Optional model card and metadata for reproducibility.