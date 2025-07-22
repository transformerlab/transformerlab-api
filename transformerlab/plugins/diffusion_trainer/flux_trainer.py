# FLUX training with memory-efficient CPU offloading
# Based on Kohya's memory-efficient implementation for consumer GPUs
# This implementation follows the structure from kohya-ss/sd-scripts

import math
import random
import json
import os
import gc
import copy
from multiprocessing import Value
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.utils import set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers

# Import Kohya utilities
from library import flux_utils
from library import flux_train_utils
from library import strategy_flux
from library import strategy_base
from library import train_util
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.device_utils import clean_memory_on_device


def cleanup_pipeline():
    """Clean up pipeline to free VRAM"""
    try:
        # Force garbage collection multiple times
        gc.collect()
        gc.collect()  # Second call often helps

        if torch.cuda.is_available():
            # Clear CUDA cache and synchronize multiple times for better cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()  # Clean up inter-process communication
            torch.cuda.empty_cache()  # Second empty_cache call

    except Exception as e:
        print(f"Warning: Failed to cleanup pipeline: {str(e)}")


cleanup_pipeline()


def compute_loss_weighting(args, timesteps, noise_scheduler):
    """
    Compute loss weighting for improved training stability.
    Supports min-SNR weighting similar to Kohya's implementation.
    """
    if args.get("min_snr_gamma") is not None and args.get("min_snr_gamma") != "":
        snr = compute_snr(noise_scheduler, timesteps)
        min_snr_gamma = float(args.get("min_snr_gamma"))
        snr_weight = torch.stack([snr, min_snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        return snr_weight
    elif args.get("snr_gamma") is not None and args.get("snr_gamma") != "":
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, float(args["snr_gamma"]) * torch.ones_like(timesteps)], dim=1).min(dim=1)[
            0
        ]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        return mse_loss_weights
    return None


def compute_loss(model_pred, target, timesteps, noise_scheduler, args):
    """
    Compute loss with support for different loss types and weighting schemes.
    """
    loss_type = args.get("loss_type", "l2")

    if loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(model_pred.float(), target.float(), reduction="none", beta=args.get("huber_c", 0.1))
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

    # Apply loss weighting if specified
    loss_weights = compute_loss_weighting(args, timesteps, noise_scheduler)

    if loss_weights is not None and not torch.all(loss_weights == 0):
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_weights
        return loss.mean()
    else:
        return loss.mean()


def load_flux_components_memory_efficient(pretrained_model_name_or_path, weight_dtype, device, args):
    """
    Load FLUX components using Kohya's memory-efficient approach.
    """
    print(f"Loading FLUX components with Kohya's memory-efficient strategy from: {pretrained_model_name_or_path}")

    # Use Kohya's flux_utils to load components efficiently
    disable_mmap = args.get("disable_mmap_load_safetensors", False)

    # Load FLUX transformer model
    _, flux = flux_utils.load_flow_model(pretrained_model_name_or_path, weight_dtype, "cpu", disable_mmap)

    # Load VAE (AutoEncoder)
    ae = flux_utils.load_ae(args.get("ae", None), weight_dtype, "cpu", disable_mmap)

    # Load text encoders
    clip_l = flux_utils.load_clip_l(args.get("clip_l", None), weight_dtype, "cpu", disable_mmap)
    t5xxl = flux_utils.load_t5xxl(args.get("t5xxl", None), weight_dtype, "cpu", disable_mmap)

    # Set requires_grad to False for all components initially
    flux.requires_grad_(False)
    ae.requires_grad_(False)
    clip_l.requires_grad_(False)
    t5xxl.requires_grad_(False)

    # Set to eval mode
    flux.eval()
    ae.eval()
    clip_l.eval()
    t5xxl.eval()

    cleanup_pipeline()

    return flux, ae, clip_l, t5xxl


def get_flux_noise_scheduler(shift=1.15):
    """Create FLUX-compatible noise scheduler using Kohya's implementation"""
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)


# @tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def train_flux_lora(tlab_trainer):
    # Extract parameters from tlab_trainer
    args = tlab_trainer.params

    print("***** Running FLUX LoRA training with Kohya's memory-efficient approach *****")

    # Setup logging directory
    output_dir = args.get("output_dir", "flux-model-finetuned-lora")

    # Disable evaluation for FLUX as mentioned in the original script
    print("Note: Evaluation is disabled for FLUX training due to memory constraints")
    args["eval_prompt"] = None
    args["eval_steps"] = 0

    # Load dataset using tlab_trainer
    datasets_dict = tlab_trainer.load_dataset(["train"])
    dataset = datasets_dict["train"]

    # Model loading
    pretrained_model_name_or_path = args.get("model_name")
    if args.get("model_path") is not None and args.get("model_path").strip() != "":
        pretrained_model_name_or_path = args.get("model_path")
    else:
        # Convert model name to path from hf cache
        from huggingface_hub import snapshot_download

        model_cache_path = snapshot_download(pretrained_model_name_or_path)
        pretrained_model_name_or_path = model_cache_path

    # Mixed precision setup
    weight_dtype = torch.float32
    mixed_precision = args.get("mixed_precision", None)
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FLUX components using Kohya's approach
    transformer, vae, clip_l, t5xxl = load_flux_components_memory_efficient(
        pretrained_model_name_or_path, weight_dtype, device, args
    )

    print("FLUX components loaded successfully using Kohya's methods")
    print(f"Transformer type: {type(transformer).__name__}")
    print(f"VAE type: {type(vae).__name__}")

    # Check xformers availability
    try:
        import xformers  # noqa: F401

        xformers_available = True
    except ImportError:
        xformers_available = False

    # Enable memory optimizations
    if args.get("enable_xformers_memory_efficient_attention", False) and xformers_available:
        try:
            transformer.enable_xformers_memory_efficient_attention()
            if hasattr(vae, "enable_xformers_memory_efficient_attention"):
                vae.enable_xformers_memory_efficient_attention()
            print("xFormers memory efficient attention enabled")
        except Exception as e:
            print(f"Failed to enable xFormers: {e}")

    # Enable gradient checkpointing with CPU offloading for memory savings
    if args.get("gradient_checkpointing", True):  # Default to True for FLUX
        transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Move components to appropriate devices
    vae.to(device, dtype=weight_dtype)
    clip_l.to(device, dtype=weight_dtype)
    t5xxl.to(device, dtype=weight_dtype)

    # FLUX-specific LoRA configuration
    flux_lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",  # Attention layers
        "ff.net.0.proj",
        "ff.net.2",  # Feed-forward layers
    ]

    transformer_lora_config = LoraConfig(
        r=int(args.get("lora_r", 4)),
        lora_alpha=int(args.get("lora_alpha", 4)),
        init_lora_weights="gaussian",
        target_modules=flux_lora_target_modules,
    )

    print(f"Using FLUX LoRA target modules: {flux_lora_target_modules}")

    # Add LoRA to transformer
    transformer.add_adapter(transformer_lora_config)

    # Enable block swapping for memory efficiency (Kohya's approach)
    blocks_to_swap = int(args.get("blocks_to_swap", 0))
    if blocks_to_swap > 0:
        print(f"Enabling block swap for memory efficiency: blocks_to_swap={blocks_to_swap}")
        transformer.enable_block_swap(blocks_to_swap, device)

    # Move transformer to device after LoRA setup
    transformer.to(device, dtype=weight_dtype)
    transformer.requires_grad_(True)  # Enable gradients for LoRA training

    if mixed_precision == "fp16":
        cast_training_params(transformer, dtype=torch.float32)

    # Get trainable parameters
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    # Setup FLUX tokenization strategy following Kohya's approach
    _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(pretrained_model_name_or_path)
    t5xxl_max_token_length = 256 if is_schnell else 512

    flux_tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length)
    strategy_base.TokenizeStrategy.set_strategy(flux_tokenize_strategy)

    flux_text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(args.get("apply_t5_attn_mask", True))
    strategy_base.TextEncodingStrategy.set_strategy(flux_text_encoding_strategy)

    # Data transforms for FLUX
    transform_list = [
        transforms.Resize(args.get("resolution", 512)),
        transforms.CenterCrop(args.get("resolution", 512))
        if args.get("center_crop", False)
        else transforms.RandomCrop(args.get("resolution", 512)),
    ]

    if args.get("random_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_transforms = transforms.Compose(transform_list)

    def tokenize_captions_flux(examples, is_train=True):
        """Tokenize captions using Kohya's FLUX tokenization strategy"""
        captions = []
        caption_column = args.get("caption_column", "text")
        trigger_word = args.get("trigger_word", "").strip()
        caption_dropout_rate = float(args.get("caption_dropout_rate", 0.0))

        for caption in examples[caption_column]:
            if isinstance(caption, str):
                processed_caption = caption
            elif isinstance(caption, (list, np.ndarray)):
                processed_caption = random.choice(caption) if is_train else caption[0]
            else:
                raise ValueError("Caption column should contain strings or lists of strings.")

            # Apply caption dropout
            if is_train and caption_dropout_rate > 0 and random.random() < caption_dropout_rate:
                processed_caption = ""
            else:
                if trigger_word:
                    processed_caption = f"{trigger_word}, {processed_caption}"

            captions.append(processed_caption)

        # Use Kohya's tokenization strategy
        tokens_and_masks = []
        for caption in captions:
            tokens = flux_tokenize_strategy.tokenize(caption)
            tokens_and_masks.append(tokens)

        # Convert to batch format
        l_tokens = torch.stack([t[0] for t in tokens_and_masks])
        t5_tokens = torch.stack([t[1] for t in tokens_and_masks])
        t5_attn_masks = torch.stack([t[2] for t in tokens_and_masks])

        return {"input_ids_list": [l_tokens, t5_tokens, t5_attn_masks]}

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.get("image_column", "image")]]
        processed_images = [train_transforms(image) for image in images]

        examples["images"] = processed_images

        tokenization_results = tokenize_captions_flux(examples)
        examples["input_ids_list"] = tokenization_results["input_ids_list"]

        return examples

    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        images = torch.stack([example["images"] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()

        # Collect input_ids_list for batch
        input_ids_list = [example["input_ids_list"] for example in examples]

        # Stack tokens properly
        l_tokens_batch = torch.stack([ids[0] for ids in input_ids_list])
        t5_tokens_batch = torch.stack([ids[1] for ids in input_ids_list])
        t5_attn_masks_batch = torch.stack([ids[2] for ids in input_ids_list])

        batch = {"images": images, "input_ids_list": [l_tokens_batch, t5_tokens_batch, t5_attn_masks_batch]}

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=int(args.get("train_batch_size", 1)),  # Small batch size for FLUX
        num_workers=int(args.get("dataloader_num_workers", 0)),
    )

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=float(args.get("learning_rate", 1e-4)),
        betas=(float(args.get("adam_beta1", 0.9)), float(args.get("adam_beta2", 0.999))),
        weight_decay=float(args.get("adam_weight_decay", 1e-2)),
        eps=float(args.get("adam_epsilon", 1e-8)),
    )

    # Scheduler
    num_train_epochs = int(args.get("num_train_epochs", 100))
    gradient_accumulation_steps = int(args.get("gradient_accumulation_steps", 1))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=int(args.get("lr_warmup_steps", 500)),
        num_training_steps=max_train_steps,
    )

    # FLUX noise scheduler using Kohya's implementation
    noise_scheduler = get_flux_noise_scheduler(shift=args.get("discrete_flow_shift", 1.15))

    # Training loop
    print("***** Running FLUX training with Kohya's approach *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Batch size = {args.get('train_batch_size', 1)}")
    print(f"  Total optimization steps = {max_train_steps}")

    global_step = 0

    for epoch in range(num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            # Encode images to latents using VAE
            with torch.no_grad():
                latents = vae.encode(batch["images"].to(vae.dtype)).to(device, dtype=weight_dtype)

            # Use Kohya's noise and timestep generation
            noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, torch.randn_like(latents), device, weight_dtype
            )

            # Text encoding using Kohya's strategy
            with torch.no_grad():
                tokens_and_masks = batch["input_ids_list"]
                input_ids = [ids.to(device) for ids in tokens_and_masks]
                text_encoder_conds = flux_text_encoding_strategy.encode_tokens(
                    flux_tokenize_strategy, [clip_l, t5xxl], input_ids, args.get("apply_t5_attn_mask", True)
                )

            # Pack latents using Kohya's implementation
            packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)
            packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2

            # Get image IDs for FLUX
            bsz = packed_noisy_model_input.shape[0]
            img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=device)

            # Get guidance
            guidance_vec = torch.full((bsz,), float(args.get("guidance_scale", 3.5)), device=device)

            # Forward pass through FLUX transformer
            l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
            if not args.get("apply_t5_attn_mask", True):
                t5_attn_mask = None

            model_pred = transformer(
                img=packed_noisy_model_input,
                img_ids=img_ids,
                txt=t5_out,
                txt_ids=txt_ids,
                y=l_pooled,
                timesteps=timesteps / 1000,
                guidance=guidance_vec,
                txt_attention_mask=t5_attn_mask,
            )

            # Unpack latents using Kohya's implementation
            model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

            # Apply model prediction type using Kohya's method
            model_pred, weighting = flux_train_utils.apply_model_prediction_type(
                args, model_pred, noisy_model_input, sigmas
            )

            # Flow matching loss (FLUX-specific)
            target = torch.randn_like(latents) - latents  # noise - latents for flow matching

            # Calculate loss using Kohya's method
            huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
            loss = train_util.conditional_loss(
                model_pred.float(), target.float(), args.get("loss_type", "l2"), "none", huber_c
            )
            if weighting is not None:
                loss = loss * weighting
            loss = loss.mean([1, 2, 3])
            loss = loss.mean()  # Simple mean for now, could add loss weights per sample

            print(
                f"Epoch {epoch + 1}/{num_train_epochs}, Step {step + 1}/{len(train_dataloader)} - Loss: {loss.item()}"
            )

            # Backward pass
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(list(lora_layers), float(args.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Memory cleanup for FLUX
                if torch.cuda.is_available() and global_step % 5 == 0:
                    clean_memory_on_device(device)

                global_step += 1

                # Progress reporting
                percent_complete = 100.0 * global_step / max_train_steps
                tlab_trainer.progress_update(percent_complete)
                tlab_trainer.log_metric("train/loss", loss.item(), global_step)
                tlab_trainer.log_metric("train/lr", lr_scheduler.get_last_lr()[0], global_step)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # Save LoRA weights
    transformer = transformer.to(torch.float32)
    model_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))

    save_directory = args.get("adaptor_output_dir", output_dir)
    print(f"Saving FLUX LoRA weights to {save_directory}")
    os.makedirs(save_directory, exist_ok=True)

    # Save configuration info
    save_info = {
        "model_architecture": "FluxPipeline",
        "lora_config": {
            "r": str(transformer_lora_config.r),
            "lora_alpha": str(transformer_lora_config.lora_alpha),
            "target_modules": str(transformer_lora_config.target_modules),
        },
        "tlab_trainer_used": True,
        "flux_specific": True,
        "kohya_approach": True,
    }
    with open(os.path.join(save_directory, "tlab_adaptor_info.json"), "w") as f:
        json.dump(save_info, f, indent=4)

    # Save FLUX LoRA weights using Kohya's approach
    saved_successfully = False

    # Try FLUX-specific save method
    try:
        from diffusers import FluxPipeline

        FluxPipeline.save_lora_weights(
            save_directory=save_directory,
            transformer_lora_layers=model_lora_state_dict,
            safe_serialization=True,
        )
        print("FLUX LoRA weights saved successfully using FluxPipeline.save_lora_weights")
        saved_successfully = True
    except Exception as e:
        print(f"Error with FluxPipeline.save_lora_weights: {e}")

    # Fallback to safetensors
    if not saved_successfully:
        try:
            from safetensors.torch import save_file

            save_file(model_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.safetensors"))
            print(f"FLUX LoRA weights saved to {save_directory}/pytorch_lora_weights.safetensors")
            saved_successfully = True
        except Exception as e:
            print(f"Error saving with safetensors: {e}")
            # Final fallback
            torch.save(model_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.bin"))
            print(f"FLUX LoRA weights saved to {save_directory}/pytorch_lora_weights.bin")
            saved_successfully = True

    if saved_successfully:
        print(f"FLUX LoRA weights successfully saved to {save_directory}")
    else:
        print(f"Failed to save FLUX LoRA weights to {save_directory}")

    print("***** FLUX training completed using Kohya's memory-efficient approach *****")
