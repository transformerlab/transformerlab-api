import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop

from diffusers import AutoencoderKL, DDPMScheduler, AutoPipelineForText2Image, UNet2DConditionModel, StableDiffusionPipeline
# Additional pipeline imports will be done conditionally in the save section to avoid import errors
# SD3 and FLUX pipelines will be imported as needed during save process
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers

from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR

check_min_version("0.34.0.dev0")

def cleanup_pipeline():
    """Clean up pipeline to free VRAM"""
    try:
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    except Exception as e:
        print(f"Warning: Failed to cleanup pipeline: {str(e)}")

cleanup_pipeline()


def encode_prompt(
    pipe, text_encoders, tokenizers, prompt, device, num_images_per_prompt=1, do_classifier_free_guidance=True,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None, lora_scale=None, clip_skip=None,
):
    """
    Enhanced SDXL-compatible encode_prompt function that properly handles dual text encoders
    and pooled embeddings for SDXL models.
    """
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = tokenizers if tokenizers is not None else [pipe.tokenizer, pipe.tokenizer_2] if hasattr(pipe, 'tokenizer_2') else [pipe.tokenizer]
    text_encoders = text_encoders if text_encoders is not None else [pipe.text_encoder, pipe.text_encoder_2] if hasattr(pipe, 'text_encoder_2') else [pipe.text_encoder]

    if prompt_embeds is None:
        prompt_2 = prompt if hasattr(pipe, 'text_encoder_2') else None
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            encode_prompt_sdxl(
                text_encoders,
                tokenizers,
                prompt,
                prompt_2,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                clip_skip=clip_skip,
            )
        )

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def encode_prompt_sdxl(
    text_encoders,
    tokenizers,
    prompt,
    prompt_2,
    device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt=None,
    negative_prompt_2=None,
    clip_skip=None,
):
    """
    Encodes the prompt into text encoder hidden states for SDXL.
    """
    # textual inversion: process multi-vector tokens if necessary
    prompt_embeds_list = []
    prompts = [prompt, prompt_2] if prompt_2 else [prompt]
    
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        if prompt is None:
            prompt = ""
            
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        
        max_length = tokenizer.model_max_length
        
        # Get text inputs
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            print(f"The following part of your input was truncated because CLIP can only handle sequences up to {max_length} tokens: {removed_text}")

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        
        # We are only interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None
    if do_classifier_free_guidance and negative_prompt_2 is None:
        negative_prompt_2 = negative_prompt

    if do_classifier_free_guidance and negative_prompt is None:
        negative_prompt = ""
        negative_prompt_2 = ""

    # normalize embeddings
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds_list = []
        negative_prompts = [negative_prompt, negative_prompt_2] if negative_prompt_2 else [negative_prompt]
        
        for negative_prompt, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):
            if negative_prompt is None:
                negative_prompt = ""
                
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
            # We are only interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            
            if clip_skip is None:
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            else:
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-(clip_skip + 2)]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(negative_prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoders[0].dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    else:
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

    # Ensure pooled embeddings have correct dtype and device
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)
    if negative_pooled_prompt_embeds is not None:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def compute_time_ids(original_size, crops_coords_top_left, target_size, dtype, device, weight_dtype=None):
    """
    Compute time IDs for SDXL conditioning.
    """
    if weight_dtype is None:
        weight_dtype = dtype
        
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype, device=device)
    return add_time_ids


@tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def train_diffusion_lora():
    # Extract parameters from tlab_trainer
    args = tlab_trainer.params

    print("***** Running training *****")

    # Setup logging directory
    output_dir = args.get("output_dir", "sd-model-finetuned-lora")

    # Setup evaluation images directory
    job_id = tlab_trainer.params.job_id
    eval_images_dir = None
    eval_prompt = args.get("eval_prompt", "").strip()
    eval_steps = int(args.get("eval_steps", 1))
    
    if eval_prompt:
        eval_images_dir = Path(WORKSPACE_DIR) / "temp" / f"eval_images_{job_id}"
        eval_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Evaluation images will be saved to: {eval_images_dir}")
        
        # Add eval images directory to job data
        tlab_trainer.add_job_data("eval_images_dir", str(eval_images_dir))

    # Load dataset using tlab_trainer
    datasets_dict = tlab_trainer.load_dataset(["train"])
    dataset = datasets_dict["train"]

    # Model and tokenizer loading - use AutoPipeline for multi-architecture support
    pretrained_model_name_or_path = args.get("model_name")
    if args.get("model_path") is not None and args.get("model_path").strip() != "":
        pretrained_model_name_or_path = args.get("model_path")
    revision = args.get("revision", None)
    variant = args.get("variant", None)

    # Load pipeline to auto-detect architecture and get correct components
    print(f"Loading pipeline to detect model architecture: {pretrained_model_name_or_path}")
    temp_pipeline = AutoPipelineForText2Image.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        torch_dtype=torch.float32,  # Load in float32 initially for training
    )
    
    # Extract components from the loaded pipeline
    noise_scheduler = temp_pipeline.scheduler
    tokenizer = temp_pipeline.tokenizer
    text_encoder = temp_pipeline.text_encoder
    vae = temp_pipeline.vae
    unet = temp_pipeline.unet
    
    # Handle SDXL case with dual text encoders
    text_encoder_2 = getattr(temp_pipeline, 'text_encoder_2', None)
    tokenizer_2 = getattr(temp_pipeline, 'tokenizer_2', None)
    
    # Clean up temporary pipeline
    del temp_pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Model components loaded successfully: {pretrained_model_name_or_path}")
    print(f"Architecture detected - UNet type: {type(unet).__name__}")
    if text_encoder_2 is not None:
        print("Dual text encoder setup detected (likely SDXL)")
    print(f"Text encoder type: {type(text_encoder).__name__}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")

    # Freeze parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)

    # Mixed precision
    weight_dtype = torch.float32
    mixed_precision = args.get("mixed_precision", None)
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # LoRA config - adaptive target modules for different architectures
    unet_type = type(unet).__name__
    
    # Debug architecture detection
    print(f"Model path: {pretrained_model_name_or_path}")
    print(f"UNet type: {unet_type}")
    print(f"Has text_encoder_2: {text_encoder_2 is not None}")
    print(f"Has addition_embed_type: {hasattr(unet.config, 'addition_embed_type') if hasattr(unet, 'config') else 'No config'}")
    
    # Detect architecture based on multiple indicators
    is_sdxl = (text_encoder_2 is not None or 
               "sdxl" in pretrained_model_name_or_path.lower() or
               "stable-diffusion-xl" in pretrained_model_name_or_path.lower() or
               hasattr(unet.config, 'addition_embed_type'))
    
    is_sd3 = ("SD3" in unet_type or "MMDiT" in unet_type or
              "sd3" in pretrained_model_name_or_path.lower())
    
    is_flux = ("Flux" in unet_type or "FluxTransformer" in unet_type or
               "flux" in pretrained_model_name_or_path.lower())
    
    print(f"Architecture detection - SDXL: {is_sdxl}, SD3: {is_sd3}, Flux: {is_flux}")
    
    # Define target modules based on detected architecture
    if is_sdxl:
        # SDXL typically uses these modules
        target_modules = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"]
        architecture_name = "SDXL"
    elif is_sd3:
        # SD3 uses Multi-Modal DiT architecture
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        architecture_name = "SD3"
    elif is_flux:
        # Flux uses transformer-based architecture
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        architecture_name = "Flux"
    else:
        # Default SD 1.x targets
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        architecture_name = "SD 1.x"
    
    print(f"Using LoRA target modules for {architecture_name} ({unet_type}): {target_modules}")
    
    unet_lora_config = LoraConfig(
        r=int(args.get("lora_r", 4)),
        lora_alpha=int(args.get("lora_alpha", 4)),
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    if text_encoder_2 is not None:
        text_encoder_2.to(device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)
    if mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # Create evaluation pipeline function
    def generate_eval_image(epoch):
        if not eval_prompt or not eval_images_dir:
            return
            
        print(f"Generating evaluation image for epoch {epoch}...")
        
        # Create pipeline with current model state using AutoPipelineForText2Image
        pipeline = AutoPipelineForText2Image.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            torch_dtype=weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # Replace the UNet with our trained version to include LoRA weights
        pipeline.unet = unet
        pipeline = pipeline.to(device)
        
        # Generate image
        with torch.no_grad():
            image = pipeline(
                eval_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=int(args.get("resolution", 512)),
                width=int(args.get("resolution", 512)),
            ).images[0]
        
        # Save image
        image_path = eval_images_dir / f"epoch_{epoch}.png"
        image.save(image_path)
        
        print(f"Evaluation image saved to: {image_path}")

    # Data transforms
    interpolation = getattr(transforms.InterpolationMode, args.get("image_interpolation_mode", "lanczos").upper(), None)
    args["resolution"] = int(args.get("resolution", 512))
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.get("resolution", 512), interpolation=interpolation),
            transforms.CenterCrop(args.get("resolution", 512))
            if args.get("center_crop", False)
            else transforms.RandomCrop(args.get("resolution", 512)),
            transforms.RandomHorizontalFlip() if args.get("random_flip", False) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def tokenize_captions(examples, is_train=True):
        captions = []
        caption_column = args.get("caption_column", "text")
        trigger_word = args.get("trigger_word", "").strip()
        
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                processed_caption = caption
            elif isinstance(caption, (list, np.ndarray)):
                processed_caption = random.choice(caption) if is_train else caption[0]
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
            
            # Add trigger word to the beginning of the caption if specified
            if trigger_word:
                processed_caption = f"{trigger_word}, {processed_caption}"
            
            captions.append(processed_caption)
            
        # Primary tokenizer (always present)
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        result = {"input_ids": inputs.input_ids}
        
        # Secondary tokenizer for SDXL
        if tokenizer_2 is not None:
            inputs_2 = tokenizer_2(
                captions, max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            result["input_ids_2"] = inputs_2.input_ids
            
        return result

    image_column = args.get("image_column", "image")

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        
        # Enhanced preprocessing for SDXL with proper image metadata tracking
        processed_images = []
        original_sizes = []
        crop_coords_top_left = []
        target_sizes = []
        
        for image in images:
            original_size = image.size  # (width, height)
            original_sizes.append(original_size)
            
            # Apply transforms and track crop coordinates
            transformed_image = train_transforms(image)
            processed_images.append(transformed_image)
            
            # For SDXL, we need to track the crop coordinates
            # If using center crop, calculate the crop coordinates
            if args.get("center_crop", False):
                crop_size = args.get("resolution", 512)
                left = (original_size[0] - crop_size) // 2
                top = (original_size[1] - crop_size) // 2
                crop_coords_top_left.append((left, top))
            else:
                # For random crop, we'll use (0, 0) as we can't know the exact coordinates
                crop_coords_top_left.append((0, 0))
            
            # Target size is the final resolution
            target_size = (args.get("resolution", 512), args.get("resolution", 512))
            target_sizes.append(target_size)
        
        examples["pixel_values"] = processed_images
        examples["original_sizes"] = original_sizes
        examples["crop_coords_top_left"] = crop_coords_top_left 
        examples["target_sizes"] = target_sizes
        
        # Get tokenization results
        tokenization_results = tokenize_captions(examples)
        examples["input_ids"] = tokenization_results["input_ids"]
        
        # Add second input_ids for SDXL if present
        if "input_ids_2" in tokenization_results:
            examples["input_ids_2"] = tokenization_results["input_ids_2"]
            
        return examples

    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        
        batch = {"pixel_values": pixel_values, "input_ids": input_ids}
        
        # Add second input_ids for SDXL if present
        if "input_ids_2" in examples[0]:
            input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
            batch["input_ids_2"] = input_ids_2
        
        # Add SDXL-specific metadata for proper conditioning
        if "original_sizes" in examples[0]:
            batch["original_sizes"] = [example["original_sizes"] for example in examples]
            batch["crop_coords_top_left"] = [example["crop_coords_top_left"] for example in examples]
            batch["target_sizes"] = [example["target_sizes"] for example in examples]
            
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=int(args.get("train_batch_size", 16)),
        num_workers=int(args.get("dataloader_num_workers", 0)),
    )

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
    # max_train_steps = args.get("max_train_steps", None)
    max_train_steps = None
    if max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = int(max_train_steps)

    lr_scheduler = get_scheduler(
        args.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=int(args.get("lr_warmup_steps", 500)),
        num_training_steps=max_train_steps,
    )

    # Training loop
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Batch size = {args.get('train_batch_size', 16)}")
    print(f"  Total optimization steps = {max_train_steps}")
    if eval_prompt:
        print(f"  Evaluation prompt: '{eval_prompt}'")
        print(f"  Evaluation every {eval_steps} epoch(s)")
    
    args["noise_offset"] = int(args.get("noise_offset", 0))

    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            if args.get("noise_offset", 0):
                noise += args["noise_offset"] * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Enhanced text encoding - support both single and dual text encoders with proper SDXL handling
            if is_sdxl and text_encoder_2 is not None and "input_ids_2" in batch:
                # Use enhanced SDXL encode_prompt function for proper pooled embeddings
                text_encoders = [text_encoder, text_encoder_2]
                tokenizers = [tokenizer, tokenizer_2] if tokenizer_2 is not None else [tokenizer, tokenizer]
                
                # Create a temporary pipeline-like object for encode_prompt compatibility
                class TempPipeline:
                    def __init__(self, text_encoder, text_encoder_2, tokenizer, tokenizer_2):
                        self.text_encoder = text_encoder
                        self.text_encoder_2 = text_encoder_2
                        self.tokenizer = tokenizer
                        self.tokenizer_2 = tokenizer_2
                
                temp_pipe = TempPipeline(text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                
                # Decode input_ids back to text for encode_prompt function
                # This is needed because encode_prompt expects text input
                prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                
                # Use enhanced encode_prompt function
                encoder_hidden_states, _, pooled_prompt_embeds, _ = encode_prompt(
                    temp_pipe, text_encoders, tokenizers, prompts, device,
                    num_images_per_prompt=1, do_classifier_free_guidance=False
                )
            else:
                # Standard single text encoder approach
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]
                pooled_prompt_embeds = None
                
                # For SDXL with dual text encoders, handle dimension compatibility and concatenate
                if text_encoder_2 is not None and "input_ids_2" in batch:
                    encoder_hidden_states_2 = text_encoder_2(batch["input_ids_2"].to(device), return_dict=False)[0]
                    
                    # Handle dimension mismatch - ensure both tensors have the same number of dimensions
                    if encoder_hidden_states.dim() != encoder_hidden_states_2.dim():
                        # If one is 2D and the other is 3D, add a dimension to the 2D tensor
                        if encoder_hidden_states.dim() == 2 and encoder_hidden_states_2.dim() == 3:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        elif encoder_hidden_states.dim() == 3 and encoder_hidden_states_2.dim() == 2:
                            encoder_hidden_states_2 = encoder_hidden_states_2.unsqueeze(1)
                    
                    # Ensure sequence lengths match for concatenation
                    seq_len_1 = encoder_hidden_states.shape[1] if encoder_hidden_states.dim() == 3 else encoder_hidden_states.shape[0]
                    seq_len_2 = encoder_hidden_states_2.shape[1] if encoder_hidden_states_2.dim() == 3 else encoder_hidden_states_2.shape[0]
                    
                    if seq_len_1 != seq_len_2:
                        # Pad the shorter sequence to match the longer one
                        max_seq_len = max(seq_len_1, seq_len_2)
                        
                        if encoder_hidden_states.dim() == 3:
                            if encoder_hidden_states.shape[1] < max_seq_len:
                                pad_size = max_seq_len - encoder_hidden_states.shape[1]
                                padding = torch.zeros(encoder_hidden_states.shape[0], pad_size, encoder_hidden_states.shape[2], 
                                                    device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                                encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
                            
                            if encoder_hidden_states_2.shape[1] < max_seq_len:
                                pad_size = max_seq_len - encoder_hidden_states_2.shape[1]
                                padding = torch.zeros(encoder_hidden_states_2.shape[0], pad_size, encoder_hidden_states_2.shape[2], 
                                                    device=encoder_hidden_states_2.device, dtype=encoder_hidden_states_2.dtype)
                                encoder_hidden_states_2 = torch.cat([encoder_hidden_states_2, padding], dim=1)
                    
                    # Concatenate along the feature dimension (last dimension)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)

            # Loss target
            prediction_type = args.get("prediction_type", None)
            if prediction_type is not None:
                noise_scheduler.register_to_config(prediction_type=prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")            # Handle SDXL-specific conditioning parameters with proper metadata
            unet_kwargs = {
                "timestep": timesteps,
                "encoder_hidden_states": encoder_hidden_states,
                "return_dict": False
            }
            
            # SDXL requires additional conditioning kwargs with proper pooled embeddings and time_ids
            if is_sdxl:
                batch_size = noisy_latents.shape[0]
                
                # Use proper pooled embeddings if available, otherwise create dummy ones
                if pooled_prompt_embeds is not None:
                    text_embeds = pooled_prompt_embeds.repeat(batch_size, 1) if pooled_prompt_embeds.shape[0] == 1 else pooled_prompt_embeds
                else:
                    # Fallback to dummy embeddings for compatibility
                    text_embeds = torch.zeros(batch_size, 1280, device=device, dtype=weight_dtype)
                
                # Compute proper time_ids from actual image metadata if available
                if "original_sizes" in batch and "crop_coords_top_left" in batch and "target_sizes" in batch:
                    time_ids_list = []
                    for i in range(batch_size):
                        original_size = batch["original_sizes"][i]
                        crop_coords = batch["crop_coords_top_left"][i] 
                        target_size = batch["target_sizes"][i]
                        
                        # Compute time_ids for this sample
                        time_ids = compute_time_ids(
                            original_size, crop_coords, target_size,
                            dtype=weight_dtype, device=device, weight_dtype=weight_dtype
                        )
                        time_ids_list.append(time_ids)
                    
                    time_ids = torch.cat(time_ids_list, dim=0)
                else:
                    # Fallback to dummy time_ids for compatibility
                    resolution = int(args.get("resolution", 512))
                    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]] * batch_size, 
                                           device=device, dtype=weight_dtype)
                
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids
                }
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
            
            model_pred = unet(noisy_latents, **unet_kwargs)[0]

            if args.get("snr_gamma", None) is None or args["snr_gamma"] == "":
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args["snr_gamma"] * torch.ones_like(timesteps)], dim=1).min(dim=1)[
                    0
                ]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(lora_layers), float(args.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Progress reporting
                percent_complete = 100.0 * global_step / max_train_steps
                tlab_trainer.progress_update(percent_complete)
                tlab_trainer.log_metric("train/loss", loss.item(), global_step)
                tlab_trainer.log_metric("train/lr", lr_scheduler.get_last_lr()[0], global_step)

                if global_step >= max_train_steps:
                    break

        # Generate evaluation image at the end of epoch
        if eval_prompt and (epoch + 1) % eval_steps == 0:
            unet.eval()
            generate_eval_image(epoch + 1)
            unet.train()

        if global_step >= max_train_steps:
            break

    # Final evaluation image
    if eval_prompt:
        unet.eval()
        generate_eval_image("final")

    # Save LoRA weights using the proven working method that worked perfectly with SD 1.5
    unet = unet.to(torch.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    save_directory = args.get("adaptor_output_dir", output_dir)
    
    print(f"Saving LoRA weights to {save_directory}")
    import os
    os.makedirs(save_directory, exist_ok=True)
    
    # Primary method: Use the original working approach that was perfect for SD 1.5
    # Try architecture-specific save methods first, then fall back to universal methods
    saved_successfully = False
    
    # Method 1: Try the original SD 1.x approach that worked perfectly
    if not is_sdxl and not is_sd3 and not is_flux:
        try:
            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_directory,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            print(f"LoRA weights saved to {save_directory} using StableDiffusionPipeline.save_lora_weights (SD 1.x)")
            saved_successfully = True
        except Exception as e:
            print(f"Error with StableDiffusionPipeline.save_lora_weights: {e}")
    
    # Method 2: Try SDXL-specific save method
    if not saved_successfully and is_sdxl:
        try:
            from diffusers import StableDiffusionXLPipeline
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=save_directory,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            print(f"LoRA weights saved to {save_directory} using StableDiffusionXLPipeline.save_lora_weights (SDXL)")
            saved_successfully = True
        except Exception as e:
            print(f"Error with StableDiffusionXLPipeline.save_lora_weights: {e}")
    
    # Method 3: Try SD3-specific save method
    if not saved_successfully and is_sd3:
        try:
            # SD3 pipelines may have their own save method
            from diffusers import StableDiffusion3Pipeline
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=save_directory,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            print(f"LoRA weights saved to {save_directory} using StableDiffusion3Pipeline.save_lora_weights (SD3)")
            saved_successfully = True
        except Exception as e:
            print(f"Error with StableDiffusion3Pipeline.save_lora_weights: {e}")
    
    # Method 4: Try FLUX-specific save method
    if not saved_successfully and is_flux:
        try:
            # FLUX pipelines may have their own save method
            from diffusers import FluxPipeline
            FluxPipeline.save_lora_weights(
                save_directory=save_directory,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            print(f"LoRA weights saved to {save_directory} using FluxPipeline.save_lora_weights (FLUX)")
            saved_successfully = True
        except Exception as e:
            print(f"Error with FluxPipeline.save_lora_weights: {e}")
    
    # Method 5: Try the generic StableDiffusionPipeline method as fallback for all architectures
    if not saved_successfully:
        try:
            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_directory,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            print(f"LoRA weights saved to {save_directory} using StableDiffusionPipeline.save_lora_weights (generic fallback)")
            saved_successfully = True
        except Exception as e:
            print(f"Error with generic StableDiffusionPipeline.save_lora_weights: {e}")
    
    # Method 6: Direct safetensors save as universal fallback
    if not saved_successfully:
        try:
            from safetensors.torch import save_file
            save_file(unet_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.safetensors"))
            print(f"LoRA weights saved to {save_directory}/pytorch_lora_weights.safetensors using safetensors (universal fallback)")
            print(f"To load this LoRA, use: pipeline.load_lora_weights('{save_directory}', weight_name='pytorch_lora_weights.safetensors')")
            saved_successfully = True
        except ImportError:
            # Final fallback to standard PyTorch format
            torch.save(unet_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.bin"))
            print(f"LoRA weights saved to {save_directory}/pytorch_lora_weights.bin using PyTorch format (final fallback)")
            print(f"To load this LoRA, use: pipeline.load_lora_weights('{save_directory}', weight_name='pytorch_lora_weights.bin')")
            saved_successfully = True
        except Exception as e:
            print(f"Error saving LoRA weights with safetensors: {e}")
    
    if saved_successfully:
        print(f"LoRA weights successfully saved to {save_directory}")
    else:
        print(f"Failed to save LoRA weights to {save_directory}")


train_diffusion_lora()