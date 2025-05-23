import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers

from transformerlab.sdk.v1.train import tlab_trainer

check_min_version("0.34.0.dev0")


@tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def train_diffusion_lora():
    # Extract parameters from tlab_trainer
    args = tlab_trainer.params

    print("***** Running training *****")

    # Setup logging directory
    output_dir = args.get("output_dir", "sd-model-finetuned-lora")

    # Load dataset using tlab_trainer
    datasets_dict = tlab_trainer.load_dataset(["train"])
    dataset = datasets_dict["train"]

    # Model and tokenizer loading
    pretrained_model_name_or_path = args.get("model_name")
    if args.get("model_path") is not None and args.get("model_path").strip() != "":
        pretrained_model_name_or_path = args.get("model_path")
    revision = args.get("revision", None)
    variant = args.get("variant", None)

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )
    print(f"Model and tokenizer loaded successfully: {pretrained_model_name_or_path}")

    # Freeze parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Mixed precision
    weight_dtype = torch.float32
    mixed_precision = args.get("mixed_precision", None)
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # LoRA config
    unet_lora_config = LoraConfig(
        r=int(args.get("lora_r", 4)),
        lora_alpha=int(args.get("lora_alpha", 4)),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)
    if mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

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
            
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    image_column = args.get("image_column", "image")

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

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
    max_train_steps = args.get("max_train_steps", None)
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
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]

            # Loss target
            prediction_type = args.get("prediction_type", None)
            if prediction_type is not None:
                noise_scheduler.register_to_config(prediction_type=prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

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

        if global_step >= max_train_steps:
            break

    # Save LoRA weights
    unet = unet.to(torch.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    save_directory = args.get("adaptor_output_dir", output_dir)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_directory,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    print(f"LoRA weights saved to {save_directory}")


train_diffusion_lora()
