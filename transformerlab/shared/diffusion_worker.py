#!/usr/bin/env python3
"""
Multi-GPU diffusion worker script for handling FLUX and other large models
that cause CUDA OOM on single GPU.
"""

import argparse
import json
import os
import sys
import time
from PIL import Image
import torch
import random
import base64
from io import BytesIO

# Add the API directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, api_dir)

from diffusers import (  # noqa: E402
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionXLPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionLatentUpscalePipeline,
)


def setup_device_map(config):
    """Setup device mapping for multi-GPU usage"""
    if not torch.cuda.is_available():
        return "cpu"

    # # Set PyTorch memory allocation settings to reduce fragmentation and improve stability
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:1"

    # Enable CUDA debugging for better error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")
    if config.get("num_gpus") is not None:
        num_gpus = min(num_gpus, config["num_gpus"])
        print(f"Using up to {num_gpus} GPU(s) as per config")

    # Check memory usage on each GPU and avoid heavily used ones
    available_gpus = []
    for i in range(num_gpus):
        try:
            torch.cuda.set_device(i)

            # Test GPU health with a simple operation
            test_tensor = torch.tensor([1.0], device=f"cuda:{i}")
            result = test_tensor * 2
            del test_tensor, result
            torch.cuda.synchronize(i)

            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            free_memory = total_memory - reserved_memory
            usage_percent = (reserved_memory / total_memory) * 100

            print(
                f"GPU {i}: Total {total_memory:.1f}GB, Allocated {allocated_memory:.1f}GB, Reserved {reserved_memory:.1f}GB, Free {free_memory:.1f}GB, Usage {usage_percent:.1f}% [HEALTHY]"
            )

            # Only use GPUs with less than 50% memory usage and at least 8GB free
            if usage_percent < 50 and free_memory > 8:
                available_gpus.append(i)
            else:
                print(f"GPU {i} skipped due to high memory usage or insufficient free memory")

        except Exception as e:
            print(f"GPU {i}: UNHEALTHY - Failed health check: {e}")
            print(f"GPU {i} will be excluded from available devices")

    if not available_gpus:
        print("Warning: No GPUs with sufficient free memory found, will try GPU 0 anyway")
        if num_gpus == 1:
            return "cuda:0"
        else:
            return "auto"

    if len(available_gpus) == 1:
        print(f"Using single available GPU: {available_gpus[0]}")
        return f"cuda:{available_gpus[0]}"

    print(f"Using multi-GPU setup with available GPUs: {available_gpus}")
    # For multi-GPU, let accelerate handle the device mapping
    return "auto"


def load_pipeline_with_device_map(model_path, adaptor_path, is_img2img, is_inpainting, device):
    """Load pipeline with proper device mapping for multi-GPU"""

    # Common pipeline kwargs
    pipeline_kwargs = {
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        "safety_checker": None,
        "requires_safety_checker": False,
        "use_safetensors": True,  # Use safetensors format for better memory efficiency
    }

    # Add memory-efficient attention for large models
    if device != "cpu":
        pipeline_kwargs["enable_xformers_memory_efficient_attention"] = True

    # For multi-GPU setups, use device_map
    if device == "auto":
        pipeline_kwargs["device_map"] = "balanced"
        # Set realistic memory limits for FLUX and other large models
        # Leave some memory for CUDA overhead and other processes
        num_gpus = torch.cuda.device_count()
        memory_per_gpu = {}
        total_available_memory = 0

        for i in range(num_gpus):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            free_memory = total_memory - reserved_memory

            # Only allocate memory if GPU has sufficient free space
            if free_memory > 8:  # At least 8GB free
                # Use 70% of free memory to be conservative
                usable_memory = max(1, int(free_memory * 0.7))
                memory_per_gpu[i] = f"{usable_memory}GiB"
                total_available_memory += usable_memory
                print(f"GPU {i}: Total {total_memory:.1f}GB, Free {free_memory:.1f}GB, Allocating {usable_memory}GB")
            else:
                # Allocate minimal memory for GPUs with limited space
                memory_per_gpu[i] = "1GiB"
                print(f"GPU {i}: Total {total_memory:.1f}GB, Free {free_memory:.1f}GB, Allocating 1GB (limited)")

        print(f"Total memory allocated across GPUs: {total_available_memory}GB")
        pipeline_kwargs["max_memory"] = memory_per_gpu

        # If total available memory is too low, warn user
        if total_available_memory < 12:
            print(
                f"Warning: Only {total_available_memory}GB available across all GPUs. FLUX models typically need 12-24GB."
            )

    elif device.startswith("cuda:"):
        # For single GPU, check if it has enough memory
        gpu_id = int(device.split(":")[-1])
        torch.cuda.set_device(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        free_memory = total_memory - reserved_memory
        print(f"Single GPU {gpu_id}: Total {total_memory:.1f}GB, Free {free_memory:.1f}GB")

        if free_memory < 8:
            print(f"Warning: GPU {gpu_id} only has {free_memory:.1f}GB free. FLUX models typically need 12-24GB.")
            # Try to clear any cached memory
            torch.cuda.empty_cache()

    # Load appropriate pipeline
    if is_inpainting:
        pipe = AutoPipelineForInpainting.from_pretrained(model_path, **pipeline_kwargs)
        print(f"Loaded inpainting pipeline for model: {model_path}")
    elif is_img2img:
        pipe = AutoPipelineForImage2Image.from_pretrained(model_path, **pipeline_kwargs)
        print(f"Loaded image-to-image pipeline for model: {model_path}")
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model_path, **pipeline_kwargs)
        print(f"Loaded text-to-image pipeline for model: {model_path}")

    # Move to device if not using device_map
    if device != "auto":
        pipe = pipe.to(device)

    # Load LoRA adaptor if provided
    if adaptor_path and adaptor_path.strip():
        try:
            if not isinstance(pipe, StableDiffusionXLPipeline):
                pipe.load_lora_weights(adaptor_path)
            else:
                # Only for SDXL Pipelines because they use a different kind of UNet
                state_dict, network_alphas = pipe.lora_state_dict(adaptor_path, prefix=None)
                pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)

        except Exception as e:
            print(f"Warning: Failed to load LoRA adaptor '{adaptor_path}': {str(e)}")

    return pipe


def upscale_image_worker(image_path, prompt, upscale_factor, device):
    """Upscale an image using Stable Diffusion upscaler"""

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Setup upscale pipeline
    if upscale_factor == 2:
        model_name = "stabilityai/sd-x2-latent-upscaler"
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # For latent upscaler, resize the image
        width, height = image.size
        image = image.resize((width // 8 * 8, height // 8 * 8))

        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            guidance_scale=0,
        )
    else:
        model_name = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            guidance_scale=0,
        )

    if device != "auto":
        pipe = pipe.to(device)

    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Diffusion Worker")
    parser.add_argument("--config", required=True, help="Path to generation config JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated images")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    print(f"Worker {args.worker_id} starting generation...")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Clear CUDA cache at start to free any lingering memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        print("Cleared CUDA cache at startup")

    try:
        # Setup device
        device = setup_device_map(config)
        print(f"Using device: {device}")

        # Extract config parameters
        model_path = config["model"]
        adaptor_path = config.get("adaptor", "")
        is_img2img = config.get("is_img2img", False)
        is_inpainting = config.get("is_inpainting", False)
        prompt = config["prompt"]
        num_images = config.get("num_images", 1)
        num_inference_steps = config.get("num_inference_steps", 30)
        guidance_scale = config.get("guidance_scale", 7.5)
        seed = config.get("seed")
        eta = config.get("eta", 0.0)
        negative_prompt = config.get("negative_prompt", "")
        guidance_rescale = config.get("guidance_rescale", 0.0)
        clip_skip = config.get("clip_skip", 0)
        height = config.get("height", 0)
        width = config.get("width", 0)
        adaptor_scale = config.get("adaptor_scale", 1.0)
        strength = config.get("strength", 0.8)
        input_image_data = config.get("input_image", "")
        mask_image_data = config.get("mask_image", "")
        upscale = config.get("upscale", False)
        upscale_factor = config.get("upscale_factor", 4)

        # Set seed
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.manual_seed(seed)
        print(f"Using seed: {seed}")

        # Process input image for img2img and inpainting
        input_image_obj = None
        mask_image_obj = None

        if (is_img2img or is_inpainting) and input_image_data:
            image_data = base64.b64decode(input_image_data)
            input_image_obj = Image.open(BytesIO(image_data)).convert("RGB")
            print("Loaded input image for img2img/inpainting")

        if is_inpainting and mask_image_data:
            mask_data = base64.b64decode(mask_image_data)
            mask_image_obj = Image.open(BytesIO(mask_data)).convert("L")  # Convert to grayscale
            print("Loaded mask image for inpainting")

        # Load pipeline
        print("Loading pipeline...")

        # Clear any existing CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc

            gc.collect()
            print("Cleared CUDA cache before loading pipeline")

        start_time = time.time()
        pipe = load_pipeline_with_device_map(model_path, adaptor_path, is_img2img, is_inpainting, device)
        load_time = time.time() - start_time
        print(f"Pipeline loaded in {load_time:.2f}s")

        # Check memory usage after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(
                    f"GPU {i} after loading: Allocated {allocated:.1f}GB, Reserved {reserved:.1f}GB, Total {total:.1f}GB"
                )

        # Setup generation kwargs
        generation_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": num_images,
        }

        # Add image and mask for inpainting, or image and strength for img2img
        if is_inpainting and input_image_obj and mask_image_obj:
            generation_kwargs["image"] = input_image_obj
            generation_kwargs["mask_image"] = mask_image_obj
            if strength < 1.0:  # Only add strength if it's less than 1.0 for inpainting
                generation_kwargs["strength"] = strength
        elif is_img2img and input_image_obj:
            generation_kwargs["image"] = input_image_obj
            generation_kwargs["strength"] = strength

        if eta > 0.0:
            generation_kwargs["eta"] = eta

        # Add optional parameters
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt

        if guidance_rescale > 0.0:
            generation_kwargs["guidance_rescale"] = guidance_rescale

        if clip_skip > 0:
            generation_kwargs["clip_skip"] = clip_skip

        if height > 0 and width > 0:
            generation_kwargs["height"] = height
            generation_kwargs["width"] = width

        # Add LoRA scale if adaptor is being used
        if adaptor_path and adaptor_path.strip():
            generation_kwargs["cross_attention_kwargs"] = {"scale": adaptor_scale}

        # Generate images
        print("Starting image generation...")
        gen_start_time = time.time()

        # Enable memory efficient attention and other optimizations
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Could not enable xformers attention: {e}")

        # Only use CPU offload for single GPU setups, not multi-GPU device_map
        if device != "auto" and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload")
            except Exception as e:
                print(f"Could not enable CPU offload: {e}")
        elif device == "auto":
            print("Skipping CPU offload for multi-GPU device_map setup")

        # Set CUDA debugging environment variables for better error reporting
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        # Additional memory safety checks before generation
        if torch.cuda.is_available():
            try:
                # Force synchronization to catch any pending CUDA errors
                torch.cuda.synchronize()
                print("CUDA synchronization successful before generation")

                # Check for any pending CUDA errors
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    # Try a simple operation to verify GPU health
                    test_tensor = torch.tensor([1.0], device=f"cuda:{i}")
                    _ = test_tensor * 2
                    del test_tensor
                    print(f"GPU {i} health check passed")

            except RuntimeError as e:
                print(f"CUDA error detected before generation: {e}")
                raise
        try:
            with torch.inference_mode():
                result = pipe(**generation_kwargs)
                images = result.images
        except RuntimeError as e:
            if "illegal memory access" in str(e) or "CUDA error" in str(e):
                print(f"CUDA memory access error detected: {e}")
                print("This indicates potential GPU memory corruption or hardware issues.")
                print("Attempting recovery...")

                # Clear all CUDA cache and try to reset GPU state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    import gc

                    gc.collect()

                    # Try to reset CUDA context
                    try:
                        for i in range(torch.cuda.device_count()):
                            torch.cuda.set_device(i)
                            torch.cuda.reset_peak_memory_stats(i)
                        print("CUDA context reset attempted")
                    except Exception as reset_e:
                        print(f"Could not reset CUDA context: {reset_e}")

                # Re-raise the error with additional context
                raise RuntimeError(
                    f"CUDA illegal memory access: {str(e)}. "
                    f"This may indicate GPU hardware issues, driver problems, "
                    f"or memory corruption. Try: 1) Restart the process, "
                    f"2) Reduce model size/batch size, 3) Check GPU health, "
                    f"4) Update GPU drivers."
                ) from e
            else:
                # Re-raise other runtime errors as-is
                raise

        generation_time = time.time() - gen_start_time
        print(f"Generated {len(images)} images in {generation_time:.2f}s")

        # Clean up pipeline to free memory immediately
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc

            gc.collect()
            print("Cleaned up pipeline and cleared CUDA cache")

        # Save images
        os.makedirs(args.output_dir, exist_ok=True)
        image_paths = []

        for i, image in enumerate(images):
            image_path = os.path.join(args.output_dir, f"{i}.png")

            if upscale:
                print(f"Upscaling image {i + 1}/{len(images)}...")
                upscale_start = time.time()
                # Save the original image first for upscaling
                temp_path = os.path.join(args.output_dir, f"temp_{i}.png")
                image.save(temp_path, format="PNG")
                upscaled_image = upscale_image_worker(temp_path, prompt, upscale_factor, device)
                upscaled_image.save(image_path, format="PNG")
                # Clean up temp file
                os.remove(temp_path)
                upscale_time = time.time() - upscale_start
                print(f"Upscaled in {upscale_time:.2f}s")
            else:
                image.save(image_path, format="PNG")

            image_paths.append(image_path)
            print(f"Saved image: {image_path}")

        # Save result metadata
        result_data = {
            "success": True,
            "images": image_paths,
            "generation_time": generation_time,
            "load_time": load_time,
            "seed": seed,
            "num_images": len(images),
            "worker_id": args.worker_id,
        }

        result_path = os.path.join(args.output_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"Worker {args.worker_id} completed successfully")
        print(f"Results saved to: {result_path}")

    except RuntimeError as e:
        if "illegal memory access" in str(e) or "CUDA error" in str(e):
            print(f"Worker {args.worker_id} failed with CUDA hardware error: {str(e)}")

            # Provide detailed debugging information
            print("\n=== CUDA ERROR DEBUGGING INFO ===")
            print("This error indicates a serious CUDA problem, likely:")
            print("1. GPU hardware failure or instability")
            print("2. GPU driver issues or corruption")
            print("3. Memory corruption in CUDA context")
            print("4. Incompatible CUDA/PyTorch versions")
            print("5. Overheating or power supply issues")

            if torch.cuda.is_available():
                print("\nGPU Status:")
                for i in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        total = props.total_memory / (1024**3)
                        print(
                            f"GPU {i} ({props.name}): Allocated {allocated:.1f}GB, Reserved {reserved:.1f}GB, Total {total:.1f}GB"
                        )
                    except Exception as gpu_e:
                        print(f"GPU {i}: Error accessing GPU info - {gpu_e}")

            print("\nRecommended Actions:")
            print("1. Restart the entire process/container")
            print("2. Check GPU temperature and power supply")
            print("3. Update/reinstall GPU drivers")
            print("4. Run GPU memory test (e.g., gpu-burn)")
            print("5. Try with CPU-only mode")
            print("6. Reduce model complexity or batch size")
            print("7. Check for hardware conflicts")

            # Clean up any remaining GPU state
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    import gc

                    gc.collect()
                    print("Attempted CUDA cleanup")
                except Exception as cleanup_e:
                    print(f"CUDA cleanup failed: {cleanup_e}")

            # Save error result with specific CUDA error info
            result_data = {
                "success": False,
                "error": f"CUDA Hardware Error: {str(e)}",
                "error_type": "CUDA_ILLEGAL_MEMORY_ACCESS",
                "worker_id": args.worker_id,
                "recommendations": [
                    "Restart the entire process/container",
                    "Check GPU temperature and power supply",
                    "Update/reinstall GPU drivers",
                    "Run GPU memory test (e.g., gpu-burn)",
                    "Try with CPU-only mode",
                    "Reduce model complexity or batch size",
                    "Check for hardware conflicts",
                ],
            }

            os.makedirs(args.output_dir, exist_ok=True)
            result_path = os.path.join(args.output_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

            sys.exit(2)  # Different exit code for hardware errors
        else:
            # Handle other RuntimeErrors
            print(f"Worker {args.worker_id} failed with runtime error: {str(e)}")
            import traceback

            traceback.print_exc()

            result_data = {
                "success": False,
                "error": str(e),
                "error_type": "RUNTIME_ERROR",
                "worker_id": args.worker_id,
            }

            os.makedirs(args.output_dir, exist_ok=True)
            result_path = os.path.join(args.output_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

            sys.exit(1)

    except torch.cuda.OutOfMemoryError as e:
        print(f"Worker {args.worker_id} failed with CUDA OOM: {str(e)}")

        # Provide detailed memory information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
                reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
                free_memory = total_memory - reserved_memory
                print(
                    f"GPU {i} Memory Status: Total {total_memory:.1f}GB, Allocated {allocated_memory:.1f}GB, Reserved {reserved_memory:.1f}GB, Free {free_memory:.1f}GB"
                )

        print("Suggestions:")
        print("1. Reduce image resolution (height/width)")
        print("2. Reduce number of inference steps")
        print("3. Use a smaller model variant")

        # Clean up any remaining GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        # Save error result
        result_data = {
            "success": False,
            "error": f"CUDA Out of Memory: {str(e)}",
            "error_type": "OOM",
            "worker_id": args.worker_id,
            "suggestions": [
                "Reduce image resolution (height/width)",
                "Reduce number of inference steps",
                "Use a smaller model variant",
            ],
        }

        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)

        sys.exit(1)

    except Exception as e:
        print(f"Worker {args.worker_id} failed: {str(e)}")
        import traceback

        traceback.print_exc()

        # Save error result
        result_data = {"success": False, "error": str(e), "worker_id": args.worker_id}

        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
