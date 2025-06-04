from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import model_info
import base64
from fastapi.responses import FileResponse, JSONResponse
from io import BytesIO
import torch
import asyncio
import gc
from diffusers import (
    StableDiffusionUpscalePipeline,
    StableDiffusionLatentUpscalePipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionXLPipeline,
)
import threading
import os
import random
from werkzeug.utils import secure_filename
import json
import uuid
from datetime import datetime
import time
from typing import List
from PIL import Image
import shutil
import transformerlab.db as db
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify
import subprocess
import sys


router = APIRouter(prefix="/diffusion", tags=["diffusion"])

ALLOWED_TEXT2IMG_ARCHITECTURES = [
    "StableDiffusionPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusion3PAGPipeline",
    "StableDiffusionControlNetPAGPipeline",
    "StableDiffusionXLPAGPipeline",
    "StableDiffusionXLControlNetPAGPipeline",
    "FluxPipeline",
    "FluxControlPipeline",
    "FluxControlNetPipeline",
    "LuminaPipeline",
    "Lumina2Pipeline",
    "CogView3PlusPipeline",
    "CogView4Pipeline",
    "CogView4ControlPipeline",
    "IFPipeline",
    "HunyuanDiTPipeline",
    "HunyuanDiTPAGPipeline",
    "KandinskyCombinedPipeline",
    "KandinskyV22CombinedPipeline",
    "Kandinsky3Pipeline",
    "StableDiffusionControlNetPipeline",
    "StableDiffusionXLControlNetPipeline",
    "StableDiffusionXLControlNetUnionPipeline",
    "StableDiffusion3ControlNetPipeline",
    "WuerstchenCombinedPipeline",
    "StableCascadeCombinedPipeline",
    "LatentConsistencyModelPipeline",
    "PixArtAlphaPipeline",
    "PixArtSigmaPipeline",
    "SanaPipeline",
    "PixArtSigmaPAGPipeline",
    "AuraFlowPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusion3Img2ImgPipeline",
    "StableDiffusion3PAGImg2ImgPipeline",
    "IFImg2ImgPipeline",
    "KandinskyImg2ImgCombinedPipeline",
    "KandinskyV22Img2ImgCombinedPipeline",
    "Kandinsky3Img2ImgPipeline",
    "StableDiffusionControlNetImg2ImgPipeline",
    "StableDiffusionPAGImg2ImgPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
    "StableDiffusionXLControlNetUnionImg2ImgPipeline",
    "StableDiffusionXLPAGImg2ImgPipeline",
    "StableDiffusionXLControlNetPAGImg2ImgPipeline",
    "LatentConsistencyModelImg2ImgPipeline",
    "FluxImg2ImgPipeline",
    "FluxControlNetImg2ImgPipeline",
    "FluxControlImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusion3InpaintPipeline",
    "StableDiffusionPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionXLPipeline",
    "KandinskyInpaintPipeline",
    "KandinskyV22InpaintPipeline",
    "Kandinsky3Pipeline",
    "StableDiffusionControlNetInpaintPipeline",
    "StableDiffusionXLControlNetInpaintPipeline",
    "IFInpaintingPipeline",
    "IFPipeline",
]

# Allowed architectures for img2img pipelines
ALLOWED_IMG2IMG_ARCHITECTURES = [
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusion3Img2ImgPipeline",
    "StableDiffusion3PAGImg2ImgPipeline",
    "StableDiffusionPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusion3PAGPipeline",
    "IFImg2ImgPipeline",
    "IFPipeline",
    "KandinskyImg2ImgCombinedPipeline",
    "KandinskyCombinedPipeline",
    "KandinskyV22CombinedPipeline",
    "KandinskyV22Img2ImgCombinedPipeline",
    "Kandinsky3Img2ImgPipeline",
    "Kandinsky3Pipeline",
    "StableDiffusionControlNetImg2ImgPipeline",
    "StableDiffusionControlNetPipeline",
    "StableDiffusionPAGImg2ImgPipeline",
    "StableDiffusionPAGPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
    "StableDiffusionXLControlNetPipeline",
    "StableDiffusionXLControlNetUnionImg2ImgPipeline",
    "StableDiffusionXLControlNetUnionPipeline",
    "StableDiffusionXLPAGImg2ImgPipeline",
    "StableDiffusionXLPAGPipeline",
    "StableDiffusionXLControlNetPAGImg2ImgPipeline",
    "StableDiffusionXLControlNetPAGPipeline",
    "LatentConsistencyModelImg2ImgPipeline",
    "LatentConsistencyModelPipeline",
]

# Allowed architectures for inpainting pipelines
ALLOWED_INPAINTING_ARCHITECTURES = [
    "StableDiffusionInpaintPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusion3InpaintPipeline",
    "StableDiffusionPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionXLPipeline",
    "KandinskyInpaintPipeline",
    "KandinskyV22InpaintPipeline",
    "Kandinsky3Pipeline",
    "StableDiffusionControlNetInpaintPipeline",
    "StableDiffusionXLControlNetInpaintPipeline",
    "IFInpaintingPipeline",
    "IFPipeline",
]


# Fixed upscaling models
UPSCALE_MODEL_STANDARD = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALE_MODEL_LATENT = "stabilityai/sd-x2-latent-upscaler"


# Request schema for image generation
class DiffusionRequest(BaseModel):
    model: str
    prompt: str = ""
    adaptor: str = ""
    use_multi_gpu: bool = False
    enable_sharding: bool = True
    adaptor_scale: float = 1.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None
    upscale: bool = False
    upscale_factor: int = 4
    num_images: int = 1

    @property
    def validated_num_images(self) -> int:
        """Ensure num_images is within reasonable bounds"""
        return max(1, min(self.num_images, 8))

    # Negative prompting
    negative_prompt: str = ""
    # Advanced guidance control
    eta: float = 0.0
    clip_skip: int = 0
    guidance_rescale: float = 0.0
    height: int = 0
    width: int = 0
    # Image-to-image specific fields
    input_image: str = ""  # Base64 encoded input image
    strength: float = 0.8  # Denoising strength for img2img (0.0 = no change, 1.0 = full generation)
    is_img2img: bool = False  # Whether this is an img2img generation
    # Inpainting specific fields
    mask_image: str = ""  # Base64 encoded mask image for inpainting
    is_inpainting: bool = False  # Whether this is an inpainting generation


# Response schema for history
class ImageHistoryItem(BaseModel):
    id: str
    model: str
    prompt: str
    adaptor: str
    adaptor_scale: float
    num_inference_steps: int
    guidance_scale: float
    seed: int | None
    image_path: str
    timestamp: str
    upscaled: bool = False
    upscale_factor: int = 1
    negative_prompt: str = ""
    eta: float = 0.0
    clip_skip: int = 0
    guidance_rescale: float = 0.0
    height: int = 0
    width: int = 0
    generation_time: float = 0.0
    num_images: int = 1
    # Image-to-image specific fields
    input_image_path: str = ""  # Path to input image (for img2img)
    strength: float = 0.8  # Denoising strength used
    is_img2img: bool = False  # Whether this was an img2img generation
    # Inpainting specific fields
    mask_image_path: str = ""  # Path to mask image (for inpainting)
    is_inpainting: bool = False  # Whether this was an inpainting generation


class HistoryResponse(BaseModel):
    images: List[ImageHistoryItem]
    total: int


class CreateDatasetRequest(BaseModel):
    dataset_name: str
    image_ids: List[str]
    description: str = ""
    include_metadata: bool = True


# Global cache for loaded pipelines
# _PIPELINES: dict = {}
_PIPELINES_LOCK = threading.Lock()

# History file path
HISTORY_FILE = "history.json"


def cleanup_pipeline(pipe=None):
    """Clean up pipeline to free VRAM"""
    try:
        if pipe is not None:
            del pipe

        # Force garbage collection and clear CUDA cache
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        print(f"Warning: Failed to cleanup pipeline: {str(e)}")


def get_diffusion_dir():
    """Get the diffusion directory path"""
    return os.path.join(os.environ.get("_TFL_WORKSPACE_DIR", ""), "diffusion")


def get_images_dir():
    """Get the images directory path"""
    return os.path.join(get_diffusion_dir(), "images")


def get_history_file_path():
    """Get the history file path"""
    # Create a history file in the diffusion directory if it doesn't exist
    return os.path.join(get_diffusion_dir(), HISTORY_FILE)


def ensure_directories():
    """Ensure diffusion and images directories exist"""
    diffusion_dir = get_diffusion_dir()
    images_dir = get_images_dir()
    history_file_path = get_history_file_path()

    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    if not os.path.exists(history_file_path):
        with open(history_file_path, "a"):
            # Create the history file if it doesn't exist
            pass


def save_to_history(item: ImageHistoryItem):
    """Save an image generation record to history"""
    ensure_directories()
    history_file = get_history_file_path()

    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []

    # Add new item to the beginning of the list
    history.insert(0, item.model_dump())

    # Save updated history
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def load_history(limit: int = 50, offset: int = 0) -> HistoryResponse:
    """Load image generation history"""
    history_file = get_history_file_path()

    if not os.path.exists(history_file):
        return HistoryResponse(images=[], total=0)

    try:
        with open(history_file, "r") as f:
            history = json.load(f)

        total = len(history)
        paginated_history = history[offset : offset + limit]

        # Convert to ImageHistoryItem objects
        items = [ImageHistoryItem(**item) for item in paginated_history]

        return HistoryResponse(images=items, total=total)
    except (json.JSONDecodeError, FileNotFoundError):
        return HistoryResponse(images=[], total=0)


def find_image_by_id(image_id: str) -> ImageHistoryItem | None:
    """Find a specific image by ID without loading all history"""
    history_file = get_history_file_path()

    if not os.path.exists(history_file):
        return None

    try:
        with open(history_file, "r") as f:
            history = json.load(f)

        # Search for the specific image ID
        for item in history:
            if item.get("id") == image_id:
                return ImageHistoryItem(**item)

        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_pipeline_key(model: str, adaptor: str = "", is_img2img: bool = False, is_inpainting: bool = False) -> str:
    """Generate cache key for model + adaptor + pipeline type combination"""
    if is_inpainting:
        pipeline_type = "inpainting"
    elif is_img2img:
        pipeline_type = "img2img"
    else:
        pipeline_type = "txt2img"
    base_key = f"{model}::{adaptor}" if adaptor else model
    return f"{base_key}::{pipeline_type}"


def get_pipeline(
    model: str, adaptor: str = "", device: str = "cuda", is_img2img: bool = False, is_inpainting: bool = False
):
    # cache_key = get_pipeline_key(model, adaptor, is_img2img, is_inpainting)

    with _PIPELINES_LOCK:
        # if cache_key in _PIPELINES:
        #     return _PIPELINES[cache_key]

        # Load appropriate pipeline based on type
        if is_inpainting:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            print(f"Loaded inpainting pipeline for model: {model}")
        elif is_img2img:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            print(f"Loaded image-to-image pipeline for model: {model}")
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            print(f"Loaded text-to-image pipeline for model: {model} with dtype {pipe.dtype}")
        pipe = pipe.to(device)

        # Load LoRA adaptor if provided - same code for local and HF Hub!
        if adaptor and adaptor.strip():
            try:
                adaptor_dir = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR"), "adaptors", secure_filename(model))
                adaptor_path = os.path.join(adaptor_dir, secure_filename(adaptor))
                if os.path.exists(adaptor_path):
                    if not isinstance(pipe, StableDiffusionXLPipeline):
                        pipe.load_lora_weights(adaptor_path)
                    else:
                        # Only for SDXL Pipelines because they use a different kind of UNet
                        state_dict, network_alphas = pipe.lora_state_dict(adaptor_path, prefix=None)
                        pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)
                    print(f"Loaded LoRA adaptor: {adaptor}")
                else:
                    print(f"Error: Adaptor file not found at {adaptor_path}")
                    raise FileNotFoundError(f"Adaptor file not found: {adaptor_path}")
            except Exception as e:
                print(f"Warning: Failed to load LoRA adaptor '{adaptor}'")
                print(f"Error: {str(e)}")
                # Continue without LoRA rather than failing

        # _PIPELINES[cache_key] = pipe
        return pipe


def get_upscale_pipeline(upscale_factor: int = 4, device: str = "cuda"):
    """Get the appropriate upscaling pipeline based on the factor"""
    # cache_key = f"upscale_{upscale_factor}"

    with _PIPELINES_LOCK:
        # if cache_key in _PIPELINES:
        #     return _PIPELINES[cache_key]

        if upscale_factor == 2:
            # Use latent upscaler for 2x
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
                UPSCALE_MODEL_LATENT,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            # Use standard upscaler for 4x (default)
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                UPSCALE_MODEL_STANDARD,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

        pipe = pipe.to(device)
        # _PIPELINES[cache_key] = pipe
        return pipe


def upscale_image(image: Image.Image, prompt: str, upscale_factor: int = 4, device: str = "cuda"):
    """Upscale an image using Stable Diffusion upscaler"""
    upscale_pipe = get_upscale_pipeline(upscale_factor, device)

    try:
        if upscale_factor == 2:
            # For latent upscaler, we need to resize the image first
            # The latent upscaler expects a specific size
            width, height = image.size
            # Resize to be compatible with latent upscaler
            image = image.resize((width // 8 * 8, height // 8 * 8))

            result = upscale_pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                guidance_scale=0,
            )
        else:
            # For standard 4x upscaler
            result = upscale_pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                guidance_scale=0,
            )

        return result.images[0]
    finally:
        # Clean up the upscale pipeline to free VRAM
        cleanup_pipeline(upscale_pipe)


@router.post("/generate", summary="Generate image with Stable Diffusion")
async def generate_image(request: DiffusionRequest):
    try:
        # Validate num_images parameter
        if request.num_images < 1 or request.num_images > 10:
            raise HTTPException(status_code=400, detail="num_images must be between 1 and 10")

        # Generate unique ID and timestamp
        generation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Create folder for images
        ensure_directories()
        images_folder = os.path.join(get_images_dir(), generation_id)
        os.makedirs(images_folder, exist_ok=True)

        # Determine pipeline type based on flags and provided images
        is_inpainting = request.is_inpainting or (
            bool(request.input_image.strip()) and bool(request.mask_image.strip())
        )
        is_img2img = request.is_img2img or (bool(request.input_image.strip()) and not is_inpainting)

        # Process input image and mask if provided
        input_image_obj = None
        mask_image_obj = None
        input_image_path = ""
        mask_image_path = ""

        if is_inpainting or is_img2img:
            try:
                # Decode base64 input image
                image_data = base64.b64decode(request.input_image)
                input_image_obj = Image.open(BytesIO(image_data)).convert("RGB")

                # Save input image for history
                ensure_directories()
                input_image_filename = f"input_{str(uuid.uuid4())}.png"
                input_image_path = os.path.join(get_images_dir(), input_image_filename)
                input_image_obj.save(input_image_path, format="PNG")
                print(f"Input image saved: {input_image_path}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid input image: {str(e)}")

        if is_inpainting:
            try:
                # Decode base64 mask image
                mask_data = base64.b64decode(request.mask_image)
                mask_image_obj = Image.open(BytesIO(mask_data)).convert("L")  # Convert to grayscale

                # Save mask image for history
                ensure_directories()
                mask_image_filename = f"mask_{str(uuid.uuid4())}.png"
                mask_image_path = os.path.join(get_images_dir(), mask_image_filename)
                mask_image_obj.save(mask_image_path, format="PNG")
                print(f"Mask image saved: {mask_image_path}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid mask image: {str(e)}")

        # Check if we should use multi-GPU approach
        if should_use_diffusion_worker(request.model):
            print(f"Using Diffusion Worker subprocess approach for model: {request.model}")
            use_single_gpu = False
            try:
                result = await run_multi_gpu_generation(
                    request, generation_id, images_folder, input_image_path, mask_image_path, is_img2img, is_inpainting
                )

                images = []
                for img_path in result["images"]:
                    images.append(Image.open(img_path))

                total_generation_time = result["generation_time"]
                seed = result["seed"]

                # Get dimensions from the first image
                first_image = images[0]
                actual_height = request.height if request.height > 0 else first_image.height
                actual_width = request.width if request.width > 0 else first_image.width

            except Exception as e:
                print(f"Multi-GPU generation failed, falling back to single GPU: {str(e)}")
                # Fall back to single GPU approach
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cleanup_pipeline()

        else:
            use_single_gpu = True

        if use_single_gpu:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cleanup_pipeline()  # Clean up any previous pipelines
            pipe = get_pipeline(
                request.model, request.adaptor, device=device, is_img2img=is_img2img, is_inpainting=is_inpainting
            )

            # Set seed - use provided seed or generate a random one
            if request.seed is None or request.seed < 0:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = request.seed

            generator = torch.manual_seed(seed)

            # Process input image and mask for single GPU path
            input_image_obj = None
            mask_image_obj = None
            if is_inpainting or is_img2img:
                try:
                    # Decode base64 input image
                    image_data = base64.b64decode(request.input_image)
                    input_image_obj = Image.open(BytesIO(image_data)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid input image: {str(e)}")

            if is_inpainting:
                try:
                    # Decode base64 mask image
                    mask_data = base64.b64decode(request.mask_image)
                    mask_image_obj = Image.open(BytesIO(mask_data)).convert("L")  # Convert to grayscale
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid mask image: {str(e)}")

            # Run in thread to avoid blocking event loop
            def run_pipe():
                generation_kwargs = {
                    "prompt": request.prompt,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "generator": generator,
                    "num_images_per_prompt": request.num_images,  # Generate multiple images
                }

                # Add image and mask for inpainting
                if is_inpainting:
                    generation_kwargs["image"] = input_image_obj
                    generation_kwargs["mask_image"] = mask_image_obj
                    generation_kwargs["strength"] = request.strength
                # Add image and strength for img2img
                elif is_img2img:
                    generation_kwargs["image"] = input_image_obj
                    generation_kwargs["strength"] = request.strength

                # Add negative prompt if provided
                if request.negative_prompt:
                    generation_kwargs["negative_prompt"] = request.negative_prompt

                if request.eta > 0.0:
                    generation_kwargs["eta"] = request.eta

                # Add guidance rescale if provided
                if request.guidance_rescale > 0.0:
                    generation_kwargs["guidance_rescale"] = request.guidance_rescale

                # Add clip skip if provided (requires scheduler support)
                if request.clip_skip > 0:
                    generation_kwargs["clip_skip"] = request.clip_skip

                # Set height and width if specified
                if request.height > 0 and request.width > 0:
                    generation_kwargs["height"] = request.height
                    generation_kwargs["width"] = request.width

                # Add LoRA scale if adaptor is being used
                if request.adaptor and request.adaptor.strip():
                    generation_kwargs["cross_attention_kwargs"] = {"scale": request.adaptor_scale}

                result = pipe(**generation_kwargs)
                return result.images  # Return all images

            # Time the main generation
            generation_start = time.time()
            print("Starting image generation...")
            images = await asyncio.get_event_loop().run_in_executor(None, run_pipe)
            generation_time = time.time() - generation_start

            # Clean up the main pipeline to free VRAM
            cleanup_pipeline(pipe)

            # Get dimensions from the first image
            first_image = images[0]
            actual_height = request.height if request.height > 0 else first_image.height
            actual_width = request.width if request.width > 0 else first_image.width

            total_generation_time = generation_time

        # Apply upscaling if requested (for both paths)
        if request.upscale:
            print(f"Upscaling {len(images)} images with factor {request.upscale_factor}x")

            def run_upscale():
                upscaled_images = []
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for i, image in enumerate(images):
                    print(f"Upscaling image {i + 1}/{len(images)}")
                    upscaled_image = upscale_image(image, request.prompt, request.upscale_factor, device)
                    upscaled_images.append(upscaled_image)
                return upscaled_images

            upscale_start = time.time()
            images = await asyncio.get_event_loop().run_in_executor(None, run_upscale)
            upscale_time = time.time() - upscale_start
            total_generation_time += upscale_time
            print(
                f"Generation took {total_generation_time - upscale_time:.2f}s, upscaling took {upscale_time:.2f}s, total: {total_generation_time:.2f}s"
            )
        else:
            print(f"Generation took {total_generation_time:.2f}s")

        # Save images to the folder (for single GPU path, multi-GPU already saved)
        if use_single_gpu:
            for i, image in enumerate(images):
                image_filename = f"{i}.png"
                image_path = os.path.join(images_folder, image_filename)
                image.save(image_path, format="PNG")

        # Get dimensions from the first image
        first_image = images[0]
        actual_height = request.height if request.height > 0 else first_image.height
        actual_width = request.width if request.width > 0 else first_image.width

        # Save to history
        history_item = ImageHistoryItem(
            id=generation_id,
            model=request.model,
            prompt=request.prompt,
            adaptor=request.adaptor,
            adaptor_scale=request.adaptor_scale,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=seed,
            image_path=images_folder,  # Now pointing to the folder
            timestamp=timestamp,
            upscaled=request.upscale,
            upscale_factor=request.upscale_factor if request.upscale else 1,
            negative_prompt=request.negative_prompt,
            eta=request.eta,
            clip_skip=request.clip_skip,
            guidance_rescale=request.guidance_rescale,
            height=actual_height,
            width=actual_width,
            generation_time=total_generation_time,
            num_images=len(images),  # Store the number of images generated
            # Image-to-image specific fields
            input_image_path=input_image_path,
            strength=request.strength if (is_img2img or is_inpainting) else 0.8,
            is_img2img=is_img2img,
            # Inpainting specific fields
            mask_image_path=mask_image_path,
            is_inpainting=is_inpainting,
        )
        save_to_history(history_item)

        # Return metadata (no more base64 images)
        return JSONResponse(
            content={
                "id": generation_id,
                "prompt": request.prompt,
                "adaptor": request.adaptor,
                "adaptor_scale": request.adaptor_scale,
                "image_folder": images_folder,
                "num_images": len(images),
                "timestamp": timestamp,
                "generation_time": total_generation_time,
                "error_code": 0,
            }
        )
    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.post("/is_valid_diffusion_model", summary="Check if model is Stable Diffusion")
async def is_valid_diffusion(request: DiffusionRequest):
    """
    Returns {"is_valid_diffusion_model": True/False, "reason": "..."}
    """
    model_id = request.model
    if not model_id or model_id.strip() == "":
        return {"is_valid_diffusion_model": False, "reason": "Model ID is empty"}
    try:
        info = model_info(model_id)
        config = getattr(info, "config", {})
        diffusers_config = config.get("diffusers", {})
        architectures = diffusers_config.get("_class_name", "")
        if isinstance(architectures, str):
            architectures = [architectures]

        if request.is_inpainting:
            # First check if it's already an inpainting-specific architecture
            if any(a in ALLOWED_INPAINTING_ARCHITECTURES for a in architectures):
                return {"is_valid_diffusion_model": True, "reason": "Architecture matches allowed SD inpainting"}

            # Then check if we can derive an inpainting pipeline from a text2img architecture
            # This follows the same logic as diffusers AutoPipelineForInpainting
            for arch in architectures:
                if arch in ALLOWED_TEXT2IMG_ARCHITECTURES and 'flux' not in arch.lower():
                    return {
                        "is_valid_diffusion_model": True,
                        "reason": f"Text2img architecture {arch} can be used for inpainting",
                    }
        elif request.is_img2img:
            # Check if this is an img2img model
            if any(a in ALLOWED_IMG2IMG_ARCHITECTURES for a in architectures):
                return {"is_valid_diffusion_model": True, "reason": "Architecture matches allowed SD img2img"}
        else:
            if any(a in ALLOWED_TEXT2IMG_ARCHITECTURES for a in architectures):
                return {"is_valid_diffusion_model": True, "reason": "Architecture matches allowed SD"}

        return {"is_valid_diffusion_model": False, "reason": "No SD indicators found"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found or error: {str(e)}")


@router.get("/history", summary="Get image generation history", response_model=HistoryResponse)
async def get_history(limit: int = 50, offset: int = 0):
    """
    Get paginated history of generated images

    Args:
        limit: Maximum number of items to return (default 50)
        offset: Number of items to skip (default 0)

    Returns:
        HistoryResponse with list of images and total count
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Limit must be greater than 1")
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")

    return load_history(limit=limit, offset=offset)


@router.get("/history/{image_id}", summary="Get the actual image by ID")
async def get_image_by_id(image_id: str, index: int = 0, input_image: bool = False, mask_image: bool = False):
    """
    Get an image from history by ID and index

    Args:
        image_id: The unique ID of the image set
        index: The index of the image in the set (default 0)
        input_image: Whether to return the input image instead of generated image
        mask_image: Whether to return the mask image instead of generated image
    """
    # Use the efficient function to find the specific image
    image_item = find_image_by_id(image_id)

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Determine which image to return based on parameters
    if mask_image:
        # Return the mask image if requested and available
        if not image_item.mask_image_path or not image_item.mask_image_path.strip():
            raise HTTPException(status_code=404, detail=f"No mask image found for image ID {image_id}")
        image_path = image_item.mask_image_path
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Mask image file not found at {image_path}")
    elif input_image:
        # Return the input image if requested and available
        if not image_item.input_image_path or not image_item.input_image_path.strip():
            raise HTTPException(status_code=404, detail=f"No input image found for image ID {image_id}")

        image_path = image_item.input_image_path
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Input image file not found at {image_path}")
    else:
        # Return the generated output image (default behavior)
        # Check if image_path is a folder (new format) or a file (old format)
        if os.path.isdir(image_item.image_path):
            # New format: folder with numbered images
            if index < 0 or index >= (image_item.num_images if hasattr(image_item, "num_images") else 1):
                raise HTTPException(
                    status_code=404,
                    detail=f"Image index {index} out of range. Available: 0-{(image_item.num_images if hasattr(image_item, 'num_images') else 1) - 1}",
                )

            # image_path = os.path.join(image_item.image_path, f"{index}.png")
            # image_path = os.path.normpath(os.path.join(image_item.image_path, f"{index}.png"))
            # Sanitize the filename and construct the path
            sanitized_filename = secure_filename(f"{index}.png")
            image_path = os.path.normpath(os.path.join(image_item.image_path, sanitized_filename))
            expected_directory = os.path.abspath(image_item.image_path)
            # Ensure the normalized path is within the expected directory
            if (
                not image_path.startswith(expected_directory)
                or not os.path.commonpath([expected_directory, image_path]) == expected_directory
            ):
                raise HTTPException(status_code=400, detail="Invalid image path")
        else:
            # Old format: single image file
            if index != 0:
                raise HTTPException(status_code=404, detail="Only index 0 available for this image set")
            image_path = os.path.normpath(image_item.image_path)

            # # Ensure the normalized path is within the expected directory
            # if not image_path.startswith(os.path.abspath(image_item.image_path)):
            expected_directory = os.path.abspath(image_item.image_path)
            if (
                not image_path.startswith(expected_directory)
                or not os.path.commonpath([expected_directory, image_path]) == expected_directory
            ):
                raise HTTPException(status_code=400, detail="Invalid image path")

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found at {image_path}")

    return FileResponse(image_path)


@router.get("/history/{image_id}/info", summary="Get image metadata by ID")
async def get_image_info_by_id(image_id: str):
    """
    Get metadata for a specific image set by its ID

    Args:
        image_id: The unique ID of the image set

    Returns:
        Image metadata including number of images available
    """
    # Use the efficient function to find the specific image
    image_item = find_image_by_id(image_id)

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Check if image folder/file exists
    if not os.path.exists(image_item.image_path):
        raise HTTPException(status_code=404, detail=f"Image path not found at {image_item.image_path}")

    # Determine number of images available
    num_images = 1  # Default for old format
    if os.path.isdir(image_item.image_path):
        # Count PNG files in the directory
        png_files = [
            f for f in os.listdir(image_item.image_path) if f.endswith(".png") and f.replace(".png", "").isdigit()
        ]
        num_images = len(png_files)

    # Update the metadata to include actual number of images
    metadata = image_item.model_dump()
    metadata["num_images"] = num_images

    return JSONResponse(content={"id": image_item.id, "metadata": metadata})


@router.get("/history/{image_id}/count", summary="Get image count for an image set")
async def get_image_count(image_id: str):
    """
    Get the number of images available for a given image_id

    Args:
        image_id: The unique ID of the image set

    Returns:
        Number of images available
    """
    # Use the efficient function to find the specific image
    image_item = find_image_by_id(image_id)

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Check if image folder/file exists
    if not os.path.exists(image_item.image_path):
        raise HTTPException(status_code=404, detail=f"Image path not found at {image_item.image_path}")

    # Determine number of images available
    num_images = 1  # Default for old format
    if os.path.isdir(image_item.image_path):
        # Count PNG files in the directory
        png_files = [
            f for f in os.listdir(image_item.image_path) if f.endswith(".png") and f.replace(".png", "").isdigit()
        ]
        num_images = len(png_files)

    return JSONResponse(content={"id": image_id, "num_images": num_images})


@router.get("/history/{image_id}/all", summary="Get all images for an image set as a zip file")
async def get_all_images(image_id: str):
    """
    Get all images for a given image_id as a zip file

    Args:
        image_id: The unique ID of the image set

    Returns:
        Zip file containing all images
    """
    import zipfile
    import tempfile

    # Use the efficient function to find the specific image
    image_item = find_image_by_id(image_id)

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Check if image folder/file exists
    if not os.path.exists(image_item.image_path):
        raise HTTPException(status_code=404, detail=f"Image path not found at {image_item.image_path}")

    # Create a temporary zip file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isdir(image_item.image_path):
                # New format: add all PNG files from the directory
                for filename in os.listdir(image_item.image_path):
                    if filename.endswith(".png") and filename.replace(".png", "").isdigit():
                        file_path = os.path.join(image_item.image_path, filename)
                        zipf.write(file_path, filename)
            else:
                # Old format: add the single file
                filename = os.path.basename(image_item.image_path)
                zipf.write(image_item.image_path, filename)

        return FileResponse(
            temp_zip.name,
            media_type="application/zip",
            filename=f"images_{image_id}.zip",
            headers={"Content-Disposition": f"attachment; filename=images_{image_id}.zip"},
        )
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_zip.name):
            os.unlink(temp_zip.name)
        raise HTTPException(status_code=500, detail=f"Failed to create zip file: {str(e)}")


@router.delete("/history/{image_id}", summary="Delete image from history")
async def delete_image_from_history(image_id: str):
    """
    Delete a specific image set from history and remove the image files

    Args:
        image_id: The unique ID of the image set to delete
    """
    history_file = get_history_file_path()

    if not os.path.exists(history_file):
        raise HTTPException(status_code=404, detail="No history found")

    try:
        # Load current history
        with open(history_file, "r") as f:
            history = json.load(f)

        # Find and remove the item
        item_to_remove = None
        updated_history = []
        for item in history:
            if item["id"] == image_id:
                item_to_remove = item
            else:
                updated_history.append(item)

        if not item_to_remove:
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

        # Remove image files/folder
        image_path = item_to_remove["image_path"]
        if os.path.exists(image_path):
            if os.path.isdir(image_path):
                # New format: remove entire folder
                shutil.rmtree(image_path)
            else:
                # Old format: remove single file
                os.remove(image_path)

        # Remove input image if it exists
        if item_to_remove.get("input_image_path") and os.path.exists(item_to_remove["input_image_path"]):
            os.remove(item_to_remove["input_image_path"])

        # Save updated history
        with open(history_file, "w") as f:
            json.dump(updated_history, f, indent=2)

        return JSONResponse(
            content={"message": f"Image set {image_id} deleted successfully", "deleted_item": item_to_remove}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


@router.delete("/history", summary="Clear all history")
async def clear_history():
    """
    Clear all image generation history and remove all image files
    """
    try:
        history_file = get_history_file_path()
        images_dir = get_images_dir()

        # Load current history to get image paths
        deleted_count = 0
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)

            # Remove all image files/folders
            for item in history:
                image_path = item["image_path"]
                if os.path.exists(image_path):
                    if os.path.isdir(image_path):
                        # New format: remove folder and count files inside
                        file_count = len([f for f in os.listdir(image_path) if f.endswith(".png")])
                        shutil.rmtree(image_path)
                        deleted_count += file_count
                    else:
                        # Old format: remove single file
                        os.remove(image_path)
                        deleted_count += 1

                # Remove input image if it exists
                if item.get("input_image_path") and os.path.exists(item["input_image_path"]):
                    os.remove(item["input_image_path"])

            # Clear history file
            with open(history_file, "w") as f:
                json.dump([], f)

        # Remove any remaining files/folders in images directory
        if os.path.exists(images_dir):
            for item_name in os.listdir(images_dir):
                item_path = os.path.join(images_dir, item_name)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif item_name.endswith(".png"):
                    os.remove(item_path)

        return JSONResponse(
            content={
                "message": f"History cleared successfully. Deleted {deleted_count} images.",
                "deleted_images": deleted_count,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.post("/dataset/create", summary="Create dataset from history images")
async def create_dataset_from_history(request: CreateDatasetRequest):
    """
    Create a dataset from selected images in history

    Args:
        request: Contains list of image IDs to include in the dataset

    Returns:
        JSON response with dataset details
    """
    image_ids = request.image_ids
    if not image_ids or not isinstance(image_ids, list):
        raise HTTPException(status_code=400, detail="Invalid image IDs list")

    # Sanitize dataset name
    dataset_id = slugify(request.dataset_name)
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    # Check if dataset already exists
    existing_dataset = await db.get_dataset(dataset_id)
    if existing_dataset:
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_id}' already exists")

    # Find selected images efficiently
    selected_images = []
    for image_id in image_ids:
        image_item = find_image_by_id(image_id)
        if image_item:
            selected_images.append(image_item)

    if not selected_images:
        raise HTTPException(status_code=404, detail="No images found for the given IDs")

    # Calculate total image count (accounting for multi-image generations)
    total_image_count = 0
    for image_item in selected_images:
        if os.path.isdir(image_item.image_path):
            # Count images in folder
            image_files = [
                f for f in os.listdir(image_item.image_path) if f.endswith(".png") and f.replace(".png", "").isdigit()
            ]
            total_image_count += len(image_files)
        else:
            # Single image
            total_image_count += 1

    # Create dataset in database
    try:
        json_data = {
            "generated": True,
            "source": "diffusion_history",
            "description": request.description or f"Dataset created from {total_image_count} diffusion images",
            "image_count": total_image_count,
            "created_from_image_ids": image_ids,
        }
        await db.create_local_dataset(dataset_id, json_data=json_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset in database: {str(e)}")

    # Create dataset directory
    dataset_dir = dirs.dataset_dir_by_id(dataset_id)
    images_dir = os.path.join(dataset_dir, "train")
    os.makedirs(images_dir, exist_ok=True)

    # Prepare dataset metadata and copy images
    dataset_records = []
    file_counter = 0

    for image_item in selected_images:
        try:
            # Check if this is a multi-image generation (folder) or single image
            if os.path.isdir(image_item.image_path):
                # Multi-image generation - process each image in the folder
                image_files = []
                for filename in os.listdir(image_item.image_path):
                    if filename.endswith(".png") and filename.replace(".png", "").isdigit():
                        image_files.append(filename)

                # Sort by numeric order (0.png, 1.png, etc.)
                image_files.sort(key=lambda x: int(x.replace(".png", "")))

                for img_filename in image_files:
                    src_image_path = os.path.join(image_item.image_path, img_filename)

                    # Generate new filename for the dataset
                    dataset_filename = f"image_{file_counter:04d}.png"
                    dest_image_path = os.path.join(images_dir, dataset_filename)

                    # Copy image file
                    if os.path.exists(src_image_path):
                        shutil.copy2(src_image_path, dest_image_path)
                    else:
                        print(f"Warning: Image file not found at {src_image_path}")
                        continue

                    # Create record with essential fields
                    record = {
                        "file_name": dataset_filename,
                        "text": image_item.prompt,
                    }

                    # Add metadata if requested
                    if request.include_metadata:
                        record.update(
                            {
                                "model": image_item.model,
                                "adaptor": image_item.adaptor,
                                "adaptor_scale": image_item.adaptor_scale,
                                "num_inference_steps": image_item.num_inference_steps,
                                "guidance_scale": image_item.guidance_scale,
                                "seed": image_item.seed,
                                "negative_text": image_item.negative_prompt,
                                "upscaled": image_item.upscaled,
                                "upscale_factor": image_item.upscale_factor,
                                "eta": image_item.eta,
                                "clip_skip": image_item.clip_skip,
                                "guidance_rescale": image_item.guidance_rescale,
                                "height": image_item.height,
                                "width": image_item.width,
                                "timestamp": image_item.timestamp,
                                "original_id": image_item.id,
                                "image_index": int(
                                    img_filename.replace(".png", "")
                                ),  # Add image index for multi-image generations
                            }
                        )

                    dataset_records.append(record)
                    file_counter += 1

            else:
                # Single image generation (backward compatibility)
                dataset_filename = f"image_{file_counter:04d}.png"
                dest_image_path = os.path.join(images_dir, dataset_filename)

                # Copy image file
                if os.path.exists(image_item.image_path):
                    shutil.copy2(image_item.image_path, dest_image_path)
                else:
                    print(f"Warning: Image file not found at {image_item.image_path}")
                    continue

                # Create record with essential fields
                record = {
                    "file_name": dataset_filename,
                    "text": image_item.prompt,
                    "negative_text": image_item.negative_prompt,
                }

                # Add metadata if requested
                if request.include_metadata:
                    record.update(
                        {
                            "model": image_item.model,
                            "adaptor": image_item.adaptor,
                            "adaptor_scale": image_item.adaptor_scale,
                            "num_inference_steps": image_item.num_inference_steps,
                            "guidance_scale": image_item.guidance_scale,
                            "seed": image_item.seed,
                            "upscaled": image_item.upscaled,
                            "upscale_factor": image_item.upscale_factor,
                            "eta": image_item.eta,
                            "clip_skip": image_item.clip_skip,
                            "guidance_rescale": image_item.guidance_rescale,
                            "height": image_item.height,
                            "width": image_item.width,
                            "timestamp": image_item.timestamp,
                            "original_id": image_item.id,
                        }
                    )

                dataset_records.append(record)
                file_counter += 1

        except Exception as e:
            print(f"Warning: Failed to process image {image_item.id}: {str(e)}")
            continue

    if not dataset_records:
        # Clean up if no images were successfully processed
        await db.delete_dataset(dataset_id)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to process any images")

    # Save dataset as JSONL file
    try:
        # Make train directory if it doesn't exist
        os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
        dataset_file = os.path.join(dataset_dir, "train", "metadata.jsonl")
        with open(dataset_file, "w") as f:
            for record in dataset_records:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        # Clean up on failure
        await db.delete_dataset(dataset_id)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset: {str(e)}")

    return JSONResponse(
        content={
            "status": "success",
            "message": f"Dataset '{dataset_id}' created successfully with {len(dataset_records)} images.",
            "dataset_id": dataset_id,
            "dataset_dir": dataset_dir,
            "records_count": len(dataset_records),
        }
    )


def should_use_diffusion_worker(model) -> bool:
    """Use the diffusion worker only for FLUX models"""
    # return use_multi_gpu and torch.cuda.device_count() > 1
    try:
        # Check if model has FLUX components by looking for config
        from huggingface_hub import model_info
        info = model_info(model)
        config = getattr(info, "config", {})
        diffusers_config = config.get("diffusers", {})
        architectures = diffusers_config.get("_class_name", "")
        if isinstance(architectures, str):
            architectures = [architectures]
        for arch in architectures:
            if "flux" in arch.lower():
                return True

        return False
    except Exception:
        return False


def get_python_executable():
    """Get the Python executable path"""
    return sys.executable


async def run_multi_gpu_generation(
    request: DiffusionRequest,
    generation_id: str,
    images_folder: str,
    input_image_path: str = "",
    mask_image_path: str = "",
    is_img2img: bool = False,
    is_inpainting: bool = False,
) -> dict:
    """Run image generation using multi-GPU subprocess approach"""

    # Set seed - use provided seed or generate a random one
    if request.seed is None or request.seed < 0:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = request.seed

    # Prepare configuration for worker
    config = {
        "model": request.model,
        "adaptor": request.adaptor,
        "adaptor_scale": request.adaptor_scale,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "num_images": request.num_images,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "seed": seed,
        "eta": request.eta,
        "clip_skip": request.clip_skip,
        "guidance_rescale": request.guidance_rescale,
        "height": request.height,
        "width": request.width,
        "strength": request.strength,
        "is_img2img": is_img2img,
        "input_image": request.input_image if (is_img2img or is_inpainting) else "",
        "is_inpainting": is_inpainting,
        "mask_image": request.mask_image if is_inpainting else "",
        "upscale": request.upscale,
        "upscale_factor": request.upscale_factor,
        "enable_sharding": request.enable_sharding,
    }

    # Save config to temporary file
    ensure_directories()
    config_path = os.path.join(get_diffusion_dir(), f"config_{generation_id}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Get worker script path
    # current_dir = os.path.dirname(os.path.abspath(__file__))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(os.path.dirname(current_dir), "shared", "diffusion_worker.py")

    try:
        # Setup environment for accelerate
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

        # Build command for accelerate launch
        cmd = [
            get_python_executable(),
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(1),
            worker_script,
            "--config",
            config_path,
            "--output-dir",
            images_folder,
            "--worker-id",
            generation_id,
        ]

        print(f"Running multi-GPU generation with command: {' '.join(cmd)}")

        # Start the process asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Redirect stderr to stdout
        )

        # Print output in real-time asynchronously
        output_lines = []
        while True:
            line = await process.stdout.readline()
            if line:
                line_text = line.decode('utf-8').rstrip()
                print(line_text)  # Print to console in real-time
                output_lines.append(line_text + '\n')
            else:
                break

        # Wait for process to complete and get return code
        return_code = await process.wait()

        # Combine all output for error checking
        combined_output = "".join(output_lines)

        if return_code != 0:
            print(f"Worker subprocess failed with return code {return_code}")
            print(f"Combined output: {combined_output}")

            # Check if it's an OOM error (exitcode -9 indicates process was killed)
            if return_code == -9 or "CUDA out of memory" in combined_output or "OutOfMemoryError" in combined_output:
                # Try to load any partial result to get OOM details
                result_path = os.path.join(images_folder, "result.json")
                if os.path.exists(result_path):
                    try:
                        with open(result_path, "r") as f:
                            worker_result = json.load(f)
                        if worker_result.get("error_type") == "OOM":
                            oom_suggestions = worker_result.get("suggestions", [])
                            suggestion_text = "\n".join([f"  • {s}" for s in oom_suggestions])
                            raise RuntimeError(
                                f"CUDA Out of Memory during multi-GPU generation.\n\nSuggestions:\n{suggestion_text}"
                            )
                    except Exception:
                        pass
                raise RuntimeError(
                    "CUDA Out of Memory during multi-GPU generation. Try reducing image resolution, inference steps, or closing other GPU processes."
                )

            raise RuntimeError(f"Multi-GPU generation failed: {combined_output}")

        # Load result from worker
        result_path = os.path.join(images_folder, "result.json")
        if not os.path.exists(result_path):
            raise RuntimeError("Worker did not produce result file")

        with open(result_path, "r") as f:
            worker_result = json.load(f)

        if not worker_result.get("success", False):
            error_msg = worker_result.get("error", "Unknown error")
            error_type = worker_result.get("error_type", "")

            if error_type == "OOM":
                suggestions = worker_result.get("suggestions", [])
                suggestion_text = "\n".join([f"  • {s}" for s in suggestions])
                raise RuntimeError(f"CUDA Out of Memory: {error_msg}\n\nSuggestions:\n{suggestion_text}")
            else:
                raise RuntimeError(f"Worker reported failure: {error_msg}")

        # Clean up config file
        try:
            os.remove(config_path)
        except Exception:
            pass

        return {
            "images": worker_result["images"],
            "generation_time": worker_result["generation_time"],
            "seed": worker_result["seed"],
            "num_images": worker_result["num_images"],
        }

    except subprocess.TimeoutExpired:
        print("Multi-GPU generation timed out")
        raise RuntimeError("Generation timed out after 10 minutes")
    except Exception as e:
        # Clean up config file on error
        try:
            os.remove(config_path)
        except Exception:
            pass
        raise e
