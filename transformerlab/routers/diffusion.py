from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import model_info, list_repo_files
import base64
from fastapi.responses import FileResponse, JSONResponse
from io import BytesIO
import torch
import asyncio
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline, AutoPipelineForText2Image
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
]

# Fixed upscaling models
UPSCALE_MODEL_STANDARD = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALE_MODEL_LATENT = "stabilityai/sd-x2-latent-upscaler"


# Request schema for image generation
class DiffusionRequest(BaseModel):
    model: str
    prompt: str = ""
    adaptor: str = ""
    adaptor_scale: float = 1.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None
    upscale: bool = False
    upscale_factor: int = 4
    # Negative prompting
    negative_prompt: str = ""
    # Advanced guidance control
    eta: float = 0.0
    clip_skip: int = 0
    guidance_rescale: float = 0.0
    height: int = 0
    width: int = 0


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
    generation_time: float = 0.0  # Time taken for generation in seconds


class HistoryResponse(BaseModel):
    images: List[ImageHistoryItem]
    total: int


class CreateDatasetRequest(BaseModel):
    dataset_name: str
    image_ids: List[str]
    description: str = ""
    include_metadata: bool = True


# Global cache for loaded pipelines
_PIPELINES: dict = {}
_PIPELINES_LOCK = threading.Lock()

# History file path
HISTORY_FILE = "history.json"


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


def get_pipeline_key(model: str, adaptor: str = "") -> str:
    """Generate cache key for model + adaptor combination"""
    return f"{model}::{adaptor}" if adaptor else model


def get_pipeline(model: str, adaptor: str = "", device: str = "cuda"):
    cache_key = get_pipeline_key(model, adaptor)

    with _PIPELINES_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        # Load base pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)

        # Load LoRA adaptor if provided - same code for local and HF Hub!
        if adaptor and adaptor.strip():
            try:
                adaptor_dir = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR"), "adaptors", secure_filename(model))
                adaptor_path = os.path.join(adaptor_dir, secure_filename(adaptor))
                if os.path.exists(adaptor_path):
                    pipe.load_lora_weights(adaptor_path)
                    print(f"Loaded LoRA adaptor: {adaptor}")
                else:
                    print(f"Error: Adaptor file not found at {adaptor_path}")
                    raise FileNotFoundError(f"Adaptor file not found: {adaptor_path}")
            except Exception:
                print(f"Warning: Failed to load LoRA adaptor '{adaptor}'")
                # Continue without LoRA rather than failing

        _PIPELINES[cache_key] = pipe
        return pipe


def get_upscale_pipeline(upscale_factor: int = 4, device: str = "cuda"):
    """Get the appropriate upscaling pipeline based on the factor"""
    cache_key = f"upscale_{upscale_factor}"

    with _PIPELINES_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

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
        _PIPELINES[cache_key] = pipe
        return pipe


def upscale_image(image: Image.Image, prompt: str, upscale_factor: int = 4, device: str = "cuda"):
    """Upscale an image using Stable Diffusion upscaler"""
    upscale_pipe = get_upscale_pipeline(upscale_factor, device)

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


@router.post("/generate", summary="Generate image with Stable Diffusion")
async def generate_image(request: DiffusionRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = get_pipeline(request.model, request.adaptor, device=device)

    # Set seed - use provided seed or generate a random one
    if request.seed is None or request.seed < 0:
        print("No valid seed provided, generating a random seed")
        seed = random.randint(0, 2**32 - 1)
    else:
        print(f"Using provided seed: {request.seed}")
        seed = request.seed

    generator = torch.manual_seed(seed)

    # Start timing
    start_time = time.time()

    # Run in thread to avoid blocking event loop
    def run_pipe():
        generation_kwargs = {
            "prompt": request.prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "eta": request.eta,
        }

        # Add negative prompt if provided
        if request.negative_prompt:
            generation_kwargs["negative_prompt"] = request.negative_prompt

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
        return result.images[0]

    try:
        # Time the main generation
        generation_start = time.time()
        image = await asyncio.get_event_loop().run_in_executor(None, run_pipe)
        generation_time = time.time() - generation_start

        # Apply upscaling if requested
        if request.upscale:
            print(f"Upscaling image with factor {request.upscale_factor}x")

            def run_upscale():
                return upscale_image(image, request.prompt, request.upscale_factor, device)

            upscale_start = time.time()
            image = await asyncio.get_event_loop().run_in_executor(None, run_upscale)
            upscale_time = time.time() - upscale_start
            total_generation_time = generation_time + upscale_time
            print(f"Generation took {generation_time:.2f}s, upscaling took {upscale_time:.2f}s, total: {total_generation_time:.2f}s")
        else:
            total_generation_time = generation_time
            print(f"Generation took {generation_time:.2f}s")

        # Generate unique ID and timestamp
        generation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Save image to disk
        ensure_directories()
        image_filename = f"{generation_id}.png"
        image_path = os.path.join(get_images_dir(), image_filename)
        image.save(image_path, format="PNG")

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
            image_path=image_path,
            timestamp=timestamp,
            upscaled=request.upscale,
            upscale_factor=request.upscale_factor if request.upscale else 1,
            negative_prompt=request.negative_prompt,
            eta=request.eta,
            clip_skip=request.clip_skip,
            guidance_rescale=request.guidance_rescale,
            height=request.height if request.height > 0 else image.height,
            width=request.width if request.width > 0 else image.width,
            generation_time=total_generation_time,
        )
        save_to_history(history_item)

        # Return base64 encoded image for immediate display
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(
            content={
                "id": generation_id,
                "prompt": request.prompt,
                "adaptor": request.adaptor,
                "adaptor_scale": request.adaptor_scale,
                "image_base64": img_str,
                "image_path": image_path,
                "timestamp": timestamp,
                "generation_time": total_generation_time,
                "error_code": 0,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.post("/is_stable_diffusion", summary="Check if model is Stable Diffusion")
async def is_stable_diffusion_model(request: DiffusionRequest):
    """
    Returns {"is_stable_diffusion": True/False, "reason": "..."}
    """
    model_id = request.model
    try:
        info = model_info(model_id)
        tags = getattr(info, "tags", [])
        config = getattr(info, "config", {})
        diffusers_config = config.get("diffusers", {})
        architectures = diffusers_config.get("_class_name", "")
        if isinstance(architectures, str):
            architectures = [architectures]

        # Check architectures
        if any(a in ALLOWED_TEXT2IMG_ARCHITECTURES for a in architectures):
            return {"is_stable_diffusion": True, "reason": "Architecture matches allowed SD"}
        
        return {"is_stable_diffusion": False, "reason": "No SD indicators found"}
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
    if limit <= 0 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")

    return load_history(limit=limit, offset=offset)


@router.get("/history/{image_id}", summary="Get the actual image by ID")
async def get_image_by_id(image_id: str):
    history = load_history(limit=1000)  # Load enough to find the image

    # Find the image in history
    image_item = None
    for item in history.images:
        if item.id == image_id:
            image_item = item
            break

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Check if image file exists
    if not os.path.exists(image_item.image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found at {image_item.image_path}")

    return FileResponse(
        image_item.image_path,
    )


@router.get("/history/{image_id}/info", summary="Get image by ID")
async def get_image_info_by_id(image_id: str):
    """
    Get a specific image by its ID

    Args:
        image_id: The unique ID of the image

    Returns:
        Base64 encoded image data
    """
    history = load_history(limit=1000)  # Load enough to find the image

    # Find the image in history
    image_item = None
    for item in history.images:
        if item.id == image_id:
            image_item = item
            break

    if not image_item:
        raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

    # Check if image file exists
    if not os.path.exists(image_item.image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found at {image_item.image_path}")

    try:
        # Read and encode image
        with open(image_item.image_path, "rb") as f:
            image_data = f.read()
        img_str = base64.b64encode(image_data).decode("utf-8")

        return JSONResponse(content={"id": image_item.id, "image_base64": img_str, "metadata": image_item.model_dump()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")


@router.delete("/history/{image_id}", summary="Delete image from history")
async def delete_image_from_history(image_id: str):
    """
    Delete a specific image from history and remove the image file

    Args:
        image_id: The unique ID of the image to delete
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

        # Remove image file if it exists
        if os.path.exists(item_to_remove["image_path"]):
            os.remove(item_to_remove["image_path"])

        # Save updated history
        with open(history_file, "w") as f:
            json.dump(updated_history, f, indent=2)

        return JSONResponse(
            content={"message": f"Image {image_id} deleted successfully", "deleted_item": item_to_remove}
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

            # Remove all image files
            for item in history:
                if os.path.exists(item["image_path"]):
                    os.remove(item["image_path"])
                    deleted_count += 1

            # Clear history file
            with open(history_file, "w") as f:
                json.dump([], f)

        # Remove any remaining files in images directory
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.endswith(".png"):
                    os.remove(os.path.join(images_dir, filename))

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

    # Load history and find selected images
    history = load_history(limit=1000)  # Load enough history
    selected_images = [item for item in history.images if item.id in image_ids]

    if not selected_images:
        raise HTTPException(status_code=404, detail="No images found for the given IDs")

    # Create dataset in database
    try:
        json_data = {
            "generated": True,
            "source": "diffusion_history",
            "description": request.description or f"Dataset created from {len(selected_images)} diffusion images",
            "image_count": len(selected_images),
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

    for i, image_item in enumerate(selected_images):
        try:
            # Generate new filename for the image
            image_filename = f"image_{i:04d}.png"
            dest_image_path = os.path.join(images_dir, image_filename)

            # Copy image file
            if os.path.exists(image_item.image_path):
                shutil.copy2(image_item.image_path, dest_image_path)
            else:
                print(f"Warning: Image file not found at {image_item.image_path}")
                continue

            # Create record with essential fields
            record = {
                "file_name": image_filename,
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
