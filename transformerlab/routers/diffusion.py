from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import model_info, list_repo_files
import base64
from fastapi.responses import JSONResponse
from io import BytesIO
import torch
import asyncio
from diffusers import StableDiffusionPipeline
import threading
import os
from werkzeug.utils import secure_filename
import json
import uuid
from datetime import datetime
from typing import List


router = APIRouter(prefix="/diffusion", tags=["diffusion"])

ALLOWED_STABLE_DIFFUSION_ARCHITECTURES = [
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
]


# Request schema for image generation
class DiffusionRequest(BaseModel):
    model: str
    prompt: str = ""
    adaptor: str = ""
    adaptor_scale: float = 1.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None


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


class HistoryResponse(BaseModel):
    images: List[ImageHistoryItem]
    total: int


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
    print(f"Saving history to {history_file}")

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
        pipe = StableDiffusionPipeline.from_pretrained(
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
            except Exception as e:
                print(f"Warning: Failed to load LoRA adaptor '{adaptor}'")
                # Continue without LoRA rather than failing

        _PIPELINES[cache_key] = pipe
        return pipe


@router.post("/generate", summary="Generate image with Stable Diffusion")
async def generate_image(request: DiffusionRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = get_pipeline(request.model, request.adaptor, device=device)
    generator = torch.manual_seed(request.seed) if request.seed is not None else None

    # Run in thread to avoid blocking event loop
    def run_pipe():
        generation_kwargs = {
            "prompt": request.prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
        }

        # Add LoRA scale if adaptor is being used
        if request.adaptor and request.adaptor.strip():
            generation_kwargs["cross_attention_kwargs"] = {"scale": request.adaptor_scale}

        result = pipe(**generation_kwargs)
        return result.images[0]

    try:
        image = await asyncio.get_event_loop().run_in_executor(None, run_pipe)

        # Generate unique ID and timestamp
        generation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Save image to disk
        ensure_directories()
        image_filename = f"{generation_id}.png"
        image_path = os.path.join(get_images_dir(), image_filename)
        image.save(image_path, format="PNG")
        print(f"Image saved to {image_path}")
        # Save to history
        history_item = ImageHistoryItem(
            id=generation_id,
            model=request.model,
            prompt=request.prompt,
            adaptor=request.adaptor,
            adaptor_scale=request.adaptor_scale,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            image_path=image_path,
            timestamp=timestamp,
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
        architectures = getattr(info, "architectures", [])
        # Check tags or architectures
        if any("stable-diffusion" in t or "diffusers" in t for t in tags):
            return {"is_stable_diffusion": True, "reason": "Found SD tag"}
        if any(a in ALLOWED_STABLE_DIFFUSION_ARCHITECTURES for a in architectures):
            return {"is_stable_diffusion": True, "reason": "Architecture matches allowed SD"}
        # Check for model_index.json file
        try:
            files = list_repo_files(model_id)
            if any(f.endswith("model_index.json") for f in files):
                return {"is_stable_diffusion": True, "reason": "model_index.json present"}
        except Exception:
            pass
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


@router.get("/history/{image_id}/image", summary="Get image by ID")
async def get_image_by_id(image_id: str):
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

        return JSONResponse(content={"id": image_item.id, "image_base64": img_str, "metadata": image_item.dict()})
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
