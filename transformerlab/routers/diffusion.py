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


# Global cache for loaded pipelines
_PIPELINES: dict = {}
_PIPELINES_LOCK = threading.Lock()


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
            requires_safety_checker=False
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
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return JSONResponse(
            content={
                "prompt": request.prompt,
                "adaptor": request.adaptor,
                "adaptor_scale": request.adaptor_scale,
                "image_base64": img_str,
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