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
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None


# Global cache for loaded pipelines
_PIPELINES: dict = {}
_PIPELINES_LOCK = threading.Lock()


def get_pipeline(model: str, device: str = "cuda"):
    with _PIPELINES_LOCK:
        if model in _PIPELINES:
            return _PIPELINES[model]
        pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        _PIPELINES[model] = pipe
        return pipe


@router.post("/generate", summary="Generate image with Stable Diffusion")
async def generate_image(request: DiffusionRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = get_pipeline(request.model, device=device)
    generator = torch.manual_seed(request.seed) if request.seed is not None else None

    # Run in thread to avoid blocking event loop
    def run_pipe():
        result = pipe(
            request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        return result.images[0]

    image = await asyncio.get_event_loop().run_in_executor(None, run_pipe)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return JSONResponse(
        content={
            "prompt": request.prompt,
            "image_base64": img_str,
            "error_code": 0,
        }
    )



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