from fastapi import APIRouter
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
from io import BytesIO
import torch
import asyncio
from diffusers import StableDiffusionPipeline
import threading

router = APIRouter(prefix="/diffusion", tags=["diffusion"])


# Request schema for image generation
class DiffusionRequest(BaseModel):
    model: str  # Model name or path supported by StableDiffusionPipeline
    prompt: str
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
