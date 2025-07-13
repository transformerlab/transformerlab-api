from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import model_info
from fastapi.responses import FileResponse, JSONResponse
import os
from werkzeug.utils import secure_filename
import json
import uuid
from typing import List
import shutil
from transformerlab.db.datasets import get_dataset, create_local_dataset, delete_dataset
from transformerlab.models import model_helper
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify
import transformerlab.db.jobs as db_jobs
import logging
import sys


router = APIRouter(prefix="/diffusion", tags=["diffusion"])

UNIVERSIAL_GENERATION_ID = str(uuid.uuid4())

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


def _setup_diffusion_logger():
    """Setup logging for the diffusion router that logs to both console and file"""
    logger = logging.getLogger("diffusion")

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [DIFFUSION] [%(levelname)s] %(message)s")

    # File handler
    try:
        file_handler = logging.FileHandler(dirs.GLOBAL_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Continue without file logging if there's an issue

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid affecting other routes
    logger.propagate = False

    return logger


# Initialize the diffusion logger
diffusion_logger = _setup_diffusion_logger()


def log_print(*args, **kwargs):
    """Enhanced logging function for diffusion router"""
    message = " ".join(str(arg) for arg in args)
    diffusion_logger.info(message)


class DiffusionOutputCapture:
    """Context manager to capture all stdout/stderr and redirect to diffusion logger"""

    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Redirect to our logger
        sys.stdout = LoggerWriter(diffusion_logger.info)
        sys.stderr = LoggerWriter(diffusion_logger.info)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class LoggerWriter:
    """Writer class that redirects output to logger"""

    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        message = message.strip()
        if message and "📝 File changed:" not in message:
            self.level(message)

    def flush(self):
        pass  # Needed for compatibility with some tools that use flush()


# Request schema for image generation
class DiffusionRequest(BaseModel):
    plugin: str = "image_diffusion"
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
    generation_id: str | None = None
    scheduler: str = "default"
    process_type: str | None = None

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
    is_controlnet: str = ""  # Check if using ControlNet
    # Intermediate image saving
    save_intermediate_images: bool = True  # Whether to save intermediate images during generation


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
    processed_image: str | None = None  # the preprocessed image for ControlNets
    strength: float = 0.8  # Denoising strength used
    is_img2img: bool = False  # Whether this was an img2img generation
    # Inpainting specific fields
    mask_image_path: str = ""  # Path to mask image (for inpainting)
    is_inpainting: bool = False  # Whether this was an inpainting generation
    is_controlnet: str = ""
    scheduler: str = "default"
    # Intermediate image saving
    saved_intermediate_images: bool = True  # Whether intermediate images were saved


class HistoryResponse(BaseModel):
    images: List[ImageHistoryItem]
    total: int


class CreateDatasetRequest(BaseModel):
    dataset_name: str
    image_ids: List[str]
    description: str = ""
    include_metadata: bool = True


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


@router.post("/generate", summary="Generate image with Stable Diffusion")
async def generate_image(experimentId: int, request: DiffusionRequest):
    try:
        # Validate num_images parameter
        if request.num_images < 1 or request.num_images > 10:
            raise HTTPException(status_code=400, detail="num_images must be between 1 and 10")

        # Validate diffusion type
        if request.plugin == "image_diffusion":
            request_dict = request.dict()
            generation_id = request.generation_id or UNIVERSIAL_GENERATION_ID
            request_dict["generation_id"] = generation_id

            job_config = {
                "plugin": request.plugin,
                "config": request_dict,
            }

            job_id = await db_jobs.job_create(
                type="DIFFUSION", status="QUEUED", job_data=job_config, experiment_id=experimentId
            )

            images_folder = os.path.join(get_images_dir(), generation_id)
            images_folder = os.path.normpath(images_folder)  # Normalize path
            if not images_folder.startswith(get_images_dir()):  # Validate containment
                raise HTTPException(status_code=400, detail="Invalid generation_id: Path traversal detected.")
            tmp_json_path = os.path.join(images_folder, "tmp_json.json")
            # Normalize and validate the path
            tmp_json_path = os.path.normpath(tmp_json_path)

            return {
                "job_id": job_id,
                "status": "started",
                "generation_id": generation_id,
                "images_folder": images_folder,
                "json_path": tmp_json_path,
                "message": "Diffusion job has been queued and is running in background.",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported diffusion_type: {request.diffusion_type}",
            )

    except Exception as e:
        log_print(f"Error during image generation: {str(e)}")
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
                if arch in ALLOWED_TEXT2IMG_ARCHITECTURES and "flux" not in arch.lower():
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
async def get_image_by_id(
    image_id: str,
    index: int = 0,
    input_image: bool = False,
    mask_image: bool = False,
    step: bool = False,
    preprocessed: bool = False,
):
    """
    Get an image from history by ID and index

    Args:
        image_id: The unique ID of the image set
        index: The index of the image in the set (default 0)
        input_image: Whether to return the input image instead of generated image
        mask_image: Whether to return the mask image instead of generated image
    """
    if step:
        # If step is requested, we need to check if intermediate images were saved
        images_dir = get_images_dir()
        image_dir_based_on_id = os.path.normpath(os.path.join(images_dir, image_id))

        # Ensure the constructed path is within the intended base directory
        if not image_dir_based_on_id.startswith(images_dir):
            raise HTTPException(status_code=400, detail="Invalid image ID or path traversal attempt detected")

        # Check if the image path is a directory (new format)
        if not os.path.isdir(image_dir_based_on_id):
            raise HTTPException(status_code=404, detail=f"Image path is not a directory for image ID {image_id}")

        # Construct the path for the step image
        step_image_path = os.path.normpath(os.path.join(image_dir_based_on_id, "step.png"))
        if not step_image_path.startswith(images_dir):
            raise HTTPException(status_code=400, detail="Invalid path traversal attempt detected")

        if not os.path.exists(step_image_path):
            raise HTTPException(status_code=404, detail=f"Step image file not found at {step_image_path}")

        return FileResponse(step_image_path)

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
    elif preprocessed:
        if not image_item.processed_image or not image_item.processed_image.strip():
            raise HTTPException(status_code=404, detail=f"No preprocessed image found for image ID {image_id}")
        image_path = image_item.processed_image
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Preprocessed image file not found at {image_path}")
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
        # Remove processed image if it exists
        if item_to_remove.get("processed_image") and os.path.exists(item_to_remove["processed_image"]):
            os.remove(item_to_remove["processed_image"])

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
                # Remove processed image if it exists
                if item.get("processed_image") and os.path.exists(item["processed_image"]):
                    os.remove(item["processed_image"])

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
    existing_dataset = await get_dataset(dataset_id)
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
        await create_local_dataset(dataset_id, json_data=json_data)
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
                        log_print(f"Warning: Image file not found at {src_image_path}")
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
                    log_print(f"Warning: Image file not found at {image_item.image_path}")
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
            log_print(f"Warning: Failed to process image {image_item.id}: {str(e)}")
            continue

    if not dataset_records:
        # Clean up if no images were successfully processed
        await delete_dataset(dataset_id)
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
        await delete_dataset(dataset_id)
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


@router.get("/controlnets", summary="List available ControlNet models")
async def list_controlnets():
    """
    Lists all downloaded ControlNet models by reading the controlnets directory
    and extracting `_class_name` from their config.json.
    """
    all_models = await model_helper.list_installed_models(embedding=False)
    models = []

    for model in all_models:
        json_data = model.get("json_data", {})

        # Check common locations
        arch = json_data.get("architecture")
        if not arch:
            # Try model_index fallback
            model_index = json_data.get("model_index", {})
            arch = model_index.get("_class_name")

        # Final fallback: rely on explicit controlnet flag
        is_controlnet = json_data.get("is_controlnet", False)

        if (arch and "controlnet" in arch.lower()) or is_controlnet:
            models.append(model)

    return {"controlnets": models}


@router.post("/generate_id", summary="Get a new generation ID for image generation")
async def get_new_generation_id():
    """
    Returns a new unique generation ID and creates the images folder for it.
    """
    generation_id = str(uuid.uuid4())
    ensure_directories()
    images_folder = os.path.join(get_images_dir(), generation_id)
    os.makedirs(images_folder, exist_ok=True)
    return {"generation_id": generation_id, "images_folder": images_folder}


@router.get("/get_file/{generation_id}")
async def get_file(generation_id: str):
    root_dir = get_images_dir()
    file_path = os.path.normpath(os.path.join(root_dir, generation_id, "tmp_json.json"))
    try:
        if not file_path.startswith(root_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        return JSONResponse(content=data)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
