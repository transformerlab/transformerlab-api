import json
import os
from fastapi import APIRouter

from transformerlab.shared import dirs

router = APIRouter(prefix="/prompts", tags=["rag"])


@router.get("/list")
async def list_prompts():
    """List the prompt templates available in the prompt gallery"""

    remote_gallery_file = os.path.join(
        dirs.TFL_SOURCE_CODE_DIR, "transformerlab/galleries/prompt-gallery.json")

    with open(remote_gallery_file, "r") as f:
        prompt_gallery = json.load(f)
    return prompt_gallery
