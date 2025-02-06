import json
import os
from typing import Annotated, Union
from fastapi import APIRouter, Body

from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify

router = APIRouter(prefix="/batched_prompts", tags=["batched_prompts"])


@router.get("/list")
async def list_prompts():
    """List the batched prompts that we have on disk"""

    batched_prompts = []
    batched_prompts_dir = dirs.BATCHED_PROMPTS_DIR
    for file in os.listdir(batched_prompts_dir):
        if file.endswith(".json"):
            with open(os.path.join(batched_prompts_dir, file), "r") as f:
                try:
                    p = json.load(f)
                    name = file.split(".")[0]
                    batched_prompts.append({"name": name, "prompts": p})
                except Exception as e:
                    print(f"Error loading batched prompt file: {file}: {e}, skipping")

    return batched_prompts


# The prompt can either be a list of lists of dics (for conversations)
# or a list of strings (for completions)
@router.post("/new")
async def new_prompt(name: Annotated[str, Body()], prompts: Annotated[Union[list[list[dict]], list[str]], Body()]):
    """Create a new batched prompt"""

    slug = slugify(name)
    prompts_dir = dirs.BATCHED_PROMPTS_DIR
    prompt_file = os.path.join(prompts_dir, f"{slug}.json")

    with open(prompt_file, "w") as f:
        json_str = json.dumps(prompts, indent=4)
        f.write(json_str)

    return {"status": "success", "data": prompts}


@router.get("/delete/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a batched prompt"""

    prompts_dir = dirs.BATCHED_PROMPTS_DIR
    prompt_file = os.path.join(prompts_dir, f"{prompt_id}.json")

    if os.path.exists(prompt_file):
        os.remove(prompt_file)
        return {"status": "success", "message": f"Prompt {prompt_id} deleted"}
    else:
        return {"status": "error", "message": f"Prompt {prompt_id} not found"}
