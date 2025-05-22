from fastapi import APIRouter
from transformerlab.shared import galleries

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("/list")
async def list_recipes():
    """List all recipes for a given experiment name."""
    recipes_gallery = galleries.get_exp_recipe_gallery()
    return recipes_gallery
