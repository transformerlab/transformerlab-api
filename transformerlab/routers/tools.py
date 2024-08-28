import os
import json


from fastapi import APIRouter


router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/list", summary="List the tools that are currently installed.")
async def list_tools() -> list[object]:
    result = []
    return result