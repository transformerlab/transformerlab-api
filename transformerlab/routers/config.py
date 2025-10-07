from fastapi import APIRouter
from lab.config import Config as fs_config


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/get/{key}", summary="")
async def config_get(key: str):
    value = fs_config.get_value_by_key(key)
    return value


@router.get("/set", summary="")
async def config_set(k: str, v: str):
    fs_config.set_value_by_key(k, v)
    return {"key": k, "value": v}
