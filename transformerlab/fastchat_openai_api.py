# # This file is a modified version of open-ai compatible server from
# # FastChat.
# # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py

from typing import List, Optional

import httpx
from fastchat.protocol.openai_api_protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)


try:
    from pydantic.v1 import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"

    # Used to overwrite the random seed in huggingface transformers
    seed: Optional[int] = None

    api_keys: Optional[List[str]] = None


app_settings = AppSettings()


async def show_available_models():
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        await client.post(controller_address + "/refresh_all_workers")
        ret = await client.post(controller_address + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)
