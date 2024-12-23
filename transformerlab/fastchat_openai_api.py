# This file is a modified version of open-ai compatible server from
# FastChat.
# https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py

from transformerlab.shared import dirs
from fastapi import APIRouter
from fastapi.responses import FileResponse

import os
import argparse
import asyncio
import atexit
import json
import logging
import signal
import subprocess
from contextlib import asynccontextmanager
import time
from typing import Any, Dict, Generator, List, Optional, Union

import fastapi
import httpx
import shortuuid
import tiktoken

# Using torch to test for CUDA and MPS support.
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastchat.constants import (
    WORKER_API_EMBEDDING_BATCH_SIZE,
    WORKER_API_TIMEOUT,
    ErrorCode,
)
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.protocol.api_protocol import (
    APITokenCheckRequest,
    APITokenCheckResponse,
    APITokenCheckResponseItem, BaseModel
)
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,

)


class APIChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]], List[List[Dict[str, str]]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    repetition_penalty: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    logprobs: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    logprobs: Optional[bool] = False


try:
    from pydantic.v1 import BaseSettings
except ImportError:
    from pydantic import BaseSettings


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.ERROR)
headers = {"User-Agent": "FastChat API Server"}


router = APIRouter()

get_bearer_token = HTTPBearer(auto_error=False)

conv_template_map = {}


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"

    # Used to overwrite the random seed in huggingface transformers
    seed: Optional[int] = None

    api_keys: Optional[List[str]] = None


app_settings = AppSettings()


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )


async def check_model(request) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None
    async with httpx.AsyncClient() as client:
        try:
            # First, if there is a slash in the name of the model, just use the second part:
            model_name = request.model.split("/")[-1]
            _worker_addr = await get_worker_address(model_name, client)
        except ValueError:
            models_ret = await client.post(controller_address + "/list_models")
            models = models_ret.json()["models"]
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                f"Expected model: {'&&'.join(models)}. Your model: {request.model}",
            )
    return ret


def log_prompt(prompt):
    """ Log the prompt to the global prompt.log file """
    MAX_LOG_SIZE_BEFORE_ROTATE = 1000000  # 1MB in bytes
    if os.path.exists(os.path.join(dirs.LOGS_DIR, "prompt.log")):
        if os.path.getsize(os.path.join(dirs.LOGS_DIR, "prompt.log")) > MAX_LOG_SIZE_BEFORE_ROTATE:
            with open(os.path.join(dirs.LOGS_DIR, "prompt.log"), "r") as f:
                lines = f.readlines()
            with open(os.path.join(dirs.LOGS_DIR, "prompt.log"), "w") as f:
                f.writelines(lines[-1000:])
            with open(os.path.join(dirs.LOGS_DIR, f"prompt_{time.strftime('%Y%m%d%H%M%S')}.log"), "w") as f:
                f.writelines(lines[:-1000])

    with open(os.path.join(dirs.LOGS_DIR, "prompt.log"), "a") as f:
        log_entry = {}
        log_entry["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry["log"] = prompt
        log_entry = json.dumps(log_entry)
        f.write(f"{log_entry}\n")


@router.get("/prompt_log", tags=["chat"])
async def get_prompt_log():
    return FileResponse(os.path.join(dirs.LOGS_DIR, "prompt.log"))


async def check_length(request, prompt, max_tokens):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)

        response = await client.post(
            worker_addr + "/model_details",
            headers=headers,
            json={"model": request.model},
            timeout=WORKER_API_TIMEOUT,
        )
        context_len = response.json()["context_length"]

        response = await client.post(
            worker_addr + "/count_token",
            headers=headers,
            json={"model": request.model, "prompt": prompt},
            timeout=WORKER_API_TIMEOUT,
        )
        token_num = response.json()["count"]

    if token_num + max_tokens > context_len:
        return create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(
            request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(text) for text in inp]

    return inp


async def get_gen_params(
    model_name: str,
    messages: Union[
        str,
        List[Dict[str, str]],
        # necessary for image support
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
    logprobs: Optional[bool] = False,
) -> Dict[str, Any]:
    conv = await get_conv(model_name)
    conv = Conversation(
        name=conv["name"],
        system_template=conv["system_template"],
        system_message=conv["system_message"],
        roles=conv["roles"],
        # LLMLab: 👇 We manually remove these fake messages that
        # FastChat would prepend convos with
        messages=list([]),
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )
    image_url = None
    images = None
    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.set_system_message(message["content"])
            elif msg_role == "user":
                if isinstance(message["content"], list):
                    text = message["content"][0].get("text", "")
                    # If we want to support multiple images in the future we need to change the following:
                    # 1. Support multiple instances of image_url in the message
                    # 2. Change the tuple to take in multiple image_url strings in the list
                    image_url = message["content"][1].get("image_url", "")
                    message["content"] = tuple([text, [image_url]])
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images = conv.get_images()
    if max_tokens is None:
        max_tokens = 512
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
        "logprobs": logprobs,
    }
    if images is not None and len(images) > 0:
        gen_params["images"] = images
    if not stop:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})
    return gen_params


async def get_worker_address(model_name: str, client: httpx.AsyncClient) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address

    model_name = model_name.split("/")[-1]

    ret = await client.post(
        controller_address + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")

    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model_name, client)
        conv_template = conv_template_map.get((worker_addr, model_name))
        if conv_template is None:
            response = await client.post(
                worker_addr + "/worker_get_conv_template",
                headers=headers,
                json={"model": model_name},
                timeout=WORKER_API_TIMEOUT,
            )
            conv_template = response.json()["conv"]
            conv_template_map[(worker_addr, model_name)] = conv_template
        return conv_template


@router.get("/v1/models", dependencies=[Depends(check_api_key)], tags=["chat"])
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
        model_cards.append(
            ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@router.post("/v1/chat/completions", dependencies=[Depends(check_api_key)], tags=["chat"])
async def create_openapi_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = await get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
        logprobs=request.logprobs,
    )
    error_check_ret = await check_length(
        request, gen_params["prompt"], gen_params["max_new_tokens"]
    )
    if error_check_ret is not None:
        return error_check_ret
    log_prompt(gen_params)
    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(
                    usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        # Convert the chunk to a dictionary
        chunk_dict = chunk.model_dump()

        # Convert the dictionary to a JSON string
        sorted_json = json.dumps(
            chunk_dict, sort_keys=True, ensure_ascii=False)

        # Use the JSON string in your response
        yield f"data: {sorted_json}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params):
            if content["error_code"] != 0:
                # Convert the content to a dictionary
                content_dict = content.model_dump()

                # Convert the dictionary to a JSON string
                sorted_json = json.dumps(
                    content_dict, sort_keys=True, ensure_ascii=False
                )

                yield f"data: {sorted_json}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            # Convert the chunk to a dictionary
            chunk_dict = chunk.dict(exclude_unset=True)

            # Convert the dictionary to a JSON string
            sorted_json = json.dumps(chunk_dict, ensure_ascii=False)

            # Use the JSON string in your response
            yield f"data: {sorted_json}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        # Convert the finish_chunk to a dictionary
        finish_chunk_dict = finish_chunk.dict(exclude_none=True)

        # Convert the dictionary to a JSON string
        sorted_json = json.dumps(finish_chunk_dict, ensure_ascii=False)

        # Use the JSON string in your response
        yield f"data: {sorted_json}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/completions", dependencies=[Depends(check_api_key)], tags=["chat"])
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    for text in request.prompt:
        error_check_ret = await check_length(request, text, request.max_tokens)
        if error_check_ret is not None:
            return error_check_ret

    if request.stream:
        generator = generate_completion_stream_generator(request, request.n)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = await get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
                logprobs=request.logprobs,
            )

            log_prompt(gen_params)

            for i in range(request.n):
                content = asyncio.create_task(generate_completion(gen_params))
                text_completions.append(content)

        try:
            all_tasks = await asyncio.gather(*text_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                {
                    "index": i,
                    "text": content["text"],
                    "logprobs": content.get("logprobs", None),
                    "finish_reason": content.get("finish_reason", "stop"),
                }
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(
                    usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(
                usage)
        )


async def generate_completion_stream_generator(request: CompletionRequest, n: int):
    model_name = request.model
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(n):
            previous_text = ""
            gen_params = await get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
                logprobs=request.logprobs,
            )

            log_prompt(gen_params)

            async for content in generate_completion_stream(gen_params):
                if content["error_code"] != 0:
                    # Convert the content to a dictionary
                    content_dict = content.model_dump()

                    # Convert the dictionary to a JSON string
                    sorted_json = json.dumps(
                        content_dict, sort_keys=True, ensure_ascii=False
                    )

                    # Use the JSON string in your response
                    yield f"data: {sorted_json}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text):]
                previous_text = decoded_unicode
                # todo: index is not apparent
                choice_data = {
                    "index": i,
                    "text": delta_text,
                    "logprobs": content.get("logprobs", None),
                    "finish_reason": content.get("finish_reason", None),
                }
                chunk = {
                    "id": id,
                    "object": "text_completion",
                    "choices": [choice_data],
                    "model": model_name,
                }
                if len(delta_text) == 0:
                    print('delta_text', delta_text)
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue
                # Convert the chunk to a dictionary
                chunk_dict = chunk

                # Convert the dictionary to a JSON string
                sorted_json = json.dumps(
                    chunk_dict, sort_keys=True, ensure_ascii=False)

                # Use the JSON string in your response
                yield f"data: {sorted_json}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        # Convert the finish_chunk to a dictionary
        finish_chunk_dict = finish_chunk

        print('finish_chunk_dict', finish_chunk_dict)

        # Convert the dictionary to a JSON string
        sorted_json = json.dumps(finish_chunk_dict, ensure_ascii=False)

        # Use the JSON string in your response
        yield f"data: {sorted_json}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(payload: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(payload["model"], client)
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            async for raw_chunk in response.aiter_raw():
                for chunk in raw_chunk.split(delimiter):
                    if not chunk:
                        continue
                    # print(chunk.decode())
                    data = None
                    try:
                        data = json.loads(chunk.decode())
                    except Exception as e:
                        # Catching this exception is a hack -- we do it because with log probs turned on,
                        # the response gets really long, more than 63892 bytes, and the stream gets cut off.
                        # This is a workaround to prevent the stream from breaking. But we should fix
                        # the underlying issue in the worker.
                        print('Caught Exception in OpenAI API: ', e)
                        continue
                    yield data


async def generate_completion(payload: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(payload["model"], client)

        response = await client.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        completion = response.json()
        return completion


@router.post("/v1/embeddings", dependencies=[Depends(check_api_key)], tags=["chat"])
@router.post(
    "/v1/engines/{model_name}/embeddings",
    dependencies=[Depends(check_api_key)],
    tags=["chat"],
)
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    request.input = process_input(request.model, request.input)

    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i: min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


async def get_embedding(payload: Dict[str, Any]):
    model_name = payload["model"]
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model_name, client)

        response = await client.post(
            worker_addr + "/worker_get_embeddings",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        embedding = response.json()
        return embedding


### GENERAL API - NOT OPENAI COMPATIBLE ###


@router.post("/api/v1/token_check", tags=["chat"], include_in_schema=False)
async def count_tokens(request: APITokenCheckRequest):
    """
    Checks the token count for each message in your list
    This is not part of the OpenAI API spec.
    """
    checkedList = []
    async with httpx.AsyncClient() as client:
        for item in request.prompts:
            worker_addr = await get_worker_address(item.model, client)

            response = await client.post(
                worker_addr + "/model_details",
                headers=headers,
                json={"model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            context_len = response.json()["context_length"]

            response = await client.post(
                worker_addr + "/count_token",
                headers=headers,
                json={"prompt": item.prompt, "model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            token_num = response.json()["count"]

            can_fit = True
            if token_num + item.max_tokens > context_len:
                can_fit = False

            checkedList.append(
                APITokenCheckResponseItem(
                    fits=can_fit, contextLength=context_len, tokenCount=token_num
                )
            )

    return APITokenCheckResponse(prompts=checkedList)


# TODO: this more or less duplicates create_openapi_chat_completion and we
#       should merge them together. The two request types are similar, the
#       response is the same.
@router.post("/api/v1/chat/completions", tags=["chat"], include_in_schema=False)
async def create_chat_completion(request: APIChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = await get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
        logprobs=request.logprobs,
    )

    if request.repetition_penalty is not None:
        gen_params["repetition_penalty"] = request.repetition_penalty

    error_check_ret = await check_length(
        request, gen_params["prompt"], gen_params["max_new_tokens"]
    )
    if error_check_ret is not None:
        return error_check_ret

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


@router.post("/v1/chat/count_tokens", dependencies=[Depends(check_api_key)], tags=["chat"])
async def count_chat_tokens(request: ChatCompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = await get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
    )

    prompt = gen_params["prompt"]
    max_tokens = gen_params["max_new_tokens"]

    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)

        response = await client.post(
            worker_addr + "/model_details",
            headers=headers,
            json={"model": request.model},
            timeout=WORKER_API_TIMEOUT,
        )
        context_len = response.json()["context_length"]

        response = await client.post(
            worker_addr + "/count_token",
            headers=headers,
            json={"model": request.model, "prompt": prompt},
            timeout=WORKER_API_TIMEOUT,
        )
        token_num = response.json()["count"]

    return {"tokenCount": token_num + max_tokens, "contextLength": context_len, "tokensInHistory": token_num,  "tokensInCompletion": max_tokens}


@router.post("/tokenize", tags=["chat"])
async def tokenize(request: Request):
    """ Tokenize a string and return the tokenized output as a set of input_ids and strings -- this only works
    if the worker implements the tokenize endpoint."""
    data = await request.json()
    model = data["model"]
    text = data["text"]
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model, client)
        response = await client.post(
            worker_addr + "/tokenize",
            headers=headers,
            json={"model": model, "text": text},
            timeout=WORKER_API_TIMEOUT,
        )
        return response.json()
