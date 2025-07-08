"""
A model worker that executes the model based on SGLang.

Usage:
python3 -m fastchat.serve.sglang_worker --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000 --worker-address http://localhost:30000
"""

import logging
import sys

import argparse
import asyncio
import json
import os
import multiprocessing
from typing import List
from pathlib import Path
import base64
from uuid import uuid4
import shutil
import re

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.conversation import IMAGE_PLACEHOLDER_STR
from fastchat.model.model_adapter import get_conversation_template
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
import fastchat.serve.base_model_worker

from fastchat.utils import get_context_length, is_partial_stop

import traceback


def setup_model_worker_logger(name: str = "transformerlab") -> logging.Logger:
    """
    Set up a clean logger for the model worker without duplicating handlers.
    """
    if "TFL_HOME_DIR" in os.environ:
        HOME_DIR = os.environ["TFL_HOME_DIR"]
        if not os.path.exists(HOME_DIR):
            print(f"Creating home directory: {HOME_DIR}")
            os.makedirs(HOME_DIR, exist_ok=True)
    else:
        HOME_DIR = Path.home() / ".transformerlab"
        os.makedirs(HOME_DIR, exist_ok=True)

    log_path = os.path.join(HOME_DIR, "transformerlab.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent log duplication to root logger

    # Prevent adding multiple handlers
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Patch FastChat's logger
    fastchat.serve.base_model_worker.logger = logger

    return logger


logger = setup_model_worker_logger()


def safe_configure_logger(server_args, prefix=""):
    print(">>> [safe_configure_logger] Overriding broken configure_logger")
    pass


import sglang.srt.utils  # noqa: E402

sglang.srt.utils.configure_logger = safe_configure_logger

import sglang as sgl  # noqa: E402
from sglang.srt.hf_transformers_utils import get_tokenizer, get_config  # noqa: E402


app = FastAPI()

workspace = os.environ["_TFL_WORKSPACE_DIR"]
TMP_IMG_DIR = Path(f"{workspace}/plugins/sglang_server/tmp_img")


def get_assistant_tokens(conv, tokenizer):
    assistant_role = conv.roles[1]
    conv.messages = []
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], "Hi! How can I help you today?")
    prompt = conv.get_prompt()

    # Find the last assistant prefix
    assistant_lines = [line for line in prompt.splitlines() if assistant_role.lower() in line.lower()]
    if not assistant_lines:
        raise ValueError("Could not locate assistant prompt line")

    start_text = assistant_lines[-1].strip()

    start_token = tokenizer.convert_tokens_to_string(tokenizer.tokenize(start_text)).strip()
    end_token = tokenizer.eos_token or tokenizer.sep_token or None

    if end_token:
        last_assistant_re = re.compile(re.escape(start_token) + r"\s*(.*?)" + re.escape(end_token), re.S)
    else:
        last_assistant_re = re.compile(re.escape(start_token) + r"\s*(.*)", re.S)

    return start_token, end_token, last_assistant_re


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--model-names", type=lambda s: s.split(","), default=None)
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-template", type=str, default="default")
    parser.add_argument("--trust-remote-code", action="store_false", default=True)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--multimodal", action="store_true", default=False)
    parser.add_argument("--worker-id", type=str, default="sglang-worker")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated list of GPU IDs")

    args = parser.parse_args()
    # Set defaults
    args.tp_size = args.num_gpus if args.num_gpus > 1 else 1
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path

    # Safety for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # Dynamically select attention backend for specific models
    lower_model_path = args.model_path.lower()

    if "gemma" in lower_model_path or "google" in lower_model_path:
        attention_backend = None
    elif "llama-3.2" in lower_model_path or "vision" in lower_model_path:
        attention_backend = "flashinfer"
    else:
        attention_backend = None

    conv_template = get_conversation_template(args.model_path).name
    # Initialize runtime
    try:
        sys.stdout.flush()

        is_multimodal_model = check_if_multimodal(args.model_path)

        runtime_kwargs = dict(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            mem_fraction_static=args.mem_fraction_static,
            tp_size=args.tp_size,
            attention_backend=attention_backend,
            enable_multimodal=True,
        )

        if is_multimodal_model:
            runtime_kwargs["load_image"] = True  # only pass this if supported

        runtime = sgl.Runtime(**runtime_kwargs)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise

    sgl.set_default_backend(runtime)

    # Instantiate worker
    worker = SGLWorker(
        args.controller_address,
        args.worker_address,
        args.worker_id,
        args.model_path,
        args.tokenizer_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        conv_template,
        runtime,
        args.trust_remote_code,
    )

    return args, worker


def check_if_multimodal(model_path: str) -> bool:
    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        return False
    with open(config_file, "r") as f:
        config = json.load(f)
    return "image_token_index" in config or config.get("_from_multimodal", False)


class SGLWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        tokenizer_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        runtime: sgl.Runtime,
        trust_remote_code: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            check_if_multimodal(model_path),
        )

        self.tokenizer = get_tokenizer(tokenizer_path)
        config = get_config(model_path, trust_remote_code=trust_remote_code)
        try:
            self.context_len = get_context_length(config)
        except (KeyError, AttributeError, TypeError):
            self.context_len = getattr(config, "max_position_embeddings", 2048)  # Safe default

        self.model_path = model_path
        self.runtime = runtime

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        try:
            self.call_ct += 1

            conv_template = get_conversation_template(self.model_path)
            start_token, end_token, LAST_ASSISTANT_RE = get_assistant_tokens(conv_template, tokenizer=self.tokenizer)

            prompt: str = params.pop("prompt")
            images = params.get("images", [])
            if start_token in prompt:
                is_chat = True
            else:
                is_chat = False
            # echo = params.get("echo", True)
            image_paths = []
            if len(images) > 0:
                if os.path.exists(TMP_IMG_DIR):
                    shutil.rmtree(TMP_IMG_DIR)
                os.makedirs(TMP_IMG_DIR, exist_ok=True)

                for i, b64_img in enumerate(images):
                    header, encoded = b64_img.split(",", 1)
                    ext = header.split("/")[1].split(";")[0]
                    img_data = base64.b64decode(encoded)
                    img_path = os.path.join(TMP_IMG_DIR, f"{uuid4()}-image_{i}.{ext}")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    image_paths.append(img_path)

            if prompt.count(IMAGE_PLACEHOLDER_STR) != len(images):
                raise ValueError("Mismatched <image> tokens vs. images")

            temperature = float(params.get("temperature", 1.0))
            top_p = float(params.get("top_p", 1.0))
            top_k = params.get("top_k", -1.0)
            min_p = params.get("min_p", 0.0)
            frequency_penalty = float(params.get("frequency_penalty", 0.0))
            presence_penalty = float(params.get("presence_penalty", 0.0))
            max_new_tokens = int(params.get("max_new_tokens", 256))
            stop_str = params.get("stop", None)
            stop_token_ids = params.get("stop_token_ids", None) or []
            # echo = params.get("echo", True)

            # Collect stop sequences
            stop = []
            if isinstance(stop_str, str) and stop_str.strip():
                stop.append(stop_str)
            elif isinstance(stop_str, list):
                stop.extend([s for s in stop_str if isinstance(s, str) and s.strip()])

            for tid in stop_token_ids:
                if tid is not None:
                    s = self.tokenizer.decode(tid)
                    if s:
                        stop.append(s)

            # Prepare sampling params
            top_p = max(top_p, 1e-5)
            if temperature <= 1e-5:
                top_p = 1.0

            # split prompt by image token
            split_prompt = prompt.split(IMAGE_PLACEHOLDER_STR)
            if prompt.count(IMAGE_PLACEHOLDER_STR) != len(images):
                raise ValueError(
                    "The number of images passed in does not match the number of <image> tokens in the prompt!"
                )
            prompt = []

            for i in range(len(split_prompt)):
                text_part = split_prompt[i].strip()
                if text_part:
                    prompt.append(text_part)

                if i < len(image_paths):
                    prompt.append(image_paths[i])  # actual path, not "<image>"

            @sgl.function
            def pipeline(s, prompt_parts, max_tokens, is_chat: bool):
                if is_chat:
                    user_msg = ""
                    for part in prompt_parts:
                        if os.path.isfile(part):
                            user_msg = sgl.image(str(part))
                        else:
                            user_msg += str(part)
                    s += sgl.user(user_msg)
                    s += sgl.assistant(sgl.gen(max_tokens=max_tokens))
                else:
                    raw_prompt = ""
                    for part in prompt_parts:
                        if os.path.isfile(part):
                            raw_prompt += sgl.image(str(part))  # register + embed token
                        else:
                            raw_prompt += str(part)
                    s += raw_prompt
                    s += sgl.gen(max_tokens=max_tokens)

            state = pipeline.run(
                prompt,
                max_new_tokens,
                is_chat=is_chat,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True,
            )

            stream_iter = state.text_async_iter()

            assistant_started = False
            assistant_text = ""

            if is_chat:
                async for out in stream_iter:
                    if any(is_partial_stop(out, i) for i in stop):
                        continue

                    if not assistant_started:
                        matches = LAST_ASSISTANT_RE.findall(out)
                        if matches:
                            assistant_started = True
                            chunk = matches[-1]
                            assistant_text += chunk
                            yield {
                                "text": assistant_text,
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                "error_code": 0,
                            }
                        continue

                    # If assistant has started, just keep appending normally
                    assistant_text += out.replace(start_token, "").replace(end_token, "")
                    yield {
                        "text": assistant_text,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "error_code": 0,
                    }
            else:
                prompt_str = "".join(part for part in prompt if isinstance(part, str)).strip().lower()
                prompt_str_nospace = prompt_str.replace(" ", "")

                assistant_started = False

                async for out in stream_iter:
                    if any(is_partial_stop(out, i) for i in stop):
                        continue

                    out_strip = out.strip().lower().replace(" ", "")

                    if not assistant_started:
                        if out_strip.startswith(prompt_str_nospace):
                            # Skip the prompt echo
                            out = out[len(prompt_str) :]
                        assistant_started = True

                    assistant_text += out
                    yield {
                        "text": assistant_text,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "error_code": 0,
                    }

        except Exception as e:
            raise ValueError(f"Failed: {e}")

    async def generate_stream_gate(self, params):
        try:
            async for ret in self.generate_stream(params):
                yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError):
            ret = {
                "text": f"{SERVER_ERROR_MSG})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    async def generate_gate(self, params):
        async for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    args, worker = create_model_worker()

    # run uvicorn directly
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
