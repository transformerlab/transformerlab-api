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

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.conversation import IMAGE_PLACEHOLDER_STR
from fastchat.model.model_adapter import get_conversation_template
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import logger

from fastchat.utils import get_context_length

import traceback


def safe_configure_logger(server_args, prefix=""):
    print(">>> [safe_configure_logger] Overriding broken configure_logger")
    try:
        logging.basicConfig(
            level=logging.ERROR,
            format=f"[%(asctime)s{prefix}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        print(">>> [safe_configure_logger] Logging configured safely")
    except Exception:
        print(">>> [safe_configure_logger] Failed to configure logging")


import sglang.srt.utils  # noqa: E402

sglang.srt.utils.configure_logger = safe_configure_logger
import sglang.srt.entrypoints.engine as engine  # noqa: E402

engine.configure_logger = safe_configure_logger
import sglang as sgl  # noqa: E402
from sglang.srt.hf_transformers_utils import get_tokenizer, get_config  # noqa: E402

app = FastAPI()


class EngineWithChatTemplate:
    def __init__(self, engine, template_name):
        self._engine = engine
        self._chat_template = template_name

    def __getattr__(self, key):
        return getattr(self._engine, key)

    def get_chat_template(self):
        return self._chat_template


def run_generate_in_thread(engine, prompt, image_data, sampling_params):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return engine.generate(
            prompt=prompt,
            image_data=image_data,
            sampling_params=sampling_params,
        )
    finally:
        loop.close()


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

        engine_kwargs = dict(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            mem_fraction_static=args.mem_fraction_static,
            tp_size=args.tp_size,
            attention_backend=attention_backend,
            enable_multimodal=True,
        )

        if is_multimodal_model:
            engine_kwargs["load_image"] = True  # only pass this if supported

        runtime = EngineWithChatTemplate(sgl.Engine(**engine_kwargs), conv_template)

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
        runtime: sgl.Engine,
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

        logger.info(f"Loading the model {self.model_names} on worker {worker_id}, worker type: SGLang worker...")

        self.tokenizer = get_tokenizer(tokenizer_path)
        config = get_config(model_path, trust_remote_code=trust_remote_code)
        try:
            self.context_len = get_context_length(config)
        except (KeyError, AttributeError, TypeError):
            self.context_len = getattr(config, "max_position_embeddings", 2048)  # Safe default

        self.runtime = runtime

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        try:
            self.call_ct += 1

            prompt: str = params.pop("prompt")
            images = params.get("images", [])

            if prompt.count(IMAGE_PLACEHOLDER_STR) != len(images):
                raise ValueError("Mismatched <image> tokens vs. images")

            temperature = float(params.get("temperature", 1.0))
            top_p = float(params.get("top_p", 1.0))
            top_k = params.get("top_k", -1.0)
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

            sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_new_tokens": max_new_tokens,
                "stop": stop,
            }

            # split prompt by image token
            split_prompt = prompt.split(IMAGE_PLACEHOLDER_STR)
            if prompt.count(IMAGE_PLACEHOLDER_STR) != len(images):
                raise ValueError(
                    "The number of images passed in does not match the number of <image> tokens in the prompt!"
                )
            prompt = IMAGE_PLACEHOLDER_STR.join(split_prompt)

            # SGLang VLM only supports one image currently
            if len(images) > 1:
                raise ValueError("Multiple images not supported in current VLM mode.")
            image_data = images[0] if images else None

            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None, lambda: run_generate_in_thread(self.runtime, prompt, image_data, sampling_params)
            )

            ret = {
                "text": output["text"],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "error_code": 0,
            }
            yield ret

        except Exception as e:
            raise ValueError(f"Failed: {e}")

    async def generate_stream_gate(self, params):
        try:
            async for ret in self.generate_stream(params):
                yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
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
