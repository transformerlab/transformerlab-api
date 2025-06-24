"""
A model worker that executes the model based on SGLang.

Usage:
python3 -m fastchat.serve.sglang_worker --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000 --worker-address http://localhost:30000
"""

import logging
import sys

import sglang.srt.utils
import sglang.srt.entrypoints.engine as engine

import argparse
import asyncio
import json
import os
import multiprocessing
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import sglang as sgl
from sglang.srt.utils import load_image
from sglang.srt.hf_transformers_utils import get_tokenizer, get_config

from fastchat.conversation import IMAGE_PLACEHOLDER_STR
from fastchat.model.model_adapter import get_conversation_template
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import logger

from fastchat.utils import get_context_length, is_partial_stop

import traceback
import torch
import psutil
import signal
import gc
import atexit

app = FastAPI()

sys.setrecursionlimit(8000)


def safe_configure_logger(server_args, prefix=""):
    print(">>> [safe_configure_logger] Overriding broken configure_logger")
    try:
        logging.basicConfig(
            level=logging.ERROR,
            format=f"[%(asctime)s{prefix}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        print(">>> [safe_configure_logger] Logging configured safely")
    except Exception as e:
        print(f">>> [safe_configure_logger] Failed to configure logging: {e}")


sglang.srt.utils.configure_logger = safe_configure_logger
engine.configure_logger = safe_configure_logger


def kill_sglang_subprocesses():
    print(">>> [main] Checking for lingering sglang scheduler subprocesses...")
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline_list = proc.info.get("cmdline")
            if not cmdline_list:  # Handles None or empty list
                continue

            cmdline = " ".join(cmdline_list)
            if "sglang" in cmdline or "sglang::scheduler" in cmdline:
                print(f">>> [main] Killing lingering sglang process: PID {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


# Clear CUDA memory (if CUDA is available)
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        print(">>> [main] Emptying CUDA memory cache and collecting garbage...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def create_model_worker():
    print("Inside Create Model Worker")
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
    print("Args Parsed")
    # Safety for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    print("MultiProcessing Done")

    # Dynamically select attention backend for specific models
    attention_backend = None
    lower_model_path = args.model_path.lower()
    if "llama-3.2" in lower_model_path or "vision" in lower_model_path:
        attention_backend = "flashinfer"
        print(f">>> [create_model_worker] Forcing attention_backend={attention_backend} for model {args.model_path}")

    # Initialize runtime
    try:
        print(">>> Creating SGLang Runtime")
        sys.stdout.flush()

        runtime = sgl.Runtime(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            mem_fraction_static=args.mem_fraction_static,
            tp_size=args.tp_size,
            attention_backend=attention_backend,
        )

        print(">>> Runtime created successfully")
        sys.stdout.flush()

    except Exception:
        print(">>> Exception during Runtime init:")
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print(">>> Runtime created successfully")

    sgl.set_default_backend(runtime)
    conv_template = get_conversation_template(args.model_path).name
    print(f">>> [create_model_worker] Using conv_template: {conv_template}")

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


@sgl.function
def pipeline(s, prompt, max_tokens):
    for p in prompt:
        if isinstance(p, str):
            s += p
        else:
            s += sgl.image(p)
    s += sgl.gen("response", max_tokens=max_tokens)


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

        logger.info(f"Loading the model {self.model_names} on worker {worker_id}, worker type: SGLang worker...")

        self.tokenizer = get_tokenizer(tokenizer_path)
        config = get_config(model_path, trust_remote_code=trust_remote_code)
        try:
            self.context_len = get_context_length(config)
        except (KeyError, AttributeError, TypeError) as e:
            print(f"!!! [SGLWorker] get_context_length failed: {e}. Falling back to config.max_position_embeddings")
            self.context_len = getattr(config, "max_position_embeddings", 2048)  # Safe default

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        if params.get("stop") is True:
            print(">>> [generate_stream] Stop flag detected in params. Performing cleanup...")
            clear_vram()
            kill_sglang_subprocesses()
            raise RuntimeError("Job stopped by user.")

        prompt = params.pop("prompt")
        images = params.get("images", [])
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        presence_penalty = float(params.get("presence_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        echo = params.get("echo", True)

        # Handle stop_str
        stop = []
        if isinstance(stop_str, str) and stop_str != "":
            stop.append(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.extend(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.append(s)

        # make sampling params for sgl.gen
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
            prompt.append(split_prompt[i])
            if i < len(images):
                prompt[-1] = prompt[-1].strip()
                prompt.append(load_image(images[i]))

        state = pipeline.run(
            prompt,
            max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
        )

        entire_output = prompt if echo else ""
        async for out, meta_info in state.text_async_iter(var_name="response", return_meta_data=True):
            partial_stop = any(is_partial_stop(out, i) for i in stop)

            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            entire_output += out
            prompt_tokens = meta_info["prompt_tokens"]
            completion_tokens = meta_info["completion_tokens"]

            ret = {
                "text": entire_output,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "error_code": 0,
            }
            yield ret

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


def cleanup_on_exit():
    print(">>> [model_worker] Cleanup on exit triggered.")
    kill_sglang_subprocesses()
    clear_vram()


if __name__ == "__main__":
    print("Inside Main Worker, Calling Create Model Worker")
    args, worker = create_model_worker()

    # Register atexit hook BEFORE long-running uvicorn call
    atexit.register(cleanup_on_exit)

    # Register cleanup signal handlers
    def handle_exit(signum, frame):
        print(f">>> [model_worker] Caught signal {signum}, cleaning up...")
        cleanup_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Start the server (this blocks until shutdown)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
