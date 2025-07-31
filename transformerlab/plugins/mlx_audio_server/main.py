"""
A model worker using Apple MLX Audio
"""

import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from fastchat.serve.model_worker import logger

from mlx_audio.tts.generate import generate_audio

worker_id = str(uuid.uuid4())[:8]

from fastchat.serve.base_model_worker import BaseModelWorker  # noqa


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # This function is called when the app shuts down
    cleanup_at_exit()


app = FastAPI(lifespan=lifespan)


class MLXAudioWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        model_architecture: str,
        limit_worker_concurrency: int,
        no_register: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
        )

        logger.info(f"Loading the model {self.model_names} on worker" + f"{worker_id}, worker type: MLX Audio worker...")
        logger.info(f"Model architecture: {model_architecture}")

        self.model_name = model_path
        
        if not no_register:
            self.init_heart_beat()


    async def generate(self, params):
        self.call_ct += 1

        text = params.get("text", "")
        model = params.get("model", None) # temporarily only for testing
        speed = params.get("speed", 1.0)
        voice = params.get("voice", "af_heart")
        lang_code = params.get("lang_code", "a")
        file_prefix = params.get("file_prefix", "audio")
        audio_format = params.get("audio_format", "wav")
        sample_rate = params.get("sample_rate", 24000)
        temperature = params.get("temperature", 0.0)
        ref_text = params.get("ref_text", None)
        ref_audio = params.get("ref_audio", None)
        logger.info("heeyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")

        generate_audio(
            text=text,
            model_path=model,
            voice=voice,
            speed=speed,
            lang_code=lang_code, # The language code
            file_prefix=file_prefix,
            audio_format=audio_format,
            sample_rate=sample_rate,
            join_audio=True,  # Whether to join multiple audio files into one
            verbose=False,  # Set to False to disable print messages
            #temperature=temperature,
            #ref_text=ref_text,  # Caption for reference audio
            #ref_audio=ref_audio,  # Reference audio you would like to clone the voice from
            stream=False,

)

        logger.info(f"{file_prefix} successfully generated!")

        return {
            "status": "success",
            "message": f"{file_prefix}.{audio_format}",
        }


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        print("trying to abort but not implemented")

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    output = await worker.generate(params)
    release_worker_semaphore()
    # await engine.abort(request_id)
    #logger.debug("Trying to abort but not implemented")
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


def cleanup_at_exit():
    global worker
    print("Cleaning up...")
    del worker


def main():
    global app, worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="microsoft/phi-2")
    parser.add_argument("--model-architecture", type=str, default="MLX")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) whendownloading the model and tokenizer.",
    )
    parser.add_argument("--parameters", type=str, default="{}")
    parser.add_argument("--plugin_dir", type=str)

    args, unknown = parser.parse_known_args()

    if args.model_path:
        args.model = args.model_path

    worker = MLXAudioWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.model_architecture,
        1024,
        False,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
