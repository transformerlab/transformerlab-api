"""
A model worker using Apple MLX Audio
"""

import os
import sys
import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import List
import json
from datetime import datetime
import uvicorn
import torch

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from fastchat.serve.model_worker import logger
from transformerlab.plugin import WORKSPACE_DIR, get_model_class_from_architecture


from unsloth import FastLanguageModel
from snac import SNAC



worker_id = str(uuid.uuid4())[:8]

from fastchat.serve.base_model_worker import BaseModelWorker  # noqa


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # This function is called when the app shuts down
    cleanup_at_exit()


app = FastAPI(lifespan=lifespan)


class UnslothAudioWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        context_length: int,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker" + f"{worker_id}, worker type: MLX Audio worker..."
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.model_name = model_path
        self.context_length = context_length


        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name = self.model_name,
        max_seq_length= self.context_length,
        dtype = None, # Select None for auto detection
        auto_model = get_model_class_from_architecture(self.model_name),
        load_in_4bit = False, # Keep this set to False because voice models are small, so we can maintain high quality results.
    )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

        # snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz") # Should we remove hardcoded model name?
        # self.snac_model = snac_model.to(self.device)

        if not no_register:
            self.init_heart_beat()

    async def generate(self, params):
        self.call_ct += 1

        text = params.get("text", "")
        model = params.get("model", None)
        speed = params.get("speed", 1.0)
        # file_prefix = params.get("file_prefix", "audio")
        audio_format = params.get("audio_format", "wav")
        sample_rate = params.get("sample_rate", 24000)
        temperature = params.get("temperature", 0.0)
        stream = params.get("stream", False)
        
        audio_dir = params.get("audio_dir", None)
        if not audio_dir:
            audio_dir = os.path.join(WORKSPACE_DIR, "audio")
        os.makedirs(name=audio_dir, exist_ok=True)

        # Generate a UUID for this file name:
        file_prefix = str(uuid.uuid4())

        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.device)

        audio_values = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1200,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.95,
            # repetition_penalty=1.1,
            # num_return_sequences=1,
            eos_token_id=128258,  # Orpheus EOS
            # use_cache=True
        )

        logger.info(f"Generated audio values: {audio_values}")

        # try:
        #     generate_audio(
        #         text=text,
        #         model_path=model,
        #         speed=speed,
        #         file_prefix=os.path.join(audio_dir, file_prefix),
        #         sample_rate=sample_rate,
        #         join_audio=True,  # Whether to join multiple audio files into one
        #         verbose=True,  # Set to False to disable print messages
        #         temperature=temperature,
        #         stream=stream,
        #         voice=None,
        #     )

        #     # Also save the parameters and metadata used to generate the audio
        #     metadata = {
        #         "type": "audio",
        #         "text": text,
        #         "filename": f"{file_prefix}.{audio_format}",
        #         "model": model,
        #         "speed": speed,
        #         "audio_format": audio_format,
        #         "sample_rate": sample_rate,
        #         "temperature": temperature,
        #         "date": datetime.now().isoformat(),  # Store the real date and time
        #     }
            
        #     metadata_file = os.path.join(audio_dir, f"{file_prefix}.json")
        #     with open(metadata_file, "w") as f:
        #         json.dump(metadata, f)

        #     logger.info(f"Audio successfully generated: {audio_dir}/{file_prefix}.{audio_format}")

        #     return {
        #         "status": "success",
        #         "message": f"{audio_dir}/{file_prefix}.{audio_format}",
        #     }
        # except Exception:
        #     logger.error(f"Error generating audio: {audio_dir}/{file_prefix}.{audio_format}")
        #     return {
        #         "status": "error",
        #         "message": f"Error generating audio: {audio_dir}/{file_prefix}.{audio_format}",
        #     }


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
    # logger.debug("Trying to abort but not implemented")
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
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--parameters", type=str, default="{}")
    


    args, unknown = parser.parse_known_args()

    try:
        parameters = json.loads(args.parameters)
        context_length = int(parameters.get("context_length", "2048"))
    except Exception:
        context_length = 2048

    if args.model_path:
        args.model = args.model_path

    worker = UnslothAudioWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        1024,
        False,
        context_length
    )

    # Restore original stdout/stderr to prevent logging recursion
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
