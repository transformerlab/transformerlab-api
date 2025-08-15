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
import soundfile as sf


from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from fastchat.serve.model_worker import logger
from transformerlab.plugin import WORKSPACE_DIR
from transformers import AutoProcessor, CsmForConditionalGeneration


from unsloth import FastModel
from snac import SNAC
import re


worker_id = str(uuid.uuid4())[:8]

AUDIO_TOKENS_REGEX = re.compile(r"<custom_token_(\d+)>")


from fastchat.serve.base_model_worker import BaseModelWorker  # noqa


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # This function is called when the app shuts down
    cleanup_at_exit()

def convert_to_audio_snac(audio_ids, model):
  audio_ids = torch.tensor(audio_ids, dtype=torch.int32).reshape(-1, 7)
  codes_0 = audio_ids[:, 0].unsqueeze(0)
  codes_1 = torch.stack((audio_ids[:, 1], audio_ids[:, 4])).t().flatten().unsqueeze(0)
  codes_2 = (
    torch.stack((audio_ids[:, 2], audio_ids[:, 3], audio_ids[:, 5], audio_ids[:, 6]))
    .t()
    .flatten()
    .unsqueeze(0)
  )

  with torch.inference_mode():
    audio_hat = model.decode([codes_0, codes_1, codes_2])

  return audio_hat[0]


app = FastAPI(lifespan=lifespan)

class UnslothAudioWorker(BaseModelWorker):
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
        self.model_architecture = model_architecture

        if self.model_architecture == "CsmForConditionalGeneration":
            auto_model = CsmForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_name)

        else:
            auto_model = None
            self.processor = None

        
        self.model, self.tokenizer = FastModel.from_pretrained(
        model_name = self.model_name,
        max_seq_length= self.context_length,
        dtype = None, # Select None for auto detection
        auto_model = auto_model,
        load_in_4bit = False, # Keep this set to False because voice models are small, so we can maintain high quality results.
    )
        
        FastModel.for_inference(self.model) # Enable native 2x faster inference
        self.model = self.model.to(self.device)
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
        
        audio_dir = params.get("audio_dir", None)
        if not audio_dir:
            audio_dir = os.path.join(WORKSPACE_DIR, "audio")
        os.makedirs(name=audio_dir, exist_ok=True)

        # Generate a UUID for this file name:
        file_prefix = str(uuid.uuid4())
        try:
            if self.processor:
                speaker_id = 0
                inputs = self.processor(f"[{speaker_id}]{text}", add_special_tokens=True).to(self.device)

                logger.info(f"we're here {inputs}")
                audio_values = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                # play with these parameters to tweak results
                # depth_decoder_top_k=0,
                # depth_decoder_top_p=0.9,
                # depth_decoder_do_sample=True,
                # depth_decoder_temperature=0.9,
                # top_k=0,
                # top_p=1.0,
                temperature=temperature,
                # do_sample=True,
                output_audio=True
                )
                audio = audio_values[0].to(torch.float32).cpu().numpy()
                output_path = os.path.join(audio_dir, f"{file_prefix}.{audio_format}")
                os.makedirs(audio_dir, exist_ok=True)  # Ensure directory exists
                sf.write(output_path, audio, sample_rate)
                logger.info(f"Audio file written to: {output_path}")
                logger.info("generation is done")
                metadata = {
                    "type": "audio",
                    "text": text,
                    "filename": f"{file_prefix}.{audio_format}",
                    "model": model,
                    "speed": speed,
                    "audio_format": audio_format,
                    "sample_rate": sample_rate,
                    "temperature": temperature,
                    "date": datetime.now().isoformat(),  # Store the real date and time
                }
                metadata_file = os.path.join(audio_dir, f"{file_prefix}.json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

                logger.info(f"Audio successfully generated: {audio_dir}/{file_prefix}.{audio_format}")

                return {
                    "status": "success",
                    "message": f"{audio_dir}/{file_prefix}.{audio_format}",
                }
            
            elif "orpheus" in self.model_name:
                snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                snac_model.to(self.device)
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                
                generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10240,
                #do_sample=True,
                temperature=temperature,
                #top_p=0.95,
                # repetition_penalty=1.1,
                # num_return_sequences=1,
                eos_token_id=128258,  # Orpheus EOS
                use_cache=True
                )
                audio_ids = [
                int(token) - 10 - ((index % 7) * 4096)
                for index, token in enumerate(AUDIO_TOKENS_REGEX.findall(generated_ids))
                ]
                audio = convert_to_audio_snac(audio_ids, snac_model)
                sf.write(os.path.join(audio_dir, file_prefix, f"{file_prefix}.{audio_format}"), audio, 24000)

                metadata = {
                    "type": "audio",
                    "text": text,
                    "filename": f"{file_prefix}.{audio_format}",
                    "model": model,
                    "speed": speed,
                    "audio_format": audio_format,
                    "sample_rate": sample_rate,
                    "temperature": temperature,
                    "date": datetime.now().isoformat(),  # Store the real date and time
                }
                metadata_file = os.path.join(audio_dir, f"{file_prefix}.json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

                logger.info(f"Audio successfully generated: {audio_dir}/{file_prefix}.{audio_format}")

                return {
                    "status": "success",
                    "message": f"{audio_dir}/{file_prefix}.{audio_format}",
                }
        

            else:
                logger.info("Not implemented for this model")
                return {
                    "status": "error",
                    "message": f"Not implemented for this model: {self.model_name}",
                }
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return {
                "status": "error",
                "message": f"Exception during generation: {e}"
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
    parser.add_argument("--model-architecture", type=str, default="MLX")
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
        args.model_architecture,
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
