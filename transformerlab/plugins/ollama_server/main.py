"""
Ollama model worker

Requires that ollama is installed on your server.

Right now only generate_stream function has gone through much testing.
The generate function probably needs work.
"""

import argparse
import asyncio
import os
import subprocess
import json
import uuid
from hashlib import sha256
from typing import List
from pathlib import Path
import atexit
import traceback
import uvicorn

import ollama

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import is_partial_stop

from transformers.tokenization_utils_base import BatchEncoding


app = FastAPI()
worker = None


class OllamaTokenizer:
    """
    This is a total hack tokenizer just to get things to proceed.
    It doesn't do tokenization!
    """

    def __init__(self, model):
        self.model = model
        self.eos_token_id = None

    def __call__(self, text):
        # convert variable "text" to bytes:
        text = text.encode("utf-8")

        # TODO: Ollama has recently added tokenizer as an experimental feature
        # The current code is a fake tokenizer which is ignored by the plugin
        tokens = []
        # tokens = self.model.tokenize(text)
        batchEncoding = BatchEncoding(
            data={"input_ids": [tokens], "eos_token_id": None})
        return batchEncoding

    def decode(self, tokens):
        # This is fake code that does not detokenize. See above.
        # return self.model.detokenize(tokens)
        return [''.join(tokens)]

    def num_tokens(self, prompt):
        # Also fake. This generates a totally fake approximate number.
        # tokens = self.model.tokenize(prompt)
        # return (len(tokens))
        return len(prompt)//4


class OllamaServer(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: ollama-python..."
        )

        # We need to find the ollama cache or else this isn't going to work
        # First, check the OLLAMA_MODELS environment variable
        # If that isn't set then use the default location:
        # ~/.ollama/models
        OLLAMA_MODELS_DIR = os.getenv(
            "OLLAMA_MODELS", os.path.join(Path.home(), ".ollama", "models"))
        if os.path.isdir(OLLAMA_MODELS_DIR):
            print("Ollama models directory:", OLLAMA_MODELS_DIR)
        else:
            raise FileNotFoundError(
                f"Ollama models directory not found at: {OLLAMA_MODELS_DIR}")

        # Start the ollama client.
        # This will check if ollama is installed and running and if not return an error.
        # TODO: We should maybe first call ollama serve to start ollama for the user?
        # i.e. instead of throwing an error and making them do it?
        self.model = ollama.Client()

        # Load model into Ollama
        #
        # EXPLANATION:
        # Our GGUF models are stored in the transformerlab workspace models directory.
        # Ollama lets you import models if you have a correctly formatted Modelfile.
        # But, Ollama wants models stored in their proprietary way in ~/.ollama.
        # If you try to import a GGUF model outside of Ollama, it will copy the
        # entire file into their .ollama cache and waste your disk space.
        # However, if there is already a file (or link) with the correct name (SHA blob)
        # in the right place (under ~/.ollama/models/blobs) then it won't copy!

        # STEP 1: Make an Ollama Modelfile that points to the GGUF you want to run
        # Split model_path into the directory and filename
        model_dir, model_filename = os.path.split(model_path)

        # Our convention is that GGUF models have the same name as their filename
        self.model_name = model_filename

        # Output a modelfile
        modelfile = os.path.join(model_dir, "Modelfile")
        with open(modelfile, "w") as file:
            file.write(f"FROM {model_path}")

        # STEP 2: Create a link to our GGUF file in the Ollama cache
        # to prevent it from copying the GGUF file.

        # 2a. Figure out the SHA filename ollama expects.
        # Copied this from ollama SDK
        sha256sum = sha256()
        with open(model_path, 'rb') as r:
            while True:
                chunk = r.read(32 * 1024)
                if not chunk:
                    break
                sha256sum.update(chunk)

        # 2b. Create a link with the SHA name to the actual GGUF file
        OLLAMA_MODEL_BLOBS_CACHE = os.path.join(OLLAMA_MODELS_DIR, "blobs")
        sha_filename = os.path.join(
            OLLAMA_MODEL_BLOBS_CACHE,
            f'sha256:{sha256sum.hexdigest()}'
        )

        # Create the directory if it doesn't exist
        os.makedirs(OLLAMA_MODEL_BLOBS_CACHE, exist_ok=True)

        # Create a symbolic link if it doesn't already exist
        if not os.path.exists(sha_filename):
            print("Creating link to model in Ollama:", sha_filename)
            os.symlink(model_path, sha_filename)

        # STEP 3: Call ollama and tell it to create the model in its register
        # TODO: I think you can do this via the SDK which would be better
        # for catching errors
        subprocess.run([
            "ollama",
            "create",
            self.model_name,
            "-f",
            modelfile
        ])

        # STEP 4: Start the model!
        # You load a model into memory in ollama by not passing a prompt to model.generate
        load_model = self.model.generate(
            model=self.model_name,
        )
        print(load_model)

        # HACK: We don't really have access to the tokenization in ollama
        # But we need a tokenizer to work with fastchat
        self.tokenizer = OllamaTokenizer(model=self.model)

        # Fastchat needs to know context length to check for context overflow
        # You can try pulling this from modelinfo from ollama.show
        # As a backup, we will assume ollama default of 4096
        self.context_len = 4096
        show_response: ollama.ShowResponse = ollama.show(
            model=self.model_name
        )
        modelinfo = show_response.modelinfo
        print(modelinfo)
        model_architecture = modelinfo.get("general.architecture", None)
        if model_architecture:
            context_key = f"{model_architecture}.context_length"
            if context_key in modelinfo:
                self.context_len = modelinfo[context_key]

        print("Setting context length to", self.context_len)

        # For debugging: Output a bunch of model info
        response: ollama.ProcessResponse = ollama.ps()
        for model in response.models:
            print('Model: ', model.model)
            print('  Digest: ', model.digest)
            print('  Expires at: ', model.expires_at)
            print('  Size: ', model.size)
            print('  Size vram: ', model.size_vram)
            print('  Details: ', model.details)
            print('\n')

        self.init_heart_beat()

    async def generate_stream(self, params):

        self.call_ct += 1

        context = params.pop("prompt")
        params.pop("request_id")    # not used

        # Generation parameters
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))

        # These parameters don't seem to be in the UI
        # top_k = params.get("top_k", -1.0)
        # presence_penalty = float(params.get("presence_penalty", 0.0))

        # Create a set out of our stop_str parameter
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # If we add tokenizer we can add this in later
        # Add stop tokens to set of stop strings
        # stop_token_ids = params.get("stop_token_ids", None) or []
        # if self.tokenizer.eos_token_id is not None:
        #    stop_token_ids.append(self.tokenizer.eos_token_id)
        # for tid in stop_token_ids:
        #    if tid is not None:
        #        s = self.tokenizer.decode(tid)
        #        if s != "":
        #            stop.add(s)

        print("Stop patterns: ", stop)

        # Make sure top_p is above some minimum
        # And set to 1.0 if temperature is effectively 0
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # Bundle together generation parameters
        # TODO: Add num_gpu and figure out num_ctx?
        # TODO: Add stop set so we don't have to manually check
        generation_params = {
            "top_p": top_p,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty
        }

        decoded_tokens = []

        # context_tokens = self.model.tokenize(context.encode("utf-8"))

        finish_reason = "length"

        # TODO: Should this use generate?
        iterator = await run_in_threadpool(self.model.chat,
                                           model=self.model_name,
                                           messages=[
                                               {'role': 'user', 'content': context}],
                                           stream=True,
                                           options=generation_params)

        for i in range(max_new_tokens):
            # Try to get next token.
            # If the generator hits a stop the interator finishes and throws:
            # RuntimeError: coroutine raised StopIteration
            try:
                response = await run_in_threadpool(next, iterator)
            except RuntimeError as e:
                print(e)
                print(traceback.format_exc())
                finish_reason = "stop"
                break

            # Check if ollama returned a stop
            if response.get('done'):
                finish_reason = response.get('done_reason', "")
                break

            # Normally we'd add a response token to a list of tokens and detokenize
            # But ollama is detokenizing for us
            decoded_token = response['message']['content']
            decoded_tokens.append(decoded_token)
            tokens_decoded_str = ''.join(decoded_tokens)

            # Check for stop string
            # Note that ollama can do this if we just pass stop to it correctly
            partial_stop = any(is_partial_stop(
                tokens_decoded_str, i) for i in stop)
            if partial_stop:
                finish_reason = "stop"
                break

            ret = {
                "text": tokens_decoded_str,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": len(context),
                    "completion_tokens": len(decoded_tokens),
                    "total_tokens": len(context) + len(decoded_tokens),
                },
                "cumulative_logprob": [
                ],
                "finish_reason": None   # hard code for now
            }
            yield (json.dumps(ret) + "\0").encode()

        ret = {
            "text": ''.join(decoded_tokens),
            "error_code": 0,
            "usage": {
            },
            "cumulative_logprob": [
            ],
            "finish_reason": finish_reason
        }
        yield (json.dumps(obj={**ret, **{"finish_reason": None}}) + "\0").encode()
        yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        prompt = params.pop("prompt")

        # TODO: Figure out what to do with max_tokens
        # max_tokens = params.get("max_new_tokens", 256)

        # Setup parameters
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        params = {
            "top_p": top_p,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty
        }

        print("Generating with params: ", params)
        thread = asyncio.to_thread(self.model.generate,
                                   model=self.model_name,
                                   prompt=prompt,
                                   stream=False,
                                   options=params)

        response = await thread

        ret = {
            "text": response['repsonse'],
            "error_code": 0,
            "usage": {
            },
            "cumulative_logprob": [
            ],
            "finish_reason": response['done_reason']
        }
        return ret

    def stop_server(self):
        """
        Called by cleanup_at_exit.
        """
        # You can unload a model by not passing a prompt to generate
        # and setting keep_alive to 0
        self.model.generate(
            model=self.model_name,
            keep_alive=0
        )


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


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    output = await worker.generate(params)
    release_worker_semaphore()

    # TODO: WHat is this? Inherited from other servers
    # # await engine.abort(request_id)
    # print("Trying to abort but not implemented")
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


def cleanup_at_exit():
    global worker
    print("Cleaning up...")
    worker.stop_server()
    del worker


atexit.register(cleanup_at_exit)


def main():
    global app, worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
                        default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str,
                        default="llama3")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--parameters", type=str, default=None)

    args, _ = parser.parse_known_args()

    # No parameters right now. Uncomment when we add some.
    # parameters is a JSON string, so we parse it:
    # parameters = json.loads(args.parameters)

    # model_path can be a hugging face ID or a local file in Transformer Lab
    # But GGUF is always stored as a local path because
    # we are using a specific GGUF file
    # TODO: Make sure the path exists before continuing
    # if os.path.exists(args.model_path):
    model_path = args.model_path

    worker = OllamaServer(
        args.controller_address,
        args.worker_address,
        worker_id,
        model_path,
        args.model_names,
        1024,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
