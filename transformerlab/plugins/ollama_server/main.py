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
from typing import List
import atexit
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

from transformers.tokenization_utils_base import BatchEncoding


app = FastAPI()
worker = None


class OllamaTokenizer:
    """
    TODO: This is a total hack tokenizer just to get things to proceed.
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
        #tokens = self.model.tokenize(text)
        batchEncoding = BatchEncoding(
            data={"input_ids": [tokens], "eos_token_id": None})
        return batchEncoding

    def decode(self, tokens):
        # TODO: This is fake code that does not detokenize. See above.
        #return self.model.detokenize(tokens)
        return [''.join(tokens)]

    def num_tokens(self, prompt):
        # TODO: Also fake. This generates a totally fake approximate number.
        #tokens = self.model.tokenize(prompt)
        #return (len(tokens))
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

        # Start the ollama client.
        # This will check if ollama is installed and running and if not return an error.
        # TODO: We should maybe first call ollama serve to start ollama for the user?
        self.model = ollama.Client()

        # How to load a model in ollama:
        # Our GGUF models are stored in the transformerlab workspace models directory.
        # Ollama models must be stored in their proprietary way in ~/.ollama
        # Ollama will load the model there for you from anywhere on your computer iff
        # you have a correctly formatted Modelfile, which is very simple.
        # So...we need to make a Modelfile in the directory with OUR GGUF model.
        # TODO: Don't reimport if model already exists in ollama. It's slow!
        model_path = model_path

        # Split model_path into the directory and filename
        model_dir, model_filename = os.path.split(model_path)

        # Our convention is that GGUF models have the same name as their filename
        self.model_name = model_filename

        # Output a modelfile
        modelfile = os.path.join(model_dir, "Modelfile")
        with open(modelfile, "w") as file:
            file.write(f"FROM {model_path}")

        # Then we need to call ollama and tell it to create the model
        subprocess.run([
            "ollama",
            "create",
            self.model_name,
            "-f",
            modelfile
        ])

        # You load a model into memory in ollama by not passing a prompt to model.generate
        load_model = self.model.generate(
            model=self.model_name,
        )
        print(load_model)

        # HACK: We don't really have access to the tokenization in ollama
        # But we need a tokenizer to work with fastchat
        self.tokenizer = OllamaTokenizer(model=self.model)

        # Fastchat needs to know context length to check for context overflow
        # TODO: No idea how to get/set this but the ollama default is 4096?
        self.context_len = 4096

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
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))

        # TODO: These parameters are ignored currently
        #top_k = params.get("top_k", -1.0)
        #presence_penalty = float(params.get("presence_penalty", 0.0))
        #frequency_penalty = float(params.get("frequency_penalty", 0.0))

        # TODO: We don't handle reading in stop strings
        #stop_str = params.get("stop", None)
        #stop_token_ids = params.get("stop_token_ids", None) or []

        # TODO: Tokenizer setup might be needed after we add tokenizer
        #if self.tokenizer.eos_token_id is not None:
        #    stop_token_ids.append(self.tokenizer.eos_token_id)

        # TODO: Do I need it?
        # Handle stop_str
        """
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                print("Stop token: ", tid)
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        print("Stop patterns: ", stop)
        """

        # Make sure top_p is above some minimum
        # And set to 1.0 if temperature is effectively 0
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # TODO: Add the parameters here
        generation_params = None 

        decoded_tokens = []

        # context_tokens = self.model.tokenize(context.encode("utf-8"))

        finish_reason = "length"

        # TODO: Should this use generate?
        # TODO: Pass in all of the parameters
        iterator = await run_in_threadpool(self.model.chat, 
                                           model=self.model_name,
                                           messages=[{'role': 'user', 'content': context}],
                                           stream=True,
                                           options=generation_params)

        for i in range(max_new_tokens):
            # Try to get next token. 
            # If the generator hits a stop the interator finishes and throws:
            # RuntimeError: coroutine raised StopIteration
            try:
                response = await run_in_threadpool(next, iterator)
            except:
                print("Ollama server encountered StopIteration")
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

        # TODO: Get options and pass to generate
        #max_tokens = params.get("max_new_tokens", 256)
        #temperature = float(params.get("temperature", 1.0))
        #top_p = float(params.get("top_p", 1.0))
        params = None

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
    #print("Trying to abort but not implemented")
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

    # parameters is a JSON string, so we parse it:
    parameters = json.loads(args.parameters)

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
