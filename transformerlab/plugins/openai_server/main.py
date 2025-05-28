"""
OpenAI model worker

Requires that openai is installed on your server.

"""

import argparse
import asyncio
import os
import json
import uuid
from contextlib import asynccontextmanager
import uvicorn

import openai

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from transformers.tokenization_utils_base import BatchEncoding
from fastchat.utils import build_logger


worker_id = str(uuid.uuid4())[:8]
logfile_path = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "logs")
if not os.path.exists(logfile_path):
    os.makedirs(logfile_path)
logger = build_logger("model_worker", os.path.join(logfile_path, "model_worker.log"))

import fastchat.serve.base_model_worker  # noqa: E402

fastchat.serve.base_model_worker.logger = logger
from fastchat.serve.base_model_worker import BaseModelWorker  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # This function is called when the app shuts down
    cleanup_at_exit()


app = FastAPI(lifespan=lifespan)
worker = None

class OpenAITokenizer:
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
        batchEncoding = BatchEncoding(data={"input_ids": [tokens], "eos_token_id": None})
        return batchEncoding

    def decode(self, tokens):
        # This is fake code that does not detokenize. See above.
        # return self.model.detokenize(tokens)
        return ["".join(tokens)]

    def num_tokens(self, prompt):
        # Also fake. This generates a totally fake approximate number.
        # tokens = self.model.tokenize(prompt)
        # return (len(tokens))
        return len(prompt) // 4

class OpenAIServer(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: str,
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
        self.model_name = model_names

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        openai.api_key = self.api_key
        
        self.context_len = 4096
        print("Setting context length to", self.context_len)

        #edit this at the end
        # HACK: We don't really have access to the tokenization in openai
        # But we need a tokenizer to work with fastchat
        self.tokenizer = OpenAITokenizer(model=self.model_names)

        self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        params.pop("request_id")  # not used

        # Generation parameters
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        # frequency_penalty = float(params.get("frequency_penalty", 0.0))

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

        decoded_tokens = []

        # context_tokens = self.model.tokenize(context.encode("utf-8"))

        finish_reason = "length"

        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=[{"role": "user", "content": context}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop,
            stream=True,
        )

        async for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    decoded_tokens.append(token)
                    tokens_decoded_str = "".join(decoded_tokens)
                    
                    ret = {
                        "text": tokens_decoded_str,
                        "error_code": 0,
                        "usage": {
                            "prompt_tokens": len(context),  # crude estimate
                            "completion_tokens": len(decoded_tokens),
                            "total_tokens": len(context) + len(decoded_tokens),
                        },
                        "cumulative_logprob": [],
                        "finish_reason": None,
                    }
                    yield (json.dumps(ret) + "\0").encode()

        
        final_text = "".join(decoded_tokens)
        final_ret = {
            "text": final_text,
            "error_code": 0,
            "usage": {
                "prompt_tokens": len(context),
                "completion_tokens": len(decoded_tokens),
                "total_tokens": len(context) + len(decoded_tokens),
            },
            "cumulative_logprob": [],
            "finish_reason": finish_reason or "stop",
        }
        yield (json.dumps(final_ret) + "\0").encode()

    async def generate(self, params):
        prompt = params.pop("prompt")

        # TODO: Figure out what to do with max_tokens
        # max_tokens = params.get("max_new_tokens", 256)

        # Setup parameters
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)


        params = {"top_p": top_p, "temperature": temperature, "frequency_penalty": frequency_penalty}

        messages = [{"role": "user", "content": prompt}]

        print("Generating with params: ", params)

        response = await asyncio.to_thread(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_str,
                stream=False
            )
        )

        content = response.choices[0].message["content"]
        finish_reason = response.choices[0].finish_reason

        ret = {
            "text": content,
            "error_code": 0,
            "usage": {},
            "cumulative_logprob": [],
            "finish_reason": finish_reason,
        }
        return ret

    def stop_server(self):
        pass # Nothing to unload for OpenAI


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
    if worker:
        worker.stop_server()
        del worker


def main():
    global app, worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="openai_model")
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--parameters", type=str, default=None)

    args, _ = parser.parse_known_args()

    parameters = json.loads(args.parameters)
    print(args.parameters)
    print(parameters)
    model_name = str(parameters.get("model_name", "gpt-3.5-turbo"))
    

    worker = OpenAIServer(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        model_name,
        1024,
        args.conv_template
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()



