"""
VLLM model worker

Provides a FastChat-compatible worker interface that communicates with a VLLM OpenAI-compatible server.
"""

import argparse
import asyncio
import os
import subprocess
import json
import uuid
import sys
import atexit
import traceback
import uvicorn
import time
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import is_partial_stop

import transformerlab.plugin

app = FastAPI()
worker = None


class VLLMClient:
    """Client for communicating with the VLLM OpenAI-compatible server"""
    
    def __init__(self, base_url="http://localhost:21009"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def chat(self, model: str, messages: List[Dict[str, str]], stream=False, **options):
        """Call the chat completions endpoint"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **options
        }
        
        if not stream:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        else:
            return self.session.post(url, json=payload, stream=True)
            
    def generate(self, model: str, prompt: str, stream=False, **options):
        """Call the completions endpoint"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **options
        }
        
        if not stream:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        else:
            return self.session.post(url, json=payload, stream=True)


class VLLMTokenizer:
    """
    Simplified tokenizer interface for FastChat compatibility
    """
    def __init__(self, client):
        self.client = client
        self.eos_token_id = None
        
    def encode(self, text):
        """Approximate token count using character-based heuristic"""
        # This is a rough approximation - in a real implementation we'd call the tokenize endpoint
        return [0] * (len(text) // 4)
        
    def decode(self, tokens):
        """Fake decode implementation"""
        return [""]
        
    def __call__(self, text):
        """Make the tokenizer callable"""
        # Very minimal implementation to satisfy FastChat requirements
        from transformers.tokenization_utils_base import BatchEncoding
        tokens = self.encode(text)
        return BatchEncoding(data={"input_ids": [tokens], "eos_token_id": None})


class VLLMServer(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str,
        parameters: Dict
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

        logger.info(f"Loading model {self.model_names} on worker {worker_id}, worker type: vllm...")
        
        # Clean parameters
        self.parameters = parameters or {}
        self.clean_parameters()
        
        # Start VLLM OpenAI-compatible server
        self.vllm_port = 21009
        self.vllm_server_process = self.start_vllm_openai_server(model_path)
        
        # Create client to talk to the server
        vllm_api_url = f"http://localhost:{self.vllm_port}"
        self.model = VLLMClient(base_url=vllm_api_url)
        
        # For FastChat compatibility
        self.tokenizer = VLLMTokenizer(self.model)
        
        # Store OpenAI URL in database config
        transformerlab.plugin.set_db_config_value("OPENAI_API_URL", vllm_api_url)
        
        # Default context length - we'll try to get it from the model info if possible
        self.context_len = 4096
        
        # Wait for the server to be ready
        self.wait_for_server_ready()
        
        # Try to get model details from the server
        try:
            self.get_model_details()
        except Exception as e:
            logger.error(f"Error getting model details: {e}")
        
        self.init_heart_beat()
    
    def start_vllm_openai_server(self, model_path):
        """Start VLLM OpenAI-compatible server as a subprocess"""
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(["--model", model_path])
        cmd.extend(["--port", str(self.vllm_port)])
        
        # Add VLLM-specific parameters
        for key, value in self.parameters.items():
            # Convert to snake_case if needed
            key_snake_case = key.replace("-", "_")
            cmd.extend([f"--{key_snake_case}", str(value)])
        
        print(f"Starting VLLM OpenAI server with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            text=True,
            bufsize=1  # Line buffered
        )
        
        return process
    
    def wait_for_server_ready(self, max_retries=60, delay=1.0):
        """Wait for the VLLM server to be ready"""
        print("Waiting for VLLM server to start...")
        url = f"http://localhost:{self.vllm_port}/v1/models"
        
        for i in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("VLLM server is ready.")
                    return
            except requests.RequestException:
                pass
                
            time.sleep(delay)
            print(f"Waiting for VLLM server... ({i+1}/{max_retries})")
            
        print("VLLM server didn't start in the expected time.")
    
    def get_model_details(self):
        """Get model details from the VLLM server"""
        try:
            url = f"http://localhost:{self.vllm_port}/v1/models"
            response = requests.get(url)
            response.raise_for_status()
            models_data = response.json()
            
            if "data" in models_data and models_data["data"]:
                model = models_data["data"][0]
                if "max_context_window_size" in model:
                    self.context_len = model["max_context_window_size"]
                    print(f"Context length: {self.context_len}")
        except Exception as e:
            print(f"Failed to get model details: {e}")
    
    def clean_parameters(self):
        """Clean up parameters for VLLM compatibility."""
        if "inferenceEngine" in self.parameters:
            del self.parameters["inferenceEngine"]
            
        if "inferenceEngineFriendlyName" in self.parameters:
            del self.parameters["inferenceEngineFriendlyName"]
            
        # Convert kebab-case to snake_case
        keys_to_convert = []
        for key in self.parameters:
            if "-" in key:
                new_key = key.replace("-", "_")
                keys_to_convert.append((key, new_key))
                
        for old_key, new_key in keys_to_convert:
            self.parameters[new_key] = self.parameters.pop(old_key)
            
        # Handle empty values
        keys_to_remove = []
        for key, value in self.parameters.items():
            if value == "":
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.parameters[key]
        
    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        params.pop("request_id")  # not used

        # Generation parameters
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        
        # Create options for the VLLM server
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_new_tokens
        }
        
        # Add stop strings
        if stop_str:
            options["stop"] = stop_str
            
        print(f"Generating with options: {options}")
        
        # Format as a chat message
        messages = [{"role": "user", "content": context}]
        
        # Get streaming response from VLLM server
        try:
            response_stream = await run_in_threadpool(
                self.model.chat,
                model="default",
                messages=messages,
                stream=True,
                **options
            )
            
            # Stream each chunk
            accumulated_text = ""
            finish_reason = None
            
            for line in response_stream.iter_lines():
                if not line:
                    continue
                    
                if line.startswith(b"data: "):
                    data_str = line[len(b"data: "):].decode("utf-8")
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                        
                        # Extract the content from the chunk
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            
                            if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                chunk_text = choice["delta"]["content"]
                                accumulated_text += chunk_text
                                
                                # For usage info
                                prompt_tokens = len(context) // 4  # rough approximation
                                completion_tokens = len(accumulated_text) // 4  # rough approximation
                                
                                ret = {
                                    "text": accumulated_text,
                                    "error_code": 0,
                                    "usage": {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": prompt_tokens + completion_tokens,
                                    },
                                    "cumulative_logprob": [],
                                    "finish_reason": None,  # still generating
                                }
                                yield (json.dumps(ret) + "\0").encode()
                            
                            # Check for finish reason in the final message
                            if "finish_reason" in choice and choice["finish_reason"]:
                                finish_reason = choice["finish_reason"]
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        continue
            
            # Final message with finish reason
            prompt_tokens = len(context) // 4
            completion_tokens = len(accumulated_text) // 4
            
            ret = {
                "text": accumulated_text,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [],
                "finish_reason": finish_reason or "stop",
            }
            yield (json.dumps(obj={**ret, **{"finish_reason": None}}) + "\0").encode()
            yield (json.dumps(ret) + "\0").encode()
            
        except Exception as e:
            print(f"Error in generate_stream: {e}")
            print(traceback.format_exc())
            ret = {
                "text": f"Error: {str(e)}",
                "error_code": 1,
                "usage": {},
                "cumulative_logprob": [],
                "finish_reason": "error",
            }
            yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        prompt = params.pop("prompt")
        
        # Setup parameters
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_tokens = int(params.get("max_new_tokens", 256))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        stop = params.get("stop", None)
        
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty
        }
        
        if stop:
            options["stop"] = stop
            
        print(f"Generating with options: {options}")
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await run_in_threadpool(
                self.model.chat,
                model="default",
                messages=messages,
                stream=False,
                **options
            )
            
            # Extract the generated text
            if "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "stop")
                
                # Get usage info
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", len(prompt) // 4)
                completion_tokens = usage.get("completion_tokens", len(text) // 4)
                
                ret = {
                    "text": text,
                    "error_code": 0,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "finish_reason": finish_reason,
                }
                return ret
            else:
                return {
                    "text": "",
                    "error_code": 1,
                    "usage": {},
                    "finish_reason": "error",
                }
                
        except Exception as e:
            print(f"Error in generate: {e}")
            print(traceback.format_exc())
            return {
                "text": f"Error: {str(e)}",
                "error_code": 1,
                "usage": {},
                "finish_reason": "error",
            }

    def count_token(self, params):
        """Approximate token count for the prompt"""
        prompt = params["prompt"]
        # This is just an approximation
        return len(prompt) // 4
            
    def stop_server(self):
        """
        Called by cleanup_at_exit.
        """
        print("Stopping VLLM OpenAI server...")
        if hasattr(self, 'vllm_server_process') and self.vllm_server_process:
            self.vllm_server_process.terminate()
            try:
                self.vllm_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.vllm_server_process.kill()


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


atexit.register(cleanup_at_exit)


def main():
    global app, worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--parameters", type=str, default="{}")
    args, _ = parser.parse_known_args()

    # Parse parameters
    parameters = json.loads(args.parameters)

    # Save URL to database config for other services to use
    transformerlab.plugin.set_db_config_value("INFERENCE_SERVER_URL", "21009")
    
    # Get LLM Lab root directory
    llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")

    print(f"Starting VLLM Server worker with model: {args.model_path}", file=sys.stderr)
    
    try:
        worker = VLLMServer(
            args.controller_address,
            args.worker_address,
            worker_id,
            args.model_path,
            args.model_names,
            1024,
            args.conv_template,
            parameters,
        )
        
        # Save worker PID for process management
        with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
            f.write(str(os.getpid()))
            
        # Start the FastAPI server
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        
    except Exception as e:
        print(f"Error starting VLLM server: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()