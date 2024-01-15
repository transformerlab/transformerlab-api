import argparse
import os
import random
import subprocess
import sys
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Any, Dict, Generator, List, Optional, Union
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import generate_step

from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str)
args, unknown = parser.parse_known_args()

print("Starting MLX API Server", file=sys.stderr)

# model = args.model_path

llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')

app = FastAPI()

model_name = "microsoft/phi-2"
mlx_model = None
mlx_tokenizer = None


async def completion_stream_generator(prompt, temperature, n: int):
    global mlx_tokenizer
    global mlx_model

    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        generate_step(prompt, mlx_model,
                      temperature), range(args.max_tokens)
    ):
        if token == mlx_tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
        tokens.append(token.item())
        s = mlx_tokenizer.decode(tokens)
        yield (tokens)
        skip = len(s)
    print(mlx_tokenizer.decode(tokens)[skip:], flush=True)
    yield "data: [DONE]\n\n"


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []


@app.post("/v1/completions")
async def create_completion(model: str = "microsoft/phi-2",
                            prompt: Union[str, List[Any]] = "",
                            suffix: Optional[str] = None,
                            temperature: Optional[float] = 0.7,
                            n: Optional[int] = 1,
                            max_tokens: Optional[int] = 16,
                            stop: Optional[Union[str, List[str]]] = None,
                            stream: Optional[bool] = False,
                            top_p: Optional[float] = 1.0,
                            top_k: Optional[int] = -1,
                            logprobs: Optional[int] = None,
                            echo: Optional[bool] = False,
                            presence_penalty: Optional[float] = 0.0,
                            frequency_penalty: Optional[float] = 0.0,
                            user: Optional[str] = None,
                            use_beam_search: Optional[bool] = False,
                            best_of: Optional[int] = None):

    global mlx_tokenizer
    global mlx_model

    prompt = mx.array(mlx_tokenizer.encode(prompt))

    if stream:
        generator = completion_stream_generator(
            prompt, temperature, max_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        tokens = []
        skip = 0

        for token, n in zip(
            generate_step(prompt, mlx_model, temperature), range(max_tokens)
        ):
            if token == mlx_tokenizer.eos_token_id:
                break
            tokens.append(token.item())
            s = mlx_tokenizer.decode(tokens)
            t = mlx_tokenizer.decode([token.item()])
            print(t, end="", flush=True)
            skip = len(s)

        tokens = mlx_tokenizer.decode(tokens)

        choices = []
        choices.append({"text": tokens,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length"}
                       )
        response = {
            "choices": choices,
            "created": 0,
            "id": random.randint(0, 1000000000),
            "model": model,
            "object": "text_completion",
        }
        return response


@app.get('/v1/models')
async def get_models():
    global model_name
    return {'models': [model_name]}


async def get_conv(model_name: str):
    conv = None
    conv = get_conv_template("vicuna_v1.1")
    return conv


async def get_gen_params(
    model_name: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    conv = await get_conv(model_name)

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.set_system_message(message["content"])
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
    }

    if not stop:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    return gen_params


@app.post('/v1/chat/completions')
async def create_chat_completion(model: str = "microsoft/phi-2",
                                 messages: Union[str,
                                                 List[Dict[str, str]]] = [],
                                 temperature: Optional[float] = 0.7,
                                 top_p: Optional[float] = 1.0,
                                 top_k: Optional[int] = -1,
                                 n: Optional[int] = 1,
                                 max_tokens: Optional[int] = None,
                                 stop: Optional[Union[str, List[str]]] = None,
                                 stream: Optional[bool] = False,
                                 user: Optional[str] = None,
                                 repetition_penalty: Optional[float] = 1.0,
                                 frequency_penalty: Optional[float] = 0.0,
                                 presence_penalty: Optional[float] = 0.0):

    global mlx_tokenizer
    global mlx_model

    gen_params = await get_gen_params(
        model,
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        echo=False,
        stream=stream,
        stop=stop,
    )

    print(gen_params)

    prompt = gen_params["prompt"]
    prompt = mx.array(mlx_tokenizer.encode(prompt))

    stream = False

    if stream:
        generator = completion_stream_generator(
            prompt, temperature, max_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        tokens = []
        skip = 0

        for token, n in zip(
            generate_step(prompt, mlx_model, float(gen_params["temperature"])), range(
                gen_params["max_new_tokens"])
        ):
            if token == mlx_tokenizer.eos_token_id:
                break
            tokens.append(token.item())
            s = mlx_tokenizer.decode(tokens)
            t = mlx_tokenizer.decode([token.item()])
            print(t, end="", flush=True)
            skip = len(s)

        tokens = mlx_tokenizer.decode(tokens)

        choices = []
        choices.append({"text": tokens,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length"}
                       )
        response = {
            "choices": choices,
            "created": 0,
            "id": random.randint(0, 1000000000),
            "model": model,
            "object": "text_completion",
        }
        return response


if __name__ == "__main__":
    model_name = "microsoft/phi-2"
    mlx_model, mlx_tokenizer = load("microsoft/phi-2")
    uvicorn.run(app, port=8001)
