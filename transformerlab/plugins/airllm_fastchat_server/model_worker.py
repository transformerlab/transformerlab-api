"""
This is a copy of https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/model_worker.py
Copied on Sept 6, 2024. We use the standard model worker from Fastchat but then
Add a few of our own endpoints
A model worker that executes the model.
"""

import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid

from fastapi import Request
import torch
import torch.nn.functional as F
from transformers import set_seed
import uvicorn
from airllm import AutoModel


from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    add_model_args,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    build_logger,
    str_to_torch_dtype,
)

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        revision: str = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")

        # Use AirLLM to load model instead of FastChat's load_model
        self.model = AutoModel.from_pretrained(model_path, revision=revision)
        self.tokenizer = self.model.tokenizer

        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get context length from the model's config
        self.context_len = getattr(self.model, "context_len", 4096)  # Default to 4096 if not available

        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        if self.device == "npu":
            import torch_npu

            torch_npu.npu.set_device("npu:0")
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)

            # Prepare input for AirLLM
            prompt = params.get("prompt", "")
            max_new_tokens = params.get("max_new_tokens", 256)
            temperature = params.get("temperature", 1.0)
            top_p = params.get("top_p", 1.0)

            # Tokenize input
            input_tokens = self.tokenizer(
                prompt, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=self.context_len
            )

            # Move input to device
            input_ids = input_tokens["input_ids"].to(self.device)

            # Generate tokens with AirLLM in chunks for streaming
            token_stream = []
            current_length = input_ids.shape[1]
            chunk_length = max(1, self.stream_interval)

            for chunk_tokens in range(0, max_new_tokens, chunk_length):
                chunk_size = min(chunk_length, max_new_tokens - chunk_tokens)

                generation_output = self.model.generate(
                    input_ids,
                    max_new_tokens=chunk_size,
                    use_cache=True,
                    temperature=temperature,
                    top_p=top_p,
                    return_dict_in_generate=True,
                )

                # Get the generated sequence
                sequences = generation_output.sequences
                generated_tokens = sequences[0][current_length:]
                token_stream.extend(generated_tokens.tolist())

                # Update current context for next generation
                input_ids = sequences
                current_length = input_ids.shape[1]

                # Decode the newly generated tokens
                output_text = self.tokenizer.decode(token_stream, skip_special_tokens=True)

                ret = {
                    "text": output_text,
                    "error_code": 0,
                    "usage": {
                        "prompt_tokens": len(input_tokens["input_ids"][0]),
                        "completion_tokens": len(token_stream),
                        "total_tokens": len(input_tokens["input_ids"][0]) + len(token_stream),
                    },
                    "finish_reason": None if chunk_tokens + chunk_size < max_new_tokens else "length",
                }

                yield json.dumps(ret).encode() + b"\0"

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]

        if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
            sum_embeddings = data[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    def __encode_base64(self, embeddings: torch.Tensor) -> List[str]:
        embeddings = embeddings.cpu()
        return [base64.b64encode(e.numpy().tobytes()).decode("utf-8") for e in embeddings]

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            # For embeddings with AirLLM, we'll need to handle differently
            # This is a simplified implementation - might need adjustments based on AirLLM API
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            # Tokenize the input text
            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(params["input"], padding=True, return_tensors="pt")

            input_ids = encoding["input_ids"].to(self.device)

            # Get embeddings from the model
            # Note: This part depends on whether AirLLM has direct support for embeddings
            # For now, using the last hidden state as embeddings
            with torch.no_grad():
                # If the model has a direct method for embeddings, use it
                if hasattr(self.model, "get_embeddings"):
                    embeddings = self.model.get_embeddings(input_ids)
                else:
                    # Otherwise, we'll need to implement a fallback approach
                    # This is a placeholder and would need to be adjusted
                    outputs = self.model.model(input_ids)
                    embeddings = outputs.last_hidden_state.mean(dim=1)

            # Normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

            # Format based on encoding format requested
            base64_encode = params.get("encoding_format", None)
            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()

            ret["embedding"] = out_embeddings
            ret["token_num"] = input_ids.shape[1]

            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }

        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument("--debug", type=bool, default=False, help="Print debugging messages")
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        revision=args.revision,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
    )
    return args, worker


@app.post("/tokenize")
async def api_tokenize(request: Request):
    params = await request.json()
    text = params["text"]
    token_ids = worker.tokenizer(text).input_ids
    tokens = worker.tokenizer.convert_ids_to_tokens(token_ids)
    return {"tokens": tokens, "token_ids": token_ids}


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
