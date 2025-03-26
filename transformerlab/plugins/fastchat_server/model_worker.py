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
import uuid
from threading import Thread
from typing import List, Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import Request
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastchat.model.model_adapter import add_model_args, get_generate_stream_function, load_model
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import build_logger, get_context_length, str_to_torch_dtype
from transformers import AutoTokenizer, Gemma3ForCausalLM, TextIteratorStreamer, set_seed

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


@torch.inference_mode()
def gemma3_generate_stream(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
):
    """Custom generate stream function for Gemma-3 models"""
    # Get parameters from the request
    prompt = params.get("prompt", "")
    messages = params.get("messages", None)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    logger.info("Using custom Gemma-3 stream generation")

    model_name = params.get("model", None)

    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    is_base_model = "pt" in model_name.lower() or "base" in model_name.lower()

    if not is_base_model:
        # Format input based on whether we have messages or a plain prompt
        if messages:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    input_echo_len = input_ids.shape[1]

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else 1.0,
    }

    if top_p < 1.0:
        generate_kwargs["top_p"] = top_p
    if top_k > 0:
        generate_kwargs["top_k"] = top_k
    if repetition_penalty > 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=not echo, skip_special_tokens=True)
    generate_kwargs["streamer"] = streamer

    # Start generation in a separate thread
    thread = Thread(target=lambda: model.generate(input_ids=input_ids, **generate_kwargs))
    thread.start()

    # Track generation progress
    generated_tokens = 0
    output_text = ""

    # Stream tokens
    for new_text in streamer:
        output_text += new_text
        generated_tokens += 1

        # Check for stop strings
        should_stop = False
        if stop_str:
            if isinstance(stop_str, str):
                if stop_str in output_text:
                    output_text = output_text[: output_text.find(stop_str)]
                    should_stop = True
            elif isinstance(stop_str, list):
                for stop in stop_str:
                    if stop in output_text:
                        output_text = output_text[: output_text.find(stop)]
                        should_stop = True
                        break

        # Stream at intervals or when stopping
        if generated_tokens % stream_interval == 0 or should_stop:
            yield {
                "text": output_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": generated_tokens,
                    "total_tokens": input_echo_len + generated_tokens,
                },
                "finish_reason": "stop" if should_stop else None,
            }

        if should_stop:
            break

    # Final output with finish reason
    if thread.is_alive():
        thread.join(
            timeout=3600
        )  # Arbitrary value, but if it doesn't complete in this much time then something is wrong

    yield {
        "text": output_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": generated_tokens,
            "total_tokens": input_echo_len + generated_tokens,
        },
        "finish_reason": "length",
    }

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()


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
        # Check if this is a Gemma-3 model
        self.is_gemma3 = "gemma-3" in model_path.lower()
        if self.is_gemma3:
            logger.info("Detected Gemma-3 model, using custom loading logic")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
            self.model = Gemma3ForCausalLM.from_pretrained(
                model_path,
                revision=revision,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device != "cpu" else None,
            )
            self.generate_stream_func = gemma3_generate_stream
        else:
            self.model, self.tokenizer = load_model(
                model_path,
                revision=revision,
                device=device,
                num_gpus=num_gpus,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
                load_8bit=load_8bit,
                cpu_offloading=cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                exllama_config=exllama_config,
                xft_config=xft_config,
                debug=debug,
            )
            self.generate_stream_func = get_generate_stream_function(self.model, model_path)

        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
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
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
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
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            model_type_dict = {
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }

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
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                embedding, token_num = self.__process_embed_chunk(input_ids, attention_mask, **model_type_dict)
                if not hasattr(self.model, "use_cls_pooling") or not self.model.use_cls_pooling:
                    embedding = embedding / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    chunk_input_ids = input_ids[:, i : i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i : i + self.context_len]

                    # add cls token and mask to get cls embedding
                    if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
                        cls_tokens = (
                            torch.zeros(
                                (chunk_input_ids.size(0), 1),
                                dtype=chunk_input_ids.dtype,
                                device=chunk_input_ids.device,
                            )
                            + tokenizer.cls_token_id
                        )
                        chunk_input_ids = torch.cat([cls_tokens, chunk_input_ids], dim=-1)
                        mask = torch.ones(
                            (chunk_attention_mask.size(0), 1),
                            dtype=chunk_attention_mask.dtype,
                            device=chunk_attention_mask.device,
                        )
                        chunk_attention_mask = torch.cat([mask, chunk_attention_mask], dim=-1)

                    chunk_embeddings, token_num = self.__process_embed_chunk(
                        chunk_input_ids, chunk_attention_mask, **model_type_dict
                    )
                    if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
                        all_embeddings.append(chunk_embeddings * token_num)
                    else:
                        all_embeddings.append(chunk_embeddings)
                    all_token_num += token_num

                all_embeddings_tensor = torch.stack(all_embeddings)
                embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

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
