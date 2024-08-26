import argparse
import os
import uuid

from fastchat.model.model_adapter import add_model_args
from fastchat.serve.base_model_worker import app, logger
from fastchat.serve.model_worker import ModelWorker

from typing import List, Optional
import torch

import uvicorn
import atexit

worker_id = str(uuid.uuid4())[:8] # TODO?


class ToolsModelWorker(ModelWorker):
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
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        conv_template: Optional[str] = None,
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
            no_register,
            device,
            num_gpus,
            max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            conv_template=conv_template,
            seed=seed,
            debug=debug,
            **kwargs,
        )

worker = None  ## NEW

def cleanup_at_exit():
    global worker
    print("Cleaning up...")
    del worker


atexit.register(cleanup_at_exit)


def main():
    """
    Inspired by create_model_worker() in model_server from FastChat.

    Need to copy in order to call ToolsModelWorker.
    """
    global app, worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    # reads in device, num_gpus, model_path, load-8bit and more arguments
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    args = parser.parse_args()
    #logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    

    worker = ToolsModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        1024,
        False,
        args.device,
        args.num_gpus,
        "",
        load_8bit = args.load_8bit,
        conv_template = args.conv_template,
    )
    print("Model Worker called")
    print("Template:")
    print(worker.get_conv_template())
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    print("I shouldn't get here")


if __name__ == "__main__":
    main()