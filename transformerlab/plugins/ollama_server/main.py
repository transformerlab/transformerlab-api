"""
Ollama model worker

Requires that ollama is installed on your server.
"""

import argparse
import json


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
    parser.add_argument("--parameters", type=str, default=None)

    args, unknown = parser.parse_known_args()

    # parameters is a JSON string, so we parse it:
    parameters = json.loads(args.parameters)

    # model_path can be a hugging face ID or a local file in Transformer Lab
    # But GGUF is always stored as a local path because
    # we are using a specific GGUF file
    # TODO: Make sure the path exists before continuing
    # if os.path.exists(args.model_path):
    model_path = args.model_path

    """
    worker = OllamaServer(
        args.controller_address,
        args.worker_address,
        worker_id,
        model_path,
        args.model_names,
        1024,
        False,
        "ollama-python",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    """


if __name__ == "__main__":
    main()
