import sys
import subprocess
import contextlib
import io
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from transformerlab.sdk.v1.export import tlab_exporter
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter

tlab_exporter.add_argument("--output_model_id", type=str, help="Target directory name for the ONNX model")
tlab_exporter.add_argument("--opset", type=int, default=17, help="ONNX opset version (defaults to 17)")
tlab_exporter.add_argument("--quantize", action="store_true", help="Apply dynamic INT8 quantization after export")


@tlab_exporter.exporter_job_wrapper(progress_start=0, progress_end=100)
def onnx_export():
    input_model = tlab_exporter.params.get("model_name")
    opset = int(tlab_exporter.params.get("opset", 17))
    quantize = tlab_exporter.params.get("quantize", False)
    output_dir = Path(tlab_exporter.params.get("output_dir"))

    output_dir.mkdir(parents=True, exist_ok=True)

    tlab_exporter.add_job_data("status", "Preparing model source")

    model_path: str | Path = input_model
    if not Path(model_path).exists():
        tlab_exporter.add_job_data("status", "Downloading model from Hugging Face Hub")
        tlab_exporter.progress_update(5)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model_path = snapshot_download(repo_id=input_model)

    tlab_exporter.progress_update(7)

    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        str(model_path),
        "--opset",
        str(opset),
        "--framework",
        "pt",
        "--output",
        str(output_dir),
    ]

    if quantize:
        cmd.extend(["--quantize", "dynamic"])

    tlab_exporter.add_job_data("command", " ".join(cmd))
    tlab_exporter.add_job_data("status", "Running ONNX export")
    tlab_exporter.progress_update(10)

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        ) as proc:
            for line in proc.stdout:
                print(line, flush=True)
                if "%" in line:
                    import re

                    m = re.search(r"(\d+)%", line)
                    if m:
                        pct = int(m.group(1))
                        tlab_exporter.progress_update(10 + int(pct * 0.8))
                        tlab_exporter.add_job_data("status", f"ONNX export {pct}%")
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"ONNX export failed with code {return_code}")
    except Exception as exc:
        err = f"ONNX export failed: {exc}"
        tlab_exporter.add_job_data("status", err)
        raise

    tlab_exporter.progress_update(100)
    tlab_exporter.add_job_data("status", "ONNX export complete")
    return "Successful export to ONNX format"


if __name__ == "__main__":
    onnx_export()
