# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import asyncio

try:
    from transformerlab.sdk.v1.export import tlab_exporter
    from transformerlab.plugin import get_python_executable
except ImportError or ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter


tlab_exporter.add_argument("--q_bits", default="4", type=str, help="Bits per weight for quantization.")


@tlab_exporter.async_job_wrapper(progress_start=0, progress_end=100)
async def mlx_export():
    plugin_dir = os.path.realpath(os.path.dirname(__file__))
    python_executable = get_python_executable(plugin_dir)

    command = [
        python_executable,
        "-u",
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        tlab_exporter.params.get("model_name"),
        "--mlx-path",
        tlab_exporter.params.get("output_dir"),
        "-q",
        "--q-bits",
        str(tlab_exporter.params.get("q_bits")),
    ]

    print("Starting MLX conversion...")
    print(f"Running command: {' '.join(command)}")
    tlab_exporter.add_job_data("command", " ".join(command))

    tlab_exporter.progress_update(5)
    tlab_exporter.add_job_data("status", "Starting MLX conversion")

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=plugin_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        all_output_lines = []
        progress_value = 5

        assert process.stdout is not None
        async for line_bytes in process.stdout:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            all_output_lines.append(line)
            print(line, flush=True)

            if "Loading" in line:
                progress_value = 15
                tlab_exporter.add_job_data("status", "Loading model")
            elif "Fetching" in line:
                progress_value = 35
                tlab_exporter.add_job_data("status", "Fetching model files")
            elif "Using dtype" in line:
                progress_value = 50
                tlab_exporter.add_job_data("status", "Preparing quantization")
            elif "Quantizing" in line:
                progress_value = 65
                tlab_exporter.add_job_data("status", "Quantizing model")
            elif "Quantized model" in line:
                progress_value = 80
                tlab_exporter.add_job_data("status", "Finalizing model")

            tlab_exporter.progress_update(progress_value)

        return_code = await process.wait()
        tlab_exporter.add_job_data("stdout", "\n".join(all_output_lines))

        if return_code != 0:
            error_msg = f"MLX conversion failed with return code {return_code}"
            print(error_msg)
            tlab_exporter.add_job_data("status", error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"MLX conversion failed with exception: {str(e)}"
        print(error_msg)
        tlab_exporter.add_job_data("status", error_msg)
        raise

    print("MLX conversion completed successfully!")
    tlab_exporter.add_job_data("status", "MLX conversion complete")
    tlab_exporter.progress_update(100)

    return "Successful export to MLX format"


mlx_export()
