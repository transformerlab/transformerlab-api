import os
import asyncio
import shutil

try:
    from transformerlab.sdk.v1.export import tlab_exporter
except ImportError or ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter


tlab_exporter.add_argument(
    "--model_path", default="gpt-j-6b", type=str, help="Path to directory or file containing the model."
)


@tlab_exporter.async_job_wrapper(progress_start=0, progress_end=100)
async def llamafile_export():
    """Export a model to Llamafile format"""
    input_model = tlab_exporter.params.get("model_name")
    input_model_id_without_author = input_model.split("/")[-1]
    input_model_path = tlab_exporter.params.get("model_path")
    output_dir = tlab_exporter.params.get("output_dir")  # Must be created by plugin
    plugin_dir = os.path.realpath(os.path.dirname(__file__))
    outfile_name = f"{input_model_id_without_author}.llamafile"
    argsfile = os.path.join(plugin_dir, ".args")

    # Create output file
    os.makedirs(output_dir, exist_ok=True)

    print("Starting Llamafile conversion...")
    tlab_exporter.progress_update(15)
    tlab_exporter.add_job_data("status", "Starting Llamafile conversion")
    print("Creating .args file...")

    argsoutput = f"""-m
                {input_model_id_without_author}
                --host
                0.0.0.0
                -ngl
                9999
                """

    with open(argsfile, "w") as f:
        f.write(argsoutput)

    tlab_exporter.progress_update(30)
    tlab_exporter.add_job_data("status", "Creating base llamafile")
    print("Copying base llamafile...")

    base_llamafile = os.path.join(plugin_dir, "llamafile")
    temp_llamafile = os.path.join(plugin_dir, outfile_name)
    shutil.copy(base_llamafile, temp_llamafile)

    tlab_exporter.progress_update(50)
    tlab_exporter.add_job_data("status", "Merging model with Llamafile")

    # Merge files together in single executable using zipalign
    subprocess_cmd = ["sh", "./zipalign", "-j0", outfile_name, input_model_path, ".args"]
    export_process = await asyncio.create_subprocess_exec(
        *subprocess_cmd,
        cwd=plugin_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert export_process.stdout is not None
    stdout_lines = []
    async for line_bytes in export_process.stdout:
        line = line_bytes.decode("utf-8", errors="replace").strip()
        stdout_lines.append(line)
        print(line)

    return_code = await export_process.wait()
    stdout = "\n".join(stdout_lines)

    if return_code != 0:
        tlab_exporter.add_job_data("stderr", stdout)
        print(f"zipalign failed with code {return_code}")
        raise RuntimeError(f"zipalign failed with return code {return_code}")

    tlab_exporter.progress_update(80)
    tlab_exporter.add_job_data("status", "Moving Llamafile to output directory")
    print(f"Moving {outfile_name} to output directory {output_dir}...")

    final_llamafile_path = os.path.join(output_dir, outfile_name)
    shutil.move(temp_llamafile, final_llamafile_path)

    tlab_exporter.progress_update(100)
    tlab_exporter.add_job_data("status", "Llamafile creation complete")
    print("Llamafile export completed successfully!")

    return "Successful export to Llamafile format"


llamafile_export()
