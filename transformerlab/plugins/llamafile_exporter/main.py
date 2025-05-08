import os
import subprocess
import shutil

from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter


tlab_exporter.add_argument("--model_path", default="gpt-j-6b", type=str, help="Path to directory or file containing the model.")


@tlab_exporter.exporter_job_wrapper(progress_start=0, progress_end=100)
def llamafile_export():
    """Export a model to Llamafile format"""

    # Make sure we remove the author part (everything before and including "/")
    input_model = tlab_exporter.params.get("model_name")
    input_model_id_without_author = input_model.split("/")[-1]

    # But we need the actual model path to get the GGUF file
    input_model_path = tlab_exporter.params.get("model_path")

    # Directory to run conversion subprocess
    plugin_dir = os.path.realpath(os.path.dirname(__file__))

    # Output details - ignoring the output_model_id passed by app
    outfile = f"{input_model_id_without_author}.llamafile"
    output_dir = tlab_exporter.params.get("output_dir")

    tlab_exporter.progress_update(10)
    tlab_exporter.add_job_data("status", "Starting Llamafile conversion")

    # Setup arguments for executing this model
    argsfile = os.path.join(plugin_dir, ".args")
    argsoutput = f"""-m
    {input_model_id_without_author}
    --host
    0.0.0.0
    -ngl
    9999
    ...
    """

    # Create a .args file to include in the llamafile
    with open(argsfile, "w") as f:
        f.write(argsoutput)

    tlab_exporter.progress_update(30)
    tlab_exporter.add_job_data("status", "Creating base llamafile")

    # Create a copy of pre-built llamafile to use as a base
    shutil.copy(os.path.join(plugin_dir, "llamafile"), os.path.join(plugin_dir, outfile))

    tlab_exporter.progress_update(50)
    tlab_exporter.add_job_data("status", "Merging model with Llamafile")

    # Merge files together in single executable using zipalign
    subprocess_cmd = ["sh", "./zipalign", "-j0", outfile, input_model_path, ".args"]
    export_process = subprocess.run(
        subprocess_cmd, cwd=plugin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Add output to job data
    stdout = export_process.stdout
    stderr = export_process.stderr if hasattr(export_process, "stderr") else ""
    tlab_exporter.add_job_data("stdout", stdout)
    tlab_exporter.add_job_data("stderr", stderr)

    tlab_exporter.progress_update(80)
    tlab_exporter.add_job_data("status", "Moving Llamafile to output directory")

    # Move file to output_dir
    shutil.move(os.path.join(plugin_dir, outfile), os.path.join(output_dir, outfile))

    # Final progress update
    tlab_exporter.progress_update(100)
    tlab_exporter.add_job_data("status", "Llamafile creation complete")

    return "Successful export to Llamafile format"


llamafile_export()
