import os
import sys
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from transformerlab.sdk.v1.generate import tlab_gen
from huggingface_hub import snapshot_download


@tlab_gen.job_wrapper()
def run_generation():
    # ----- Constants -----
    workspace = os.environ["_TFL_WORKSPACE_DIR"]
    REPO_ROOT = f"{workspace}/plugins/wd14_captioner/sd-caption-wd14/sd-scripts"
    SCRIPT_PATH = f"{workspace}/plugins/wd14_captioner/sd-caption-wd14/sd-scripts/finetune/tag_images_by_wd14_tagger.py"
    TMP_DATASET_DIR = Path(f"{workspace}/plugins/wd14_captioner/tmp_dataset")

    repo_id = tlab_gen.params.repo_id
    batch_size = str(tlab_gen.params.batch_size)
    caption_separator = tlab_gen.params.caption_separator
    remove_underscore = tlab_gen.params.remove_underscore
    include_ranks = tlab_gen.params.include_ranks
    model_dir = (
        Path(workspace)
        / "plugins"
        / "wd14_captioner"
        / "sd-caption-wd14"
        / "sd-scripts"
        / "wd14_tagger_model"
        / repo_id.replace("/", "_")
    )

    if not model_dir.exists():
        snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False, repo_type="model")

    tlab_gen.progress_update(0)
    datasets = tlab_gen.load_dataset(dataset_types=["train"])
    df = datasets["train"].to_pandas()

    if tlab_gen.params.image_field not in df.columns:
        raise ValueError(
            f"❌ Field '{tlab_gen.params.image_field}' not found in dataset. Available fields: {list(df.columns)}"
        )

    image_paths = df[tlab_gen.params.image_field].tolist()

    if not image_paths:
        raise RuntimeError("❌ No images found in dataset.")

    if TMP_DATASET_DIR.exists():
        shutil.rmtree(TMP_DATASET_DIR)
    TMP_DATASET_DIR.mkdir(parents=True)

    local_paths = []
    for idx, src_path in enumerate(tqdm(image_paths)):
        real_path = src_path["path"] if isinstance(src_path, dict) else src_path
        ext = Path(real_path).suffix
        dst = TMP_DATASET_DIR / f"{idx:06d}{ext}"
        shutil.copy(real_path, dst)
        local_paths.append(dst)
        tlab_gen.progress_update(5 + (idx + 1) / len(image_paths) * 15)

    # Build subprocess command
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--onnx",
        "--repo_id",
        repo_id,
        "--batch_size",
        str(batch_size),
        "--caption_separator",
        caption_separator,
        str(TMP_DATASET_DIR),
    ]

    if remove_underscore:
        command.append("--remove_underscore")
    if include_ranks:
        command.append("--include_ranks")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )

    tlab_gen.progress_update(25)

    print("🔍 Subprocess STDOUT:")
    print(result.stdout)
    print("🔍 Subprocess STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"❌ WD14 tagger subprocess failed with exit code {result.returncode}")

    captions = []
    for i, path in enumerate(local_paths):
        caption_path = path.with_suffix(".txt")
        if not caption_path.exists():
            raise FileNotFoundError(f"❌ Missing caption file for {path.name}: expected at {caption_path}")
        caption_text = caption_path.read_text().strip()
        captions.append(caption_text)
        tlab_gen.progress_update(25 + (i + 1) / len(local_paths) * 25)

    df_output = pd.DataFrame(
        {
            "file_name": [Path(p).name if isinstance(p, str) else Path(p["path"]).name for p in image_paths],
            "caption": captions,
        }
    )

    tlab_gen.progress_update(60)

    output_path, dataset_name = tlab_gen.save_generated_dataset(df_output, is_image=True)
    dataset_id = f"{tlab_gen.params.run_name}_{tlab_gen.params.job_id}".lower()
    output_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "datasets", dataset_id)
    for src_path in image_paths:
        real_path = src_path["path"] if isinstance(src_path, dict) else src_path
        dst_path = os.path.join(output_dir, Path(real_path).name)
        shutil.copy2(real_path, dst_path)

    tlab_gen.progress_update(95)

    print(f"✅ Saved captioned dataset as '{dataset_name}' at: {output_path}")

    shutil.rmtree(TMP_DATASET_DIR)
    tlab_gen.progress_update(100)


print("Starting image dataset captioning...")
run_generation()
