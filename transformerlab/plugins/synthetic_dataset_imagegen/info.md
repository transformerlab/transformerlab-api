# Synthetic Dataset: Image Generation from Prompts

This plugin generates a synthetic image dataset from a set of text prompts using a text-to-image generation model (e.g., Stable Diffusion).

## Input

The plugin accepts a dataset of prompts in any of the following formats:
- `.txt` — one prompt per line
- `.json` or `.jsonl` — each entry must contain a `prompt`, `text`, or `doc` field
- `.pdf` — extracted line-by-line from the document text

## Parameters

| Name | Description |
|------|-------------|
| `generation_model` | The image generation model to use. You may select a local model or one provided by TransformerLab. |
| `tflabcustomui_docs` | The dataset file containing your prompts (e.g., `.txt`, `.json`, `.pdf`, etc.). |
| `image_width` | Width of generated images in pixels (default: 512). |
| `image_height` | Height of generated images in pixels (default: 512). |
| `images_per_prompt` | Number of images to generate for each prompt (default: 4, max: 8). |

## Output

The output is a structured dataset containing:
- PNG images saved under `output/images/`
- A metadata file `metadata.jsonl` containing the prompt and corresponding image path for each generated image

Example entry in `metadata.jsonl`:
```json
{
  "prompt": "a surreal castle floating in space",
  "image_path": "images/prompt_3/image_1.png"
}
```

## Notes

- You can customize the resolution and number of images per prompt.
- The plugin uses `tlab_gen.params.model` to route the request to the appropriate image generation backend, which internally conforms to the DiffusionRequest format used by `/generate`.
