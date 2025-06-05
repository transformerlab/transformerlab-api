import os
from typing import List
import pandas as pd
from io import BytesIO
from PIL import Image
import base64

from tqdm import tqdm
import fitz
from transformerlab.sdk.v1.generate import tlab_gen


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF files using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def get_docs_list(docs: str) -> List[dict]:
    """
    Convert document paths to a list of document data suitable for LangChain
    Supports text files, PDFs, and other document formats
    """
    docs_list = docs.split(",")
    documents_dir = os.path.join(
        os.environ.get("_TFL_WORKSPACE_DIR"), "experiments", tlab_gen.params.experiment_name, "documents"
    )
    # Use the markdown files if they exist
    if os.path.exists(os.path.join(documents_dir, ".tlab_markitdown")):
        documents_dir = os.path.join(documents_dir, ".tlab_markitdown")

    result_docs = []

    for doc in docs_list:
        doc_path = os.path.join(documents_dir, doc)

        if os.path.isdir(doc_path):
            print(f"Directory found: {doc_path}. Fetching all files in the directory...")
            for file in os.listdir(doc_path):
                file_full_path = os.path.join(doc_path, file)
                if not os.path.isdir(file_full_path):
                    try:
                        # Process based on file extension
                        if file_full_path.lower().endswith(".pdf"):
                            content = extract_text_from_pdf(file_full_path)
                            result_docs.append({"text": content, "source": file})
                        else:
                            with open(file_full_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                result_docs.append({"text": content, "source": file})
                    except Exception as e:
                        print(f"Error reading file {file_full_path}: {e}")
        else:
            full_path = os.path.join(documents_dir, doc)
            # Replace ending extension with .md if .tlab_markitdown is in the full_path somewhere
            if ".tlab_markitdown" in full_path:
                base, ext = os.path.splitext(full_path)
                full_path = base + ".md"
            try:
                if full_path.lower().endswith(".pdf"):
                    content = extract_text_from_pdf(full_path)
                    result_docs.append({"text": content, "source": doc})
                else:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        result_docs.append({"text": content, "source": doc})
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")

    return result_docs


@tlab_gen.job_wrapper(progress_start=0, progress_end=100)
def run_generation():
    """Generate synthetic images from prompts."""

    if not tlab_gen.params.docs:
        raise ValueError("Prompt documents must be provided for image generation.")

    # Validate and parse generation parameters
    image_width = int(tlab_gen.params.image_width)
    image_height = int(tlab_gen.params.image_height)
    images_per_prompt = int(tlab_gen.params.images_per_prompt)

    model = tlab_gen.params.model
    print("Image generation model loaded successfully")
    tlab_gen.progress_update(10)

    # Load and process prompt documents
    doc_data = get_docs_list(tlab_gen.params.docs)
    if len(doc_data) == 0:
        raise ValueError("No valid documents or prompts found.")

    print(f"Generating images for {len(doc_data)} prompt documents")

    outputs = []
    total_steps = len(doc_data) * images_per_prompt
    step = 0

    os.makedirs("output/images", exist_ok=True)

    for i, doc in enumerate(tqdm(doc_data, desc="Generating images")):
        prompt = doc["text"]
        source = doc["source"]

        prompt_dir = os.path.join("output/images", f"prompt_{i}")
        os.makedirs(prompt_dir, exist_ok=True)

        for j in range(images_per_prompt):
            try:
                result = model(
                    model=tlab_gen.params.model_name,
                    prompt=prompt,
                    width=image_width,
                    height=image_height,
                    num_images=images_per_prompt,
                )

                if isinstance(result, Image.Image):
                    image = result
                elif isinstance(result, dict) and "image" in result:
                    image = Image.open(BytesIO(base64.b64decode(result["image"])))
                else:
                    raise RuntimeError("Unsupported image output format from model")

                image_path = os.path.join(prompt_dir, f"image_{j}.png")
                image.save(image_path)

                outputs.append(
                    {"prompt": prompt, "image_path": os.path.relpath(image_path, "output"), "source_doc": source}
                )

                tlab_gen.output_writer(outputs[-1])

                step += 1
                progress = 10 + (step / total_steps) * 80
                tlab_gen.progress_update(int(progress))

            except Exception as e:
                print(f"Error generating image for prompt {prompt[:30]}...: {e}")
                continue

    df = pd.DataFrame(outputs)

    additional_metadata = {
        "document_count": len(doc_data),
        "images_per_prompt": images_per_prompt,
        "image_width": image_width,
        "image_height": image_height,
    }

    output_file, dataset_name = tlab_gen.save_generated_dataset(df, additional_metadata=additional_metadata)
    print(f"Image dataset generated successfully as dataset {dataset_name}")

    return output_file


print("Starting image dataset generation...")
run_generation()
