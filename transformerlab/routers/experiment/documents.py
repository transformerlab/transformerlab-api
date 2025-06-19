import datetime
import os
import shutil
import tempfile
import zipfile

import aiofiles
import httpx
from fastapi import APIRouter, Body, HTTPException, UploadFile
from fastapi.responses import FileResponse
from markitdown import MarkItDown
from werkzeug.utils import secure_filename
from urllib.parse import urlparse

from transformerlab.routers.experiment import rag
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify

router = APIRouter(prefix="/documents", tags=["documents"])

allowed_file_types = [".txt", ".jsonl", ".pdf", ".csv", ".epub", ".ipynb", ".md", ".ppt", ".zip"]

# Whitelist of allowed domains for URL validation
ALLOWED_DOMAINS = {"recipes.transformerlab.net", "www.learningcontainer.com"}


def is_valid_url(url: str) -> bool:
    """Validate the URL to ensure it points to an allowed domain."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            return False
        domain = parsed_url.netloc.split(":")[0]  # Extract domain without port
        return domain in ALLOWED_DOMAINS
    except Exception:
        return False


# # Get info on dataset from huggingface
# @router.get("/{document_name}/info", summary="Fetch the details of a particular document.")
# async def document_info():
#     r = {"message": "This endpoint is not yet implemented"}
#     return r


@router.get("/open/{document_name}", summary="View the contents of a document.")
async def document_view(experimentId: str, document_name: str, folder: str = None):
    try:
        experiment_dir = await dirs.experiment_dir_by_id(experimentId)

        document_name = secure_filename(document_name)
        folder = secure_filename(folder)

        if folder and folder != "":
            file_location = os.path.join(experiment_dir, "documents", folder, document_name)
        else:
            file_location = os.path.join(experiment_dir, "documents", document_name)
        print(f"Returning document from {file_location}")
        # with open(file_location, "r") as f:
        #     file_contents = f.read()
    except FileNotFoundError:
        return "error file not found"
    return FileResponse(file_location)


@router.get("/list", summary="List available documents.")
async def document_list(experimentId: str, folder: str = None):
    documents = []
    # List the files that are in the experiment/<experiment_name>/documents directory:
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    folder = secure_filename(folder)
    if folder and folder != "":
        if os.path.exists(os.path.join(documents_dir, folder)):
            documents_dir = os.path.join(documents_dir, folder)
        else:
            return {"status": "error", "message": f'Folder "{folder}" not found'}
    if os.path.exists(documents_dir):
        for filename in os.listdir(documents_dir):
            # check if the filename is a directory:
            if os.path.isdir(os.path.join(documents_dir, filename)):
                name = filename
                size = 0
                date = os.path.getmtime(os.path.join(documents_dir, filename))
                date = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")
                type = "folder"
                path = os.path.join(documents_dir, filename)
                documents.append({"name": name, "size": size, "date": date, "type": type, "path": path})
            elif any(filename.endswith(ext) for ext in allowed_file_types):
                name = filename
                size = os.path.getsize(os.path.join(documents_dir, filename))
                date = os.path.getmtime(os.path.join(documents_dir, filename))
                date = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")
                type = os.path.splitext(filename)[1]
                path = os.path.join(documents_dir, filename)
                documents.append({"name": name, "size": size, "date": date, "type": type, "path": path})

    return documents  # convert list to JSON object


@router.get("/new", summary="Create a new document.")
async def document_new(experimentId: str, dataset_id: str):
    print("Not yet implemented")
    return {"status": "success", "dataset_id": dataset_id}


@router.get("/delete", summary="Delete a document.")
async def delete_document(experimentId: str, document_name: str, folder: str = None):
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)

    document_name = secure_filename(document_name)
    path = os.path.join(experiment_dir, "documents", document_name)
    if folder and folder != "" and not os.path.isdir(path):
        folder = secure_filename(folder)
        path = os.path.join(experiment_dir, "documents", folder, document_name)
    else:
        path = os.path.join(experiment_dir, "documents", document_name)
    # first check if it is a directory:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    return {"status": "success"}


@router.post("/upload", summary="Upload the contents of a document.")
async def document_upload(experimentId: str, folder: str, files: list[UploadFile]):
    fileNames = []
    md = MarkItDown(enable_plugins=False)

    # Adding secure filename to the folder name as well
    folder = secure_filename(folder)
    for file in files:
        file_name = secure_filename(file.filename)
        print("uploading filename is: " + str(file_name))
        #
        fileNames.append(file_name)
        # ensure the filename is exactly {dataset_id}_train.jsonl or {dataset_id}_eval.jsonl
        # if not re.match(rf"^{dataset_id}_(train|eval).jsonl$", str(file.filename)):
        #     raise HTTPException(
        #         status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

        print("file content type is: " + str(file.content_type))

        restricted_file_type = False

        if file.content_type not in ["text/plain", "application/json", "application/pdf", "application/octet-stream"]:
            restricted_file_type = True
            print("File Type is Restricted from viewing, we will paste it as an md file instead")

            if file.content_type.startswith("image/"):
                raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        file_ext = os.path.splitext(file_name)[1]
        # if file_ext not in allowed_file_types:
        #     raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        experiment_dir = await dirs.experiment_dir_by_id(experimentId)
        documents_dir = os.path.join(experiment_dir, "documents")
        if folder and folder != "":
            if os.path.exists(os.path.join(documents_dir, folder)):
                documents_dir = os.path.join(documents_dir, folder)
            else:
                print(f"Creating directory as it doesn't exist: {os.path.join(documents_dir, folder)}")
                os.makedirs(os.path.join(documents_dir, folder))
                documents_dir = os.path.join(documents_dir, folder)

        markitdown_dir = os.path.join(documents_dir, ".tlab_markitdown")
        if not os.path.exists(markitdown_dir):
            os.makedirs(markitdown_dir)

        if not restricted_file_type:
            # Save the file to the dataset directory
            try:
                content = await file.read()
                if not os.path.exists(documents_dir):
                    print("Creating directory")
                    os.makedirs(documents_dir)

                newfilename = os.path.join(documents_dir, str(file_name))
                async with aiofiles.open(newfilename, "wb") as out_file:
                    await out_file.write(content)

                # Convert file to .md format using MarkitDown and save it in markitdown_dir
                # Do not do this for .jpeg, .jpg, .png, .gif, .webp
                # Check if the file is an image
                if file_ext not in [".jpeg", ".jpg", ".png", ".gif", ".webp"]:
                    try:
                        result = md.convert(newfilename)
                        # Save the converted file
                        newfilename = os.path.join(markitdown_dir, str(file_name).replace(file_ext, ".md"))
                        print(f"Saving converted file to {markitdown_dir}")

                        async with aiofiles.open(newfilename, "w", encoding="utf-8") as out_file:
                            await out_file.write(result.markdown)

                    except Exception as e:
                        print(f"Error converting file to .md format: {e}")
            except Exception:
                raise HTTPException(status_code=403, detail="There was a problem uploading the file")
        else:
            # Do the conversion to md using MarkitDown
            # Save the file to the dataset directory
            try:
                content = await file.read()
                # from io import BytesIO

                # content_io = BytesIO(content)
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    print(f"Temporary file created at {temp_file_path}")
                    # Convert the file to .md format using MarkitDown
                    result = md.convert(temp_file_path)
                    # Save the converted file
                    newfilename = os.path.join(documents_dir, str(file_name).replace(file_ext, ".md"))
                    newfilename_md = os.path.join(markitdown_dir, str(file_name).replace(file_ext, ".md"))
                    async with aiofiles.open(newfilename, "w", encoding="utf-8") as out_file:
                        await out_file.write(result.markdown)
                    print(f"Saving converted file to {markitdown_dir} as well")
                    async with aiofiles.open(newfilename_md, "w", encoding="utf-8") as out_file:
                        await out_file.write(result.markdown)
                    # Remove the temporary file
                    os.remove(temp_file_path)
                    print(f"Temporary file {temp_file_path} deleted")
            except Exception as e:
                print(f"Error converting file to .md format: {e}")
                raise HTTPException(status_code=403, detail="There was a problem uploading the file")

        # reindex the vector store on every file upload
        if folder == "rag":
            await rag.reindex(experimentId)

    return {"status": "success", "filename": fileNames}


@router.post("/create_folder", summary="Create a new folder.")
async def create_folder(experimentId: str, name: str):
    name = slugify(name)
    # Secure folder name
    name = secure_filename(name)
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    path = os.path.join(experiment_dir, "documents", name)
    print(f"Creating folder {path}")
    if not os.path.exists(path):
        os.makedirs(path)
    return {"status": "success"}


@router.post("/upload_links", summary="Upload the contents from the provided web links.")
async def document_upload_links(experimentId: str, folder: str = None, data: dict = Body(...)):
    urls = data.get("urls")
    folder = secure_filename(folder)
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    if folder and folder != "":
        if os.path.exists(os.path.join(documents_dir, folder)):
            documents_dir = os.path.join(documents_dir, folder)
        else:
            return {"status": "error", "message": f'Folder "{folder}" not found'}

    markitdown_dir = os.path.join(documents_dir, ".tlab_markitdown")

    if not os.path.exists(markitdown_dir):
        os.makedirs(markitdown_dir)

    md = MarkItDown(enable_plugins=False)
    for i, url in enumerate(urls):
        result = md.convert(url)
        # Save the converted file
        filename = os.path.join(documents_dir, f"link_{i + 1}.md")
        filename_md = os.path.join(markitdown_dir, f"link_{i + 1}.md")
        async with aiofiles.open(filename, "w", encoding="utf-8") as out_file:
            await out_file.write(result.markdown)

        async with aiofiles.open(filename_md, "w", encoding="utf-8") as out_file:
            await out_file.write(result.markdown)
        # reindex the vector store on every file upload
        if folder == "rag":
            await rag.reindex(experimentId)
    return {"status": "success", "filename": urls}


async def document_download_zip(experimentId: str, data: dict = Body(...)):
    """Download a ZIP file from a URL and extract its contents to the documents folder."""
    url = data.get("url")

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Validate the URL
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid or unauthorized URL")

    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")

    try:
        # Download ZIP file
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Save to temporary file and extract
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
                temp_zip.write(response.content)
                temp_zip_path = temp_zip.name

        # Extract ZIP file
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(documents_dir)
            extracted_files = [f for f in zip_ref.namelist() if not f.endswith("/") and not f.startswith(".")]

        # Clean up
        os.remove(temp_zip_path)

        # Reindex RAG if any files were extracted to a 'rag' folder
        rag_files = [f for f in extracted_files if f.startswith("rag/")]
        if rag_files:
            await rag.reindex(experimentId)

        return {"status": "success", "extracted_files": extracted_files, "total_files": len(extracted_files)}

    except httpx.HTTPStatusError:
        if "temp_zip_path" in locals() and os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=400, detail="Failed to download ZIP file: HTTP error.")
    except zipfile.BadZipFile:
        if "temp_zip_path" in locals() and os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=400, detail="Downloaded file is not a valid ZIP archive")
    except Exception:
        if "temp_zip_path" in locals() and os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=500, detail="Error processing ZIP file.")
