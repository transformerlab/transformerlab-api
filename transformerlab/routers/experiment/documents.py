import os
import shutil
from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException, UploadFile
import datetime
import aiofiles
from transformerlab.routers.experiment import rag
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify

from werkzeug.utils import secure_filename

router = APIRouter(prefix="/documents", tags=["documents"])

allowed_file_types = [".txt", ".jsonl", ".pdf", ".csv", ".epub", ".ipynb", ".md", ".ppt"]


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

        if file.content_type not in ["text/plain", "application/json", "application/pdf", "application/octet-stream"]:
            raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        file_ext = os.path.splitext(file_name)[1]
        if file_ext not in allowed_file_types:
            raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        experiment_dir = await dirs.experiment_dir_by_id(experimentId)
        documents_dir = os.path.join(experiment_dir, "documents")
        if folder and folder != "":
            if os.path.exists(os.path.join(documents_dir, folder)):
                documents_dir = os.path.join(documents_dir, folder)
            else:
                print(f"Creating directory as it doesn't exist: {os.path.join(documents_dir, folder)}")
                os.makedirs(os.path.join(documents_dir, folder))
                documents_dir = os.path.join(documents_dir, folder)

        # Save the file to the dataset directory
        try:
            content = await file.read()
            if not os.path.exists(documents_dir):
                print("Creating directory")
                os.makedirs(documents_dir)

            newfilename = os.path.join(documents_dir, str(file_name))
            async with aiofiles.open(newfilename, "wb") as out_file:
                await out_file.write(content)

            # reindex the vector store on every file upload
            if folder == "rag":
                await rag.reindex(experimentId)
        except Exception:
            raise HTTPException(status_code=403, detail="There was a problem uploading the file")

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
