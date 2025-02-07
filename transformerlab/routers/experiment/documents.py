import os
from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException, UploadFile
import datetime
import aiofiles
from transformerlab.routers.experiment import rag
from transformerlab.shared import dirs

router = APIRouter(prefix="/documents", tags=["documents"])

allowed_file_types = [".txt", ".jsonl", ".pdf"]


# Get info on dataset from huggingface
@router.get("/{document_name}/info", summary="Fetch the details of a particular document.")
async def document_info():
    r = {"message": "This endpoint is not yet implemented"}
    return r


@router.get("/open/{document_name}", summary="View the contents of a document.")
async def document_view(experimentId: str, document_name: str):
    try:
        experiment_dir = await dirs.experiment_dir_by_id(experimentId)
        file_location = os.path.join(experiment_dir, "documents", document_name)
        print(f"Returning document from {file_location}")
        # with open(file_location, "r") as f:
        #     file_contents = f.read()
    except FileNotFoundError:
        return "error file not found"
    return FileResponse(file_location)


@router.get("/list", summary="List available documents.")
async def document_list(experimentId: str):
    documents = []
    # List the files that are in the experiment/<experiment_name>/documents directory:
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    if os.path.exists(documents_dir):
        for filename in os.listdir(documents_dir):
            if any(filename.endswith(ext) for ext in allowed_file_types):
                name = filename
                size = os.path.getsize(os.path.join(documents_dir, filename))
                date = os.path.getmtime(os.path.join(documents_dir, filename))
                date = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")
                type = os.path.splitext(filename)[1]
                documents.append({"name": name, "size": size, "date": date, "type": type})

    return documents  # convert list to JSON object


@router.get("/new", summary="Create a new document.")
async def document_new(dataset_id: str):
    print("Not yet implemented")
    return {"status": "success", "dataset_id": dataset_id}


@router.get("/delete/{document_name}", summary="Delete a document.")
async def delete_document(experimentId: str, document_name: str):
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    path = os.path.join(experiment_dir, "documents", document_name)
    if os.path.exists(path):
        os.remove(path)
    return {"status": "success"}


@router.post("/upload", summary="Upload the contents of a document.")
async def document_upload(experimentId: str, files: list[UploadFile]):
    fileNames = []
    for file in files:
        print("uploading filename is: " + str(file.filename))
        fileNames.append(file.filename)
        # ensure the filename is exactly {dataset_id}_train.jsonl or {dataset_id}_eval.jsonl
        # if not re.match(rf"^{dataset_id}_(train|eval).jsonl$", str(file.filename)):
        #     raise HTTPException(
        #         status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

        print("file content type is: " + str(file.content_type))

        if file.content_type not in ["text/plain", "application/json", "application/pdf", "application/octet-stream"]:
            raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in allowed_file_types:
            raise HTTPException(status_code=403, detail="The file must be a text file, a JSONL file, or a PDF")

        experiment_dir = await dirs.experiment_dir_by_id(experimentId)

        if not os.path.exists(os.path.join(experiment_dir, "documents")):
            os.makedirs(os.path.join(experiment_dir, "documents"))

        # Save the file to the dataset directory
        try:
            content = await file.read()
            newfilename = os.path.join(experiment_dir, "documents", str(file.filename))
            async with aiofiles.open(newfilename, "wb") as out_file:
                await out_file.write(content)

            # reindex the vector store on every file upload
            await rag.reindex(experimentId)
        except Exception:
            raise HTTPException(status_code=403, detail="There was a problem uploading the file")

    return {"status": "success", "filename": fileNames}
