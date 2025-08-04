import json
import os
from pathlib import Path

from typing import Annotated

from fastapi import APIRouter, Body

from transformerlab.db.db import experiment_get

from transformerlab.shared import dirs

from werkzeug.utils import secure_filename
from fastapi.responses import FileResponse


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(path="/list")
async def get_conversations(experimentId: int):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    conversation_dir = experiment_dir + "/conversations/"

    # make directory if it does not exist:
    if not os.path.exists(f"{conversation_dir}"):
        os.makedirs(f"{conversation_dir}")

    # now get a list of all the files in the conversations directory
    conversations_files = []
    for filename in os.listdir(conversation_dir):
        if filename.endswith(".json"):
            conversations_files.append(filename)

    conversations_contents = []

    # now read each conversation and create a list of all conversations
    # and their contents
    for i in range(len(conversations_files)):
        with open(conversation_dir + conversations_files[i], "r") as f:
            new_conversation = {}
            new_conversation["id"] = conversations_files[i]
            # remove .json from end of id
            new_conversation["id"] = new_conversation["id"][:-5]
            new_conversation["contents"] = json.load(f)
            # use file timestamp to get a date
            new_conversation["date"] = os.path.getmtime(conversation_dir + conversations_files[i])
            conversations_contents.append(new_conversation)

    # sort the conversations by date
    conversations_contents.sort(key=lambda x: x["date"], reverse=True)

    return conversations_contents


@router.post(path="/save")
async def save_conversation(
    experimentId: int, conversation_id: Annotated[str, Body()], conversation: Annotated[str, Body()]
):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    conversation_id = secure_filename(conversation_id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    # The following prevents path traversal attacks:
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    conversation_dir = "conversations/"
    final_path = (
        Path(experiment_dir)
        .joinpath(conversation_dir + conversation_id + ".json")
        .resolve()
        .relative_to(experiment_dir)
    )

    final_path = experiment_dir + "/" + str(final_path)

    # now save the conversation
    with open(final_path, "w") as f:
        f.write(conversation)

    return {"message": f"Conversation {conversation_id} saved"}


@router.delete(path="/delete")
async def delete_conversation(experimentId: int, conversation_id: str):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    conversation_id = secure_filename(conversation_id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # The following prevents path traversal attacks:
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    conversation_dir = "conversations/"
    final_path = (
        Path(experiment_dir)
        .joinpath(conversation_dir + conversation_id + ".json")
        .resolve()
        .relative_to(experiment_dir)
    )

    final_path = experiment_dir + "/" + str(final_path)

    # now delete the conversation
    os.remove(final_path)

    return {"message": f"Conversation {conversation_id} deleted"}


@router.get(path="/list_audio")
async def list_audio(experimentId: int):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    audio_dir = os.path.join(experiment_dir, "audio/")

    # temporarily hardcode the audio directory to WORKSPACE_DIR/audio
    audio_dir = os.path.join(dirs.WORKSPACE_DIR, "audio/")

    # now get a list of all the json files in the audio directory
    audio_files_metadata = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(audio_dir, filename)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    # Add the file modification time for sorting
                    data["file_date"] = os.path.getmtime(file_path)
                    audio_files_metadata.append(data)
                except Exception:
                    continue

    # Sort by file modification time (newest first)
    audio_files_metadata.sort(key=lambda x: x["file_date"], reverse=True)

    return audio_files_metadata


@router.get(path="/download_audio")
async def download_audio(experimentId: int, filename: str):
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {experimentId} does not exist"}

    experiment_name = data["name"]

    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    audio_dir = os.path.join(experiment_dir, "audio/")

    # temporarily hardcode the audio directory to WORKSPACE_DIR/audio
    audio_dir = os.path.join(dirs.WORKSPACE_DIR, "audio/")

    # now download the audio file
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        return {"message": f"Audio file {filename} does not exist in experiment {experimentId}"}

    return FileResponse(path=file_path, filename=filename, media_type="audio/mpeg")
