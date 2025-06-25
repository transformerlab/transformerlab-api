from fastapi.testclient import TestClient
from api import app
from io import BytesIO
import json
import os
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
from datasets import DatasetDict, Dataset




def cleanup_dataset(dataset_id):
    with TestClient(app) as client:
        dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))
        shutil.rmtree(dataset_dir, ignore_errors=True)
        client.get(f"/data/delete?dataset_id={dataset_id}")


def test_data_gallery():
    with TestClient(app) as client:
        resp = client.get("/data/gallery")
        assert resp.status_code == 200
        assert "data" in resp.json() or "status" in resp.json()


def test_data_list():
    with TestClient(app) as client:
        resp = client.get("/data/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_data_preview():
    with TestClient(app) as client:
        resp = client.get("/data/preview?dataset_id=dummy_dataset")
        assert resp.status_code in (200, 400, 404)


def test_data_preview_trelis_touch_rugby_rules():
    with TestClient(app) as client:
        resp = client.get("/data/preview", params={"dataset_id": "Trelis/touch-rugby-rules", "limit": 2})
        assert resp.status_code in (200, 400, 404)
        if resp.status_code == 200 and resp.json().get("status") == "success":
            data = resp.json()["data"]
            assert "len" in data
            # Should have either columns or rows
            assert "columns" in data or "rows" in data


def test_data_info():
    with TestClient(app) as client:
        resp = client.get("/data/info?dataset_id=dummy_dataset")
        assert resp.status_code in (200, 400, 404)


def test_save_metadata():
    with TestClient(app) as client:
        source_dataset_id = "source_dataset"
        new_dataset_id = "destination_dataset"
        dataset_dir = dirs.dataset_dir_by_id(slugify(source_dataset_id))
        os.makedirs(dataset_dir, exist_ok=True)

        # Create dummy JPEG image
        image_path = os.path.join(dataset_dir, "image.jpg")
        with open(image_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"JPEG DUMMY" + b"\xff\xd9")

        # Prepare metadata JSONL
        metadata_content = json.dumps({"file_name": "image.jpg", "text": "sample caption"}) + "\n"
        metadata_filename = "metadata.jsonl"

        # Upload metadata
        files = {"files": (metadata_filename, BytesIO(metadata_content.encode()), "application/jsonl")}
        response = client.post(f"/data/fileupload?dataset_id={source_dataset_id}", files=files)
        assert response.status_code == 200

        # Register the dataset via /data/new or manually (adjust if needed)
        register_response = client.get(f"/data/new?dataset_id={source_dataset_id}")
        assert register_response.status_code in (200, 400)

        updates = [
            {
                "file_name": "image1.jpg",
                "previous_label": "cat",
                "previous_split": "train",
                "previous_caption": "Old Caption",
                "label": "cat",
                "split": "train",
                "caption": "New Caption",
            }
        ]
        updates_json = BytesIO(json.dumps(updates).encode("utf-8"))

        response = client.post(
            f"/data/save_metadata?dataset_id={source_dataset_id}&new_dataset_id={new_dataset_id}",
            files={"file": ("metadata_updates.json", updates_json, "application/json")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        new_dataset_dir = Path(dirs.dataset_dir_by_id(slugify(new_dataset_id)))
        assert new_dataset_dir.exists()

    cleanup_dataset(source_dataset_id)
    cleanup_dataset(new_dataset_id)


def test_edit_with_template():
    with TestClient(app) as client:
        dataset_id = "test_dataset"
        dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))
        os.makedirs(dataset_dir, exist_ok=True)

        image_path = os.path.join(dataset_dir, "image.jpg")
        image = Image.new("RGB", (100, 100), color="red")
        image.save(image_path, "JPEG")

        metadata_content = (
            json.dumps({"file_name": "image.jpg", "text": "sample caption", "label": "cat", "split": "train"}) + "\n"
        )
        metadata_filename = "metadata.jsonl"

        files = {"files": (metadata_filename, BytesIO(metadata_content.encode()), "application/jsonl")}
        response = client.post(f"/data/fileupload?dataset_id={dataset_id}", files=files)
        assert response.status_code == 200

        register_response = client.get(f"/data/new?dataset_id={dataset_id}")
        assert register_response.status_code in (200, 400)

        response = client.get(f"/data/edit_with_template?dataset_id={dataset_id}&template=")
        assert response.status_code == 200
        data = response.json()
        print("Response JSON:", data)
        assert data["status"] == "success"
        rows = data["data"]["rows"]
        assert len(rows) > 0
        row = rows[0]
        assert "file_name" in row
        assert "image" in row
        assert "text" in row
        assert "label" in row
        assert "split" in row

    cleanup_dataset(dataset_id)

def test_dataset_preview_with_chat_template_mocked():
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict({
            "message": [
                [{"role": "user", "content": "Hello!"}],
                [{"role": "user", "content": "What is AI?"}],
            ]
        })
    })

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = (
        lambda messages, tokenize=False: f"<formatted>{messages[0]['content']}</formatted>"
    )

    with (
        TestClient(app) as client,
        patch("api.routes.data.load_dataset", return_value=dataset_dict),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        resp = client.get(
            "/data/preview_with_chat_template",
            params={
                "model_name": "test-model",
                "chat_column": "message",
                "dataset_id": "mock-dataset",
                "template": "",
                "offset": 0,
                "limit": 2
            }
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "success"
        rows = body["data"]["rows"]
        assert len(rows) == 2
        
        assert rows[0]["__formatted__"] == "<formatted>Hello!</formatted>", "Formatted output mismatch"
        