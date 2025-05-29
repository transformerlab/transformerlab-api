from fastapi.testclient import TestClient
from api import app
from io import BytesIO
import json
import os
from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify
import shutil


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


def test_save_metadata():
    test_data = [
        {"__index__": 0, "image": "dummy.jpg", "text": "caption A"},
        {"__index__": 1, "image": "dummy2.jpg", "text": "caption B"},
    ]

    with TestClient(app) as client:
        response = client.post(
            "/data/save_metadata",
            data={"dataset_id": "dummy_dataset"},
            files={"file": ("patch.json", BytesIO(json.dumps(test_data).encode("utf-8")), "application/json")},
        )
        assert response.status_code in (200, 400, 404)


def test_save_metadata_and_preview():
    test_dataset_id = "dummy_preview_dataset"
    test_rows = [
        {"file_name": "dummy0.jpg", "previous_caption": "Original caption 0", "text": "Updated caption 1"},
        {"file_name": "dummy1.jpg", "previous_caption": "Original caption 1", "text": "Updated caption 2"},
    ]

    with TestClient(app) as client:
        dataset_dir = dirs.dataset_dir_by_id(slugify(test_dataset_id))
        os.makedirs(dataset_dir, exist_ok=True)
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

        # Create initial metadata with matching file names and captions
        with open(metadata_path, "w", encoding="utf-8") as f:
            for i in range(2):
                row = {"file_name": f"dummy{i}.jpg", "text": f"Original caption {i}"}
                f.write(json.dumps(row) + "\n")

        # Post updated metadata
        response = client.post(
            f"/data/save_metadata?dataset_id={test_dataset_id}",
            files={"file": ("patch.json", BytesIO(json.dumps(test_rows).encode("utf-8")), "application/json")},
        )
        assert response.status_code == 200

        # Reload metadata and check updated captions
        with open(metadata_path, "r", encoding="utf-8") as f:
            updated_rows = [json.loads(line) for line in f]
        for updated, expected in zip(updated_rows, test_rows):
            assert updated["text"] == expected["text"]

        # Test preview works
        resp = client.get(f"/data/preview?dataset_id={test_dataset_id}")
        assert resp.status_code == 200 or resp.status_code == 400

    cleanup_dataset(test_dataset_id)


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
        test_dataset_id = "dummy_info_dataset"
        dataset_dir = dirs.dataset_dir_by_id(slugify(test_dataset_id))
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
        response = client.post(f"/data/fileupload?dataset_id={test_dataset_id}", files=files)
        assert response.status_code == 200

        # 🔥 Register the dataset via /data/new or manually (adjust if needed)
        register_response = client.get(f"/data/new?dataset_id={test_dataset_id}")
        assert register_response.status_code in (200, 400)

        # Call /info
        resp = client.get(f"/data/info?dataset_id={test_dataset_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert "splits" in data
        assert "is_image" in data
        assert "is_parquet" in data
        assert data["is_image"] is True
        assert data["is_parquet"] is False

    cleanup_dataset(test_dataset_id)


def test_data_preview_with_template():
    with TestClient(app) as client:
        test_dataset_id = "dummy_preview_2_dataset"
        template = "{{ text }}"
        dataset_dir = dirs.dataset_dir_by_id(slugify(test_dataset_id))
        os.makedirs(dataset_dir, exist_ok=True)

        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "image": "image.jpg",
                        "text": "hello",
                    }
                )
                + "\n"
            )

        resp = client.get(f"/data/preview_with_template?dataset_id={test_dataset_id}&template={template}&limit=1")
        assert resp.status_code in (200, 400, 404)
        if resp.status_code == 200 and resp.json()["status"] == "success":
            data = resp.json()["data"]
            assert "columns" in data
            assert "rows" in data
        cleanup_dataset(test_dataset_id)


def test_duplicate_dataset():
    with TestClient(app) as client:
        original_id = "dummy_original_dataset"
        new_id = "dummy_copied_dataset"
        original_dir = dirs.dataset_dir_by_id(slugify(original_id))
        os.makedirs(original_dir, exist_ok=True)

        # Create dummy file for dataset
        image_path = os.path.join(original_dir, "file.txt")
        with open(image_path, "w") as f:
            f.write("Hello, world!")

        # Create consistent metadata
        metadata_content = json.dumps({"file_name": "file.txt", "text": "hello", "label": "test_label"}) + "\n"
        metadata_filename = "metadata.jsonl"

        # Upload metadata using /fileupload
        files = {"files": (metadata_filename, BytesIO(metadata_content.encode()), "application/jsonl")}
        upload_resp = client.post(f"/data/fileupload?dataset_id={original_id}", files=files)
        assert upload_resp.status_code == 200
        assert upload_resp.json()["status"] == "success"

        # Register dataset using /data/new
        register_resp = client.get(f"/data/new?dataset_id={original_id}")
        assert register_resp.status_code in (200, 400)

        # Attempt to duplicate dataset
        resp = client.post(f"/data/duplicate_dataset?dataset_id={original_id}&new_dataset_id={new_id}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] in ("success", "error")

        # Check if new dataset dir exists
        new_dir = dirs.dataset_dir_by_id(slugify(new_id))
        print("Test dataset dir:", dirs.dataset_dir_by_id(slugify(new_id)))
        assert os.path.exists(new_dir), f"Expected new dir {new_dir} to exist"

        # Edge case: duplicate nonexistent dataset
        resp = client.post("/data/duplicate_dataset?dataset_id=nonexistent&new_dataset_id=newset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    cleanup_dataset(original_id)
    cleanup_dataset(new_id)
