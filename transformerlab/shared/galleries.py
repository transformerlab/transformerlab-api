# Transformer Lab updates some of its galleries remotely
# It then tries to download the latest version and store in a local cache
# with a backup stored in the server code.
# This is all managed in this file.

import os
import json
import shutil
import urllib.request

from transformerlab.shared import dirs

# This is the list of galleries that are updated remotely
MODEL_GALLERY_FILE = "model-gallery.json"
GALLERY_FILES = [
    MODEL_GALLERY_FILE
]

TLAB_REMOTE_URL = "https://raw.githubusercontent.com/transformerlab/transformerlab-api/main/"
REMOTE_GALLERY_DIR_URL = f"{TLAB_REMOTE_URL}{dirs.GALLERIES_SOURCE_PATH}"


def update_gallery_cache():
    """
    Called when Transformer Lab starts up.
    Initializes any cached gallery files and tries to update from remote.
    """

    for filename in GALLERY_FILES:
        update_gallery_cache_file(filename)


def get_models_gallery():
    return get_gallery_file(MODEL_GALLERY_FILE)


######################
# INTERNAL SUBROUTINES
######################

def gallery_cache_file_path(filename: str):
    return os.path.join(dirs.GALLERIES_CACHE_DIR, filename)


def update_gallery_cache_file(filename: str):
    """
    Initialize the gallery cache file if it doesn't exist from code,
    then try to update from remote.
    """

    # First, if nothing is cached yet, then initialize with the local copy.
    cached_gallery_file = gallery_cache_file_path(filename)
    if not os.path.isfile(cached_gallery_file):
        print(f"Initializing {filename} from local source.")

        sourcefile = os.path.join(dirs.TFL_SOURCE_CODE_DIR, dirs.GALLERIES_SOURCE_PATH, filename)
        if os.path.isfile(sourcefile):
            shutil.copyfile(sourcefile, cached_gallery_file)
        else:
            print("ERROR: Unable to find local gallery file", sourcefile)

    # Then, try to update from remote.
    update_cache_from_remote(filename)


def update_cache_from_remote(gallery_filename: str):
    """
    Fetches a gallery file from source and updates the cache
    """
    try:
        remote_gallery = REMOTE_GALLERY_DIR_URL + gallery_filename
        local_cache_filename = gallery_cache_file_path(gallery_filename)
        urllib.request.urlretrieve(remote_gallery, local_cache_filename)
        print (f"Updated gallery from remote: {remote_gallery}")
    except Exception as e:
        print(f"Failed to update gallery from remote: {remote_gallery}")


def get_gallery_file(filename: str):
    # default empty gallery returned in case of failed gallery file open
    gallery = []
    gallery_path = gallery_cache_file_path(filename)

    # Check for the cached file. If it's not there then initialize.
    if not os.path.isfile(gallery_path):
        print(f"Updating gallery cache file {filename}")
        update_gallery_cache_file(filename)

    with open(gallery_path) as f:
        gallery = json.load(f)

    return gallery

