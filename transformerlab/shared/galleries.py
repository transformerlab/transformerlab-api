# Functions for managing galleries
import os
import shutil
import urllib.request

from transformerlab.shared import dirs


TLAB_REMOTE_URL = "https://raw.githubusercontent.com/transformerlab/transformerlab-api/main/"
REMOTE_GALLERY_DIR_URL = f"{TLAB_REMOTE_URL}{dirs.GALLERIES_SOURCE_PATH}"


def update_gallery_cache():
    """
    Called when Transformer Lab starts up.
    Initializes any cached gallery files and tries to update from remote.
    """
    filelist = [
        "model-gallery.json"
    ]

    for filename in filelist:
        # First, if not cached then initialize with the local copy.
        localfile = os.path.join(dirs.TFL_SOURCE_CODE_DIR, dirs.GALLERIES_SOURCE_PATH, filename)
        if not os.path.isfile(localfile):
            print("ERROR: Unable to find local gallery file", localfile)
        #shutil.copyfile(src, dst)

        # Then try to update from remote.
        update_cache_from_remote(filename)


def update_cache_from_remote(gallery_filename: str):
    """
    Fetches a gallery file from source and updates the cache
    """
    remote_gallery = REMOTE_GALLERY_DIR_URL + gallery_filename
    local_cache_filename = os.path.join(dirs.GALLERIES_CACHE_DIR, gallery_filename)
    print (f"Caching remote gallery: {remote_gallery}")
    urllib.request.urlretrieve(remote_gallery, local_cache_filename)
