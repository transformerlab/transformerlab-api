# Transformer Lab updates some of its galleries remotely
# It then tries to download the latest version and store in a local cache
# with a backup stored in the server code.
# This is all managed in this file.

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

    # This is the list of galleries to update remotely
    filelist = [
        "model-gallery.json"
    ]

    for filename in filelist:
        # First, if nothing is cached yet, then initialize with the local copy.
        cached_gallery_file = os.path.join(dirs.GALLERIES_CACHE_DIR, filename)
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
        local_cache_filename = os.path.join(dirs.GALLERIES_CACHE_DIR, gallery_filename)
        urllib.request.urlretrieve(remote_gallery, local_cache_filename)
        print (f"Updated gallery from remote: {remote_gallery}")
    except Exception as e:
        print(f"Failed to update gallery from remote: {remote_gallery}")

