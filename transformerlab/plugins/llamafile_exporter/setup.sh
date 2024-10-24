#!/usr/bin/env bash

# TODO: Get latest version dynamically?
LATEST_VERSION="0.8.14"

# Download latest built of llamafile
# This is possibly not great because this is 350MB.
# But llamafile is over half of that
curl -L -o llamafile.zip https://github.com/Mozilla-Ocho/llamafile/releases/download/$LATEST_VERSION/llamafile-$LATEST_VERSION.zip
unzip llamafile.zip
cp llamafile-$LATEST_VERSION/bin/llamafile .
cp llamafile-$LATEST_VERSION/bin/zipalign .
rm -rf llamafile-$LATEST_VERSION
rm llamafile.zip

# Set llamafile to be executable
chmod +x llamafile