#!/usr/bin/env bash
# Everything should be installed by default
uv pip install "transformerlab-inference>=0.2.38"
uv pip install bitsandbytes

# Uncomment and replace with your local installation for testing
# uv pip install -e /Users/deep.gandhi/transformerlab-repos/FastChat/