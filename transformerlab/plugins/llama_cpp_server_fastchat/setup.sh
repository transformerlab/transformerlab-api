#!/usr/bin/env bash
# The following will cause major issues
# because fastchat depends on pydantic1
# while llama-cpp-python depends on pydantic2
# so installing llama-cpp-python will break fastchat
# The best fix would be to fix FastChat
# For now, let's leave this broken.

pip install llama-cpp-python