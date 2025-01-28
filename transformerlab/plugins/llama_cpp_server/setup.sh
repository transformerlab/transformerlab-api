#!/usr/bin/env bash
# If we install llama-cpp-python[server] it will install
# Pydantic2 which will break FastChat which depends on Pydantic1
# So we will install llama-cpp-python only and implement our
# own server using FastAPI

pip install llama-cpp-python==0.2.79 --upgrade --force-reinstall --no-cache-dir
