#!/usr/bin/env bash
# Unfortunately, llama-cpp-python doesn't support conversion yet
# pip install llama-cpp-python
# So need to copy llama.cpp and run from there
git clone https://github.com/ggerganov/llama.cpp/
cd llama.cpp
git checkout 8f275a7c4593aa34147595a90282cf950a853690 # this is a known good version