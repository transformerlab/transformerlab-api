#!/usr/bin/env bash
# Official Orpheus TTS package (uses vLLM under the hood)
# uv pip install orpheus-speech
git clone https://github.com/canopyai/Orpheus-TTS.git
cd Orpheus-TTS/orpheus_tts_pypi && uv pip install -e . # uses vllm under the hood for fast inference
# Pin vLLM to a known good version per Orpheus README note (Mar 18 release was buggy)
uv pip install vllm==0.7.3 unsloth snac