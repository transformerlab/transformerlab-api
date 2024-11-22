#!/usr/bin/env bash
# Everything should be installed by default
if lspci | grep -iq "vga.*amd"; then
    if command -v rocminfo > /dev/null && command -v rocm-smi > /dev/null; then
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --force-reinstall
