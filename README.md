# Requirements

## If you want to finetune a model you will need:

* Linux or WSL
* NVIDIA GPU with >= 16 GB of VRAM

## If you just want to do inference:

* Mac M1 or M2
* Windows
* Linux

# Installation

You can use the install script to get the application running: 
```bash
./init.sh
```

This will install [conda](https://docs.conda.io/projects/miniconda/en/latest/) if it's not installed, and then use conda and pip to
install the rest of the applications requirements.

# Run

Within your virtual environment run the following command:

```bash
conda activate transformerlab
python3 api.py
```
 
If you want the API to be available to other machines on your network run:

```bash
conda activate transformerlab
python3 api.py --host 0.0.0.0
```

# Developers:

## Updating Requirements

Dependencies are managed with pip-tools (installed separately). Add new requirements to `requirements.in` and regenerate their corresponding `requirements.txt` files by running the following two commands:

```bash
# default GPU enabled requirements
pip-compile \
    --extra-index-url=https://download.pytorch.org/whl/cu118 \
    --output-file=requirements.txt \
    requirements-gpu.in requirements.in

# requirements for systmes without GPU support
pip-compile \
    --extra-index-url=https://download.pytorch.org/whl/cpu \
    --output-file=requirements-no-gpu.txt \
    requirements.in
```

# Windows Notes

We have not tested running the API on Windows extensively, but it should work.

On WSL, you might need to install CUDA manually by following: 

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

then running the following before you launch: 

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```
 
