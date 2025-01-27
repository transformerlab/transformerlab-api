<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/transformerlab/transformerlab-app">
    <img src="https://transformerlab.ai/img/flask.svg" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center" style="color: rgb(68, 73, 80); letter-spacing: -1px">Transformer Lab API</h1>

  <p align="center">
    API for <a href="http://github.com/transformerlab/transformerlab-app">Transformer Lab App</a>.
    <br />
    <a href="https://transformerlab.ai/docs/intro"><strong>Explore the docs »</strong></a>
  </p>
</div>

# API for Transformer Lab

This is the API for the [Transformer Lab App](https://github.com/transformerlab/transformerlab-app) which is the main repo for this project. Please go the Transformer Lab App repository to learn more and access documentation.

Use the instructions below if you are installing and running the API on a server, manually.

# Requirements

- An NVIDIA GPU + Linux or Windows with WSL2 support
- or MacOS with Apple Silicon
- If you do not have a GPU or have an Intel Mac, the API will run but will only be able to do inference, but not things like training

# Automatic Installation

You can use the install script to get the application running:

```bash
./install.sh
```

This will install [conda](https://docs.conda.io/projects/miniconda/en/latest/) if it's not installed, and then use conda and pip to
install the rest of the application requirements.

# Manual Installation

If you prefer to install the API without using the install script you can follow the steps on this page:

[https://transformerlab.ai/docs/advanced-install](https://transformerlab.ai/docs/advanced-install)

# Run

Once conda and dependencies are installed, run the following:

```bash
conda activate transformerlab
uvicorn api:app --port 8338 --host 0.0.0.0
```

# Developers:

## Updating Requirements

Dependencies are managed with pip-tools (installed separately). Add new requirements to `requirements.in` and regenerate their corresponding `requirements.txt` files by running the following two commands:

```bash
# default GPU enabled requirements
pip-compile \
    --extra-index-url=https://download.pytorch.org/whl/cu121 \
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
