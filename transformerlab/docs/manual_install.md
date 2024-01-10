# Manuallly Installing Transformer Lab API

## Requirements

The API should run on Mac, Windows (with WSL), and Linux.

But if you are doing training, you most likely will need a Windows or Linux machine that has access to a GPU, and has CUDA drivers installed.

Ensure that the server you run the API is network accessible by the App.

## Steps

### 1. First Install Conda

Instructions for installing miniconda for your OS can be found in the Conda projects documentation: [How to install Miniconda][miniconda].

### 2. Create a Conda Environment

```bash
conda create -y -n "transformerlab" python=3.10
```

### 3. [Optional] Install CUDA

If you are running on a system with a GPU, install the Nvidia CUDA python bindings.

```bash
conda install -y cuda -c nvidia/label/cuda-11.8.0
```

### 4. Download this Project

```bash
git clone git@github.com:transformerlab/transformerlab-api.git
```

### 4. Install Python Requirements

The python requirements are stored in a text file. On a system without a GPU, use `requirements-no-gpu.txt` instead of `requirements.txt` in the instructions below.

```bash
cd transformerlab-api
pip install -r requirements.txt
```


[miniconda]: https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install