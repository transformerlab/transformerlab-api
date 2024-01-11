# Manuallly Installing Transformer Lab API

## Requirements

The API should run on Mac, Windows (with WSL), and Linux.

But if you are doing training, you most likely will need a Windows or Linux machine that has access to a GPU, and has CUDA drivers installed.

Ensure that the server you run the API is network accessible by the App.

## Steps

### 1. First Install Conda:

https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

### 2. Create a Conda Environment and Install CUDA

```bash
conda create -y -n "transformerlab" python=3.10
conda install -y cuda -c nvidia/label/cuda-12.1.1
```

### 3. Download this Project:
```bash
git clone git@github.com:transformerlab/transformerlab-api.git
```

### 4. Install Python Requirements
```bash
cd transformerlab-api
pip install -r requirements.txt
```