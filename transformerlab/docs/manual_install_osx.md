# Manuallly Installing Transformer Lab API for MacOS

## Requirements

When running on a Mac, you do not need to use Conda but we recommend it because it creates an isolated environment for you.

Transformer Lab will work on Intel Macs using llama-cpp but in order to do things like training, you will need a Mac with Apple Silicon.

## Steps

### 1. First Install Conda:

https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

### 2. Create a Conda Environment and Install CUDA

```bash
conda create -y -n "transformerlab" python=3.11
```

### 3. Download this Project:

```bash
git clone git@github.com:transformerlab/transformerlab-api.git
```

### 4. Install Python Requirements

```bash
cd transformerlab-api
pip install -r requirements-no-gpu.txt
```
