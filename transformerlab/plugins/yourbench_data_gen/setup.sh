# 1. Clone the repo
git clone https://github.com/huggingface/yourbench.git
cd yourbench

# Use uv to install the dependencies
# pip install uv # if you do not have uv already
# uv venv
# source .venv/bin/activate
# uv sync
uv pip install -e .
# Need this to keep the current versions and preserve my sanity
uv pip install torch==2.6.0 --upgrade
uv pip install textstat evaluate