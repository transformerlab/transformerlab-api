# 1. Clone the repo
git clone https://github.com/huggingface/yourbench.git
cd yourbench
git switch -c eb50e1fff84849e19ecb62b45a25ed6afed46a2e


uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
# Need this to keep the current versions and preserve sanity
uv pip install torch==2.6.0 --upgrade
uv pip install textstat evaluate