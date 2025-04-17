# 1. Clone the repo
git clone https://github.com/huggingface/yourbench.git
cd yourbench
git switch -c eb50e1fff84849e19ecb62b45a25ed6afed46a2e

# Use uv to install the dependencies
# pip install uv # if you do not have uv already
# uv venv
# source .venv/bin/activate
# uv sync
uv sync
uv pip install -e .
uv pip install deepeval langchain-openai instructor anthropic datasets
# Need this to keep the current versions and preserve my sanity
uv pip install torch==2.6.0 --upgrade
uv pip install textstat evaluate