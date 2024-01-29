# The following conda command "run" is equivalent to
# conda activate transformerlab; unicorn api:app --port 8000 --host
conda run -n transformerlab --live-stream uvicorn api:app --port 8000 --host 0.0.0.0 