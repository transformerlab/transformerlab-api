
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS cuda

SHELL ["/bin/bash", "--login", "-c"]

# Step 1. Set up Ubuntu
RUN apt update && apt install --yes wget ssh git git-lfs vim
# NOTE: libcuda.so.1 doesn't exist in NVIDIA's base image, link the stub file to work around
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /transformerlab

# Step 2. Set up Python

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt install python3.11 python3-pip -y

WORKDIR /transformerlab
COPY . /transformerlab

RUN pip install uv
RUN uv pip install --upgrade -r requirements-uv.txt --system

# Now install flash attention

RUN uv pip install packaging --system
RUN uv pip install ninja --system
RUN uv pip install -U flash-attn==2.7.3 --no-build-isolation --system

FROM cuda AS transformerlab-api

WORKDIR /transformerlab

EXPOSE 8338

CMD uv run -v uvicorn api:app --port 8338 --host 0.0.0.0 --no-access-log
