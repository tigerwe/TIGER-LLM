# TIGER-LLM TensorRT-LLM Inference Server
# Multi-stage build for optimized image size

# ============== Stage 1: Builder ==============
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.10 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装 TensorRT-LLM
# 注意：TensorRT-LLM 安装包较大，可能需要较长时间
RUN pip install --no-cache-dir \
    tensorrt_llm==0.7.1 \
    --extra-index-url https://pypi.nvidia.com

# 安装其他依赖
COPY requirements_server.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_server.txt

# ============== Stage 2: Runtime ==============
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置 Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 从 builder 复制 Python 包
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 创建工作目录
WORKDIR /app

# 复制应用代码
COPY triton_server/ ./triton_server/
COPY model/ ./model/
COPY data/ ./data/

# 创建模型目录
RUN mkdir -p /app/model/ckpt /app/data/Tenrec

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "triton_server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
