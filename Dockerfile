# Use CUDA 12.8 base for Blackwell (RTX 50 series) support
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies and Python 3.10 (default in Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and build tools
RUN pip3 install --no-cache-dir --upgrade pip ninja packaging psutil wheel

# Install PyTorch with CUDA 12.8 support (for Blackwell sm_120)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Removed flash-attn to avoid heavy compilation overhead and WSL crashes

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    qwen-tts \
    accelerate \
    scipy \
    gradio \
    faster-whisper \
    soundfile \
    librosa \
    bitsandbytes

# Set default command
CMD ["python", "app.py"]
