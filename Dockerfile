FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    unzip \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1+cu118 \
    torchvision \
    numpy \
    scipy \
    pyyaml \
    matplotlib \
    Cython \
    requests \
    opencv-python \
    pillow

WORKDIR /root/dense_fusion

# Expose port for potential visualization
EXPOSE 6006

CMD ["/bin/bash"]