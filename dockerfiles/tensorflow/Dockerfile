FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

MAINTAINER Ruksi <ruksi@valohai.com>

ARG PYTHON_VERSION
ARG TENSORFLOW_VERSION
ARG KERAS_VERSION

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_ROOT="/usr/local/cuda/bin" CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"

# Install apt dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common \
        git \
        curl \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update --fix-missing && \
    apt-get install -y python${PYTHON_VERSION}-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 10

# Install libcupti-dev for TensorFlow (https://www.tensorflow.org/install/install_linux)
RUN /bin/echo -e "deb http://archive.ubuntu.com/ubuntu/ xenial multiverse" >> /etc/apt/sources.list && \
    apt-get update --fix-missing && \
    apt-get install -y libcupti-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Keras
RUN pip install --no-cache-dir git+git://github.com/keras-team/keras.git@${KERAS_VERSION}

# Install TensorFlow
RUN pip install --no-cache-dir tensorflow-gpu==${TENSORFLOW_VERSION}

# Make sure that common data science libraries are installed.
RUN pip install --no-cache-dir \
    h5py \
    numpy \
    scipy \
    pandas \
    sklearn \
    matplotlib \
    Pillow