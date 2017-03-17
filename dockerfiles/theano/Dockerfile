FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Ruksi <ruksi@valohai.com>

ARG PYTHON_VERSION
ARG THEANO_VERSION
ARG KERAS_VERSION

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_ROOT="/usr/local/cuda/bin" LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/libgpuarray/lib"

# Install apt dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common \
        git \
        curl \
        cmake \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN add-apt-repository -y ppa:fkrull/deadsnakes && \
    apt-get update --fix-missing && \
    apt-get install -y python${PYTHON_VERSION}-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 10

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install and configure Keras
# Python keras import initializes the config file.
RUN pip install --no-cache-dir git+git://github.com/fchollet/keras.git@${KERAS_VERSION} && \
    python -c "import keras;" && \
    sed -i'' -- 's/tensorflow/theano/g' /root/.keras/keras.json && \
    pip uninstall -y tensorflow

# Install and configure Theano
RUN pip install --no-cache-dir git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
    /bin/echo -e "\
[global]\n\
device=cuda\n\
floatX=float32\n\
optimizer_including=cudnn\n\
[lib]\n\
cnmem=0.95\n\
[nvcc]\n\
fastmath=True\n\
[cuda]\n\
root = /usr/local/cuda/\n\
[dnn]\n\
library_path = /usr/local/cuda/lib64\n\
include_path = /usr/local/cuda/include\n"\
    > /root/.theanorc;

# Install libgpuarray if Theano 0.9.0+
# https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29
RUN THEANO_MINOR_VERSION=`echo $THEANO_VERSION | sed -e "s#[^0-9]*\([0-9]*\)[.]\([0-9]*\)[.]\([0-9]*\)\([0-9A-Za-z-]*\)#\2#"`; \
    if [ $THEANO_MINOR_VERSION -ge 9 ]; then \
        sed -i'' -- 's/device=gpu/device=cuda/g' /root/.theanorc; \
        cd /; \
        git clone https://github.com/Theano/libgpuarray.git; \
        cd libgpuarray; \
        mkdir Build; \
        cd Build; \
        cmake .. -DCMAKE_BUILD_TYPE=Release; \
        make; \
        make install; \
        cd ..; \
        rm -rf Build; \
        pip install --no-cache-dir cython nose; \
        python setup.py build; \
        python setup.py install; \
    fi

# Make sure that common data science libraries are installed.
RUN pip install --no-cache-dir \
    h5py \
    numpy \
    scipy \
    pandas \
    sklearn \
    matplotlib \
    Pillow
