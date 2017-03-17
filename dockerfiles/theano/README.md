Definition for a GPU-enabled Docker image with Keras and Theano backend.

You don't need to do this if you can use the already built version defined in your `valohai.yaml`.

```bash
# Building an image.
nvidia-docker build \
    --build-arg PYTHON_VERSION=3.6 \
    --build-arg THEANO_VERSION=rel-0.9.0rc4 \
    --build-arg KERAS_VERSION=2.0.0 \
    -t valohai/keras:2.0.0-theano0.9.0rc4-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04 \
    .

# Validate that the version are right.
nvidia-docker run --rm valohai/keras:2.0.0-theano0.9.0rc4-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04 \
    /bin/bash -c '\
    python --version; \
    python -c "import keras; print(keras.__version__);"; \
    python -c "import theano; print(theano.__version__);"; \
    DEVICE="cuda0" python -c "import pygpu; print(pygpu.__version__);"; \
    nvcc --version | grep release; \
    python -c "import theano; import theano.gpuarray.dnn; print(theano.gpuarray.dnn.version());"; \
    cat /proc/driver/nvidia/version'

# Running pygpu tests required for Theano 0.9.0+, this takes a while though.
nvidia-docker run --rm valohai/keras:2.0.0-theano0.9.0rc4-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04 \
    /bin/bash -c 'DEVICE="cuda0" python -c "import pygpu; pygpu.test();"'
```
