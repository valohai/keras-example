Definition for a GPU-enabled Docker image with Keras and TensorFlow backend.

```bash
# Building an image.
nvidia-docker build \
    --build-arg PYTHON_VERSION=3.6 \
    --build-arg TENSORFLOW_VERSION=1.0.1 \
    --build-arg KERAS_VERSION=2.0.0 \
    -t valohai/keras:2.0.0-tensorflow1.0.1-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04 \
    .

# Validate that the version are right.
nvidia-docker run --rm valohai/keras:2.0.0-tensorflow1.0.1-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04 \
    /bin/bash -c '\
    python --version; \
    python -c "import keras; print(keras.__version__);"; \
    python -c "import tensorflow; print(tensorflow.__version__);"; \
    python -c "from tensorflow.python.client import device_lib; device_lib.list_local_devices();"; \
    nvcc --version | grep release; \
    cat /proc/driver/nvidia/version'
```
