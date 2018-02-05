Definition for a GPU-enabled Docker image with Keras and TensorFlow backend.

You don't need to do this if you can use the already built version defined in your `valohai.yaml`.

```bash
# Building an image.
nvidia-docker build --no-cache \
    --build-arg PYTHON_VERSION=3.5 \
    --build-arg TENSORFLOW_VERSION=1.4.0 \
    --build-arg KERAS_VERSION=2.1.3 \
    -t valohai/keras:2.1.3-tensorflow1.4.0-python3.5-cuda8.0-cudnn6-devel-ubuntu14.04 \
    .

# Validate that the version are right.
nvidia-docker run --rm valohai/keras:2.1.3-tensorflow1.4.0-python3.5-cuda8.0-cudnn6-devel-ubuntu14.04 \
    /bin/bash -c '\
    python --version; \
    python -c "import keras; print(keras.__version__);"; \
    python -c "import tensorflow; print(tensorflow.__version__);"; \
    python -c "from tensorflow.python.client import device_lib; device_lib.list_local_devices();"; \
    nvcc --version | grep release; \
    cat /proc/driver/nvidia/version'
```
