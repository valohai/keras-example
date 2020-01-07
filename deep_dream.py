"""
Original: https://github.com/keras-team/keras/blob/master/examples/deep_dream.py

#Deep Dreaming in Keras.

Run the script with:
```python
python deep_dream.py path_to_your_base_image.jpg prefix_for_results iterations step num_octave octave_scale
```
e.g.:
```python
python deep_dream.py img/my-pic.jpg results/dream 5 0.01 3 1.4
```
"""
from __future__ import print_function

import glob
import json
import os
import argparse

from keras import backend as K
from keras.applications import inception_v3
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy

from utils import use_valohai_inputs


def dream(cli_params):
    base_image_path = cli_params.base_image_path
    result_prefix = cli_params.result_prefix
    iterations = cli_params.iterations
    step = cli_params.step
    num_octave = cli_params.num_octave
    octave_scale = cli_params.octave_scale

    if not os.path.isdir(base_image_path):
        raise Exception('base must be a directory')

    # Find the first image in the given directory.
    file_types = ('*.jpg', '*.png')
    image_files = []
    for file_type in file_types:
        image_files.extend(glob.glob('{}/*{}'.format(base_image_path, file_type)))
    if not image_files:
        types_as_str = ', '.join(file_types)
        raise Exception('no image files ({}) under {}'.format(types_as_str, base_image_path))
    image_file = image_files[0]

    # These are the names of the layers
    # for which we try to maximize activation,
    # as well as their weight in the final loss
    # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    settings = {
        'features': {
            'mixed2': 0.2,
            'mixed3': 0.5,
            'mixed4': 2.,
            'mixed5': 1.5,
        },
    }

    def preprocess_image(image_path):
        # Util function to open, resize and format pictures
        # into appropriate tensors.
        img = load_img(image_path)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img

    def postprocess_image(x):
        # Util function to convert a tensor into a valid image.
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((x.shape[1], x.shape[2], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    K.set_learning_phase(0)

    # Build the InceptionV3 network with our placeholder.
    # The model will be loaded with pre-trained ImageNet weights.
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    dream = model.input
    print('Model loaded.')

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Define the loss.
    loss = K.variable(0.)
    for layer_name in settings['features']:
        # Add the L2 norm of the features of a layer to the loss.
        if layer_name not in layer_dict:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
        coeff = settings['features'][layer_name]
        x = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    def eval_loss_and_grads(x):
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values

    def resize_img(img, size):
        img = np.copy(img)
        if K.image_data_format() == 'channels_first':
            factors = (1, 1,
                       float(size[0]) / img.shape[2],
                       float(size[1]) / img.shape[3])
        else:
            factors = (1,
                       float(size[0]) / img.shape[1],
                       float(size[1]) / img.shape[2],
                       1)
        return scipy.ndimage.zoom(img, factors, order=1)

    def gradient_ascent(x, shape_number, iterations, step, max_loss=None):
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_loss:
                break
            print(json.dumps({
                'shape_number': shape_number,
                'iteration': i,
                'loss_value': loss_value.item(),
            }))
            x += step * grad_values
        return x

    """Process:

    - Load the original image.
    - Define a number of processing scales (i.e. image shapes),
        from smallest to largest.
    - Resize the original image to the smallest scale.
    - For every scale, starting with the smallest (i.e. current one):
        - Run gradient ascent
        - Upscale image to the next scale
        - Reinject the detail that was lost at upscaling time
    - Stop when we are back to the original size.

    To obtain the detail lost during upscaling, we simply
    take the original image, shrink it down, upscale it,
    and compare the result to the (resized) original image.
    """

    # Playing with these hyperparameters will also allow you to achieve new effects
    max_loss = 10.

    img = preprocess_image(image_file)
    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    shape_number = 0
    for shape in successive_shapes:
        print('Processing image shape', shape)
        shape_number += 1
        img = resize_img(img, shape)
        img = gradient_ascent(img, shape_number=shape_number, iterations=iterations, step=step, max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

    save_img(result_prefix + '.png', postprocess_image(np.copy(img)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
    parser.add_argument('base_image_path', type=str)
    parser.add_argument('result_prefix', type=str)
    parser.add_argument('iterations', type=int)
    parser.add_argument('step', type=float)
    parser.add_argument('num_octave', type=int)
    parser.add_argument('octave_scale', type=float)
    cli_parameters = parser.parse_args()
    use_valohai_inputs(
        valohai_input_name='inception-model',
        input_file_pattern='*.h5',
        keras_cache_dir='models',
        keras_example_file='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    )
    dream(cli_parameters)
