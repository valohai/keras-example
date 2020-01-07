import glob
import os
import shutil


def use_valohai_inputs(valohai_input_name, input_file_pattern, keras_cache_dir, keras_example_file):
    """
    Place input files defined through Valohai to cache where Keras example loaders expects them to be.
    This allows skipping download phase if the input file comes from the cache.

    This is optional, just make initialization faster in the context of Keras examples.
    """
    input_dir_base = os.getenv('VH_INPUTS_DIR', './')
    input_dir = os.path.realpath(os.path.join(input_dir_base, valohai_input_name))

    if not os.path.isdir(input_dir):
        print('Could not find Valohai input files at %s, using default Keras downloader as the backup' % input_dir)
        return

    # Find the tar files that were given as inputs to the execution.
    input_tar_paths = glob.glob('%s/%s' % (input_dir, input_file_pattern))
    if not input_tar_paths:
        print(
            'Could not find a %s file at %s, using default Keras downloader as the backup'
            % (input_file_pattern, input_dir)
        )
        return
    input_tar_path = input_tar_paths[0]

    # The default location where Keras example helpers download their datasets.
    data_dir_base = os.path.expanduser(os.path.join('~', '.keras'))
    data_dir = os.path.join(data_dir_base, keras_cache_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    expected_dataset_path = os.path.join(data_dir, keras_example_file)
    if os.path.isfile(expected_dataset_path):
        print('There is already a file at %s, skipping Valohai input usage' % expected_dataset_path)
        return

    try:
        os.symlink(input_tar_path, expected_dataset_path)
    except OSError:
        shutil.copyfile(input_tar_path, expected_dataset_path)
