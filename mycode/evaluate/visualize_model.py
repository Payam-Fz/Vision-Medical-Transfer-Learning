from datetime import datetime
from packaging import version
import os
from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorboard

# Preprocessing functions from data_util.py in SimCLR repository
from utils.augmentation import preprocess_image
# utilities to plot, time, and score
from utils.analysis import *

from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader

assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

FLAGS = flags.FLAGS

_MODEL_NAME = flags.DEFINE_string(
    'model_name', 'unnamed', 'e.g. "finetuned_REMEDIS"'
)
_MODEL_PATH = flags.DEFINE_string(
    'model_path', './base-models/simclr/r50_2x_sk0/hub/', 'e.g. "./base-models/simclr/r152_2x_sk1/hub/" or "./base-models/remedis/cxr-152x2-remedis-m/"'
)
_IMAGE_SIZE = flags.DEFINE_integer(
    'image_size', 448, 'Input image size.'
)


project_folder = os.getcwd()
if project_folder.endswith('/job'):
    project_folder = project_folder[:-4]
os.makedirs(project_folder + '/out/board', exist_ok=True)

def main(argv):
    
    #_____________SETUP_____________
    MODEL_NAME = _MODEL_NAME.value
    MODEL_PATH = _MODEL_PATH.value
    IMAGE_SIZE = (_IMAGE_SIZE.value, _IMAGE_SIZE.value)
    CHANNELS = 3

    START_TIME = get_curr_datetime()
    
    # for SimClr: TAGS = []  SIGNATURE = 'default'
    # for Remedis: TAGS = ['serve']  SIGNATURE = no signature
    TAGS = ['serve']
    SIGNATURE = ''
    num_images = 1


    #_____________LOAD MODEL______________
    model_path = os.path.join(project_folder, MODEL_PATH)
    # model = hub.load(model_url)
    # model = tf.saved_model.load(model_path, tags=TAGS).signatures['default']
    model = tf.saved_model.load(model_path, tags=TAGS)

    #_____________SETUP LOGGING____________
    logdir=os.path.join(project_folder, "out/board", "visualize_" + MODEL_NAME + "_" + START_TIME)
    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    #___________LOAD IMAGE__________

    def _preprocess_val(x, y, info=None):
        x = preprocess_image(
            x, *IMAGE_SIZE,
            is_training=False, color_distort=False, crop='Center')
        return x

    customLoader = MIMIC_CXR_JPG_Loader({'train': 0, 'validate': num_images, 'test': 0}, project_folder)
    _, val_tfds, _ = customLoader.load()
    val_tfds = val_tfds.shuffle(buffer_size=num_images)
    images_val_tfds = val_tfds.map(_preprocess_val)
    images_val_tfds = tf.convert_to_tensor(list(images_val_tfds), dtype=tf.float32)

    #_____________RUN MODEL____________
    model(images_val_tfds)

    #___________EXPORT TO BOARD__________
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
    
    print('\nDONE!')


if __name__ == '__main__':
    # tf.disable_eager_execution()  # Disable eager mode when running with TF2.
    app.run(main)
