# Copyright 2023 The medical_research_foundations Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Original code from medical_research_foundations repository in:
#   /colab/REMEDIS_finetuning_example.ipynb
# Modified to add support for MIMIC-CXR-JPG dataset

#------------------- IMPORTS -------------------#

import re
import os
import numpy as np
from time import time
from absl import app
from absl import flags

# import tensorflow.compat.v2 as tf
# tf.compat.v1.enable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import tensorboard
import matplotlib
import matplotlib.pyplot as plt

# LARS optimizer from lars_optimizer.py in SimCLR repository
from lars_optimizer import LARSOptimizer
# macro_soft_f1 optimizer from multi-label-soft-f1 repository
from objective_func import soft_f1_loss, f1_score
# Preprocessing functions from data_util.py in SimCLR repository
from utils.augmentation import preprocess_image
# utilities to plot, time, and score
from utils.analysis import *

from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader


#------------------- SETUP -------------------#


project_folder = os.getcwd()
if project_folder.endswith('/job'):
    project_folder = project_folder[:-4]
os.makedirs(project_folder + '/out', exist_ok=True)
os.makedirs(project_folder + '/out/figs', exist_ok=True)
os.makedirs(project_folder + '/out/models', exist_ok=True)


#------------------- PARAMS -------------------#

FLAGS = flags.FLAGS

_DATASET = flags.DEFINE_string(
    'dataset', 'MIMIC-CXR', '["Chexpert", "Camelyon", "MIMIC-CXR", "Noise"]'
)
_BASE_MODEL_PATH = flags.DEFINE_string(
    'base_model_path', './base-models/simclr/r152_2x_sk1/hub/', 'e.g. "./base-models/simclr/r152_2x_sk1/hub/" or "./base-models/remedis/cxr-152x2-remedis-m/"'
)
_EPOCHS = flags.DEFINE_integer(
    'epochs', 10, 'Number of epochs to perform fine-tuning.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64, 'Batch size for training.'
)
_IMAGE_SIZE = flags.DEFINE_integer(
  'image_size', 448, 'Input image size.'
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.1, 'Initial learning rate per batch size.'
)
_MOMENTUM = flags.DEFINE_float(
  'momentum', 0.9, 'Momentum parameter.'
)
_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay', 1e-6, 'Amount of weight decay to use.'
)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  DATASET = _DATASET.value #@param ["Chexpert", "Camelyon", "MIMIC-CXR", "Noise"]
  BASE_MODEL_PATH = _BASE_MODEL_PATH.value
  BATCH_SIZE = _BATCH_SIZE.value
  LEARNING_RATE = _LEARNING_RATE.value
  EPOCHS = _EPOCHS.value
  MOMENTUM = _MOMENTUM.value
  WEIGHT_DECAY = _WEIGHT_DECAY.value
  IMAGE_SIZE = (_IMAGE_SIZE.value, _IMAGE_SIZE.value)
  CHANNELS = 3

  START_TIME = get_curr_datetime()

  #------------------- PRINT CONFIGURATION -------------------#

  # Print date and time
  print("Start:", START_TIME)
  print('\n')
  print('GPU:', tf.config.list_physical_devices('GPU'))
  print('Dataset:', DATASET)
  print('Model:', BASE_MODEL_PATH)
  print('Epochs:', EPOCHS)
  print('Batch size:', BATCH_SIZE)
  print('Image size:', IMAGE_SIZE)


  #------------------- LOAD DATA -------------------#

  # Chexpert: TFDS.has Supervised - produces binary labels the way we were using them
  #           Chexpert data loader fails unless you have it downloaded - download & put in this directory
  #           Have self-selectors
  # Noise:    fake tfds for testing
  def _preprocess_train(x, y, info=None):
    x = preprocess_image(
        x, *IMAGE_SIZE,
        is_training=True, color_distort=False, crop='Random')
    return x, y

  def _preprocess_val(x, y, info=None):
    x = preprocess_image(
        x, *IMAGE_SIZE,
        is_training=False, color_distort=False, crop='Center')
    return x, y

  if DATASET == 'Noise':
    def generate_fake_tfds_dataset(width, height, channels, num_classes, N=1000, data_type=tf.float32):
      train_examples = np.random.normal(size=[N, width, height, channels])
      classes = np.arange(0, num_classes)
      random_labels = np.random.choice(a=classes, size=N)
      one_hot_encoded = np.zeros((N, num_classes))
      one_hot_encoded[np.arange(N), random_labels] = 1
      train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, one_hot_encoded))
      return train_dataset
    
    num_classes = 14
    train_tfds = generate_fake_tfds_dataset(*IMAGE_SIZE, CHANNELS, num_classes)
    
  elif DATASET == 'Camelyon':
    # NOTE: This is too large to run with the public runtime. Run locally.
    # To see more information about the patch_camelyon dataset, see
    # (https://github.com/basveeling/pcam).
    train_tfds, tfds_info = tfds.load('patch_camelyon',
                                        split='train[:1%]',
                                        with_info = True,
                                        as_supervised = True)
    num_images = tfds_info.splits['train'].num_examples
    num_classes = tfds_info.features['label'].num_classes
    train_tfds = train_tfds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

  elif DATASET == 'Chexpert':
    # TODO: Load chexpert data here.
    num_classes = 14
    raise Exception("not implemented. Please download the chexpert data manually and add code to read here.")

  elif DATASET == 'MIMIC-CXR':
    # customLoader = MIMIC_CXR_JPG_Loader({'train': 360000, 'validate': 2900, 'test': 0}, project_folder)
    customLoader = MIMIC_CXR_JPG_Loader({'train': 5000, 'validate': 200, 'test': 0}, project_folder)
    train_tfds, val_tfds, test_tfds = customLoader.load()
    num_classes = customLoader.metadata['num_classes']

  else:
    raise Exception('The Data Type specified does not have data loading defined.')

  train_tfds = train_tfds.shuffle(buffer_size=2*BATCH_SIZE)
  batched_train_tfds = train_tfds.map(_preprocess_train).batch(BATCH_SIZE)
  val_tfds = val_tfds.shuffle(buffer_size=2*BATCH_SIZE)
  batched_val_tfds = val_tfds.map(_preprocess_val).batch(BATCH_SIZE)

  # TODO: in case improves performance
  # AUTOTUNE = tf.data.experimental.AUTOTUNE
  # batched_train_tfds = batched_train_tfds.prefetch(buffer_size=AUTOTUNE)

  # next_batch = tf.data.make_one_shot_iterator(batched_train_tfds).get_next()

  for f, l in batched_train_tfds.take(1):
    print("Shape of features array:", f.numpy().shape)
    print("Shape of labels array:", l.numpy().shape)

  #------------------- LOAD MODLES -------------------#
  
  # Load module and construct the computation graph
  # Load the base network and set it to non-trainable (for speedup fine-tuning)
  hub_path = os.path.join(project_folder, BASE_MODEL_PATH)
  try:
    feature_extractor_layer = hub.KerasLayer(hub_path, input_shape=(*IMAGE_SIZE, CHANNELS), trainable=False)
  except:
    print(f"""The model {hub_path} did not load. Please verify the model path. It is also worth considering that the model might still be in the process of being uploaded to the designated location. If you have recently uploaded it to a notebook, there could be delays associated with the upload.""")
    raise

  #------------------- SETUP TRAINING HEAD -------------------#
  
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(num_classes, activation='sigmoid', name='multi-label_classifier')
  ])
  
  # TEMP for debugging
  print(model.summary())
  # for batch in batched_val_tfds:
  #     print('probability of an image for classes:', model.predict(batch[0])[:1])
  #     break

  # Setup optimizer and training op.
  # optimizer = LARSOptimizer(
  #     LEARNING_RATE,
  #     momentum=MOMENTUM,
  #     weight_decay=WEIGHT_DECAY,
  #     exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
  # variables_to_train = tf.trainable_variables()
  # train_op = optimizer.minimize(
  #     loss_t, global_step=tf.train.get_or_create_global_step(),
  #     var_list=variables_to_train)

  optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY)
  optimizer.exclude_from_weight_decay(var_names=['batch_normalization', 'bias', 'head_supervised'])

  model.compile(
    optimizer=optimizer,
    loss=soft_f1_loss,
    metrics=[f1_score])

  #------------------- PERFORM FINETUNING -------------------#

  # Define the Keras TensorBoard callback.
  logdir="out/board/fit_" + START_TIME
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  start = time()
  history = model.fit(batched_train_tfds,
                      epochs=EPOCHS,
                      validation_data=batched_val_tfds,
                      callbacks=[tensorboard_callback])
  print('\nTraining took {}'.format(print_time(time.time()-start)))


  #------------------- RESULTS -------------------#

  losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history, os.path.join(project_folder, './out/figs'), START_TIME)
  # model_bce_losses, model_bce_val_losses, model_bce_macro_f1s, model_bce_val_macro_f1s = learning_curves(history_bce)

  # for batch in batched_val_tfds:
  #   print('probability of an image for classes (after finetuning):', model.predict(batch[0])[:1])
  #   break

  print("Macro soft-F1 loss: %.2f" %val_losses[-1])
  print("Macro F1-score: %.2f" %val_macro_f1s[-1])

  for batch in batched_val_tfds:
    show_prediction(*batch, model, os.path.join(project_folder, './out/figs'), START_TIME)
    break
  
  
  #------------------- SAVE MODELS -------------------#
  export_path = project_folder + "./out/models/finetuned_{}".format(START_TIME)
  tf.keras.experimental.export_saved_model(model, export_path)
  print("Model with macro soft-f1 was exported in this path: '{}'".format(export_path))

if __name__ == '__main__':
  # tf.disable_eager_execution()  # Disable eager mode when running with TF2.
  app.run(main)