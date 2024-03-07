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

import re
import os
import numpy as np

# import tensorflow.compat.v2 as tf
# tf.compat.v1.enable_v2_behavior()
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib
import matplotlib.pyplot as plt

from loader.mimic_cxr_jpg_loader import MIMIC_CXR_JPG_Loader

# Preprocessing functions from data_util.py in SimCLR repository
from data_util import preprocess_image
# LARS optimizer from lars_optimizer.py in SimCLR repository (hidden)
from lars_optimizer import LARSOptimizer

#------------------- PARAMS -------------------#
DATASET = "MIMIC-CXR" #@param ["Chexpert", "Camelyon", "MIMIC-CXR", "Noise"]
BASE_MODEL = "REMEDIS" #@param ["SimCLR", "DINO", "REMEDIS"]
BATCH_SIZE = 16
IMAGE_SIZE = (448,448)
NUM_SAMPLES = 400
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.


#------------------- VARS -------------------#
project_folder = os.getcwd()

#------------------- LOAD DATA -------------------#

# Chexpert: TFDS.has Supervised - produces binary labels the way we were using them
#           Chexpert data loader fails unless you have it downloaded - download & put in this directory
#           Have self-selectors
# Noise:    fake tfds for testing
def _preprocess(x, y, info=None):
  out = {}
  out['image'] = preprocess_image(
      x, *IMAGE_SIZE,
      is_training=True, color_distort=False, crop=False)
  out['label'] = y
  out['info'] = info
  return out

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
  train_tfds = generate_fake_tfds_dataset(*IMAGE_SIZE, 3, num_classes, NUM_SAMPLES)
  
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
  customLoader = MIMIC_CXR_JPG_Loader({'train': NUM_SAMPLES, 'validate': 0, 'test': 0}, project_folder)
  train_tfds, val_tfds, test_tfds = customLoader.load()
  num_classes = customLoader.metadata['num_classes']

else:
  raise Exception('The Data Type specified does not have data loading defined.')

train_tfds = train_tfds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
batched_train_tfds = train_tfds.map(_preprocess).batch(BATCH_SIZE)
next_batch = tf.data.make_one_shot_iterator(batched_train_tfds).get_next()


#------------------- LOAD MODLES -------------------#
# Load module and construct the computation graph
# Load the base network and set it to non-trainable (for speedup fine-tuning)

if BASE_MODEL == 'REMEDIS':
  hub_path = os.path.join(project_folder, 'base-models/remedis/cxr-152x2-remedis-m/')
elif BASE_MODEL == 'SimCLR':
  hub_path = os.path.join(project_folder, 'base-models/simclr/r152_2x_sk1/hub/')
elif BASE_MODEL == 'DINO':
  raise Exception('TODO: Not implemented yet.')
  hub_path = os.path.join(project_folder, 'base-models/dino/???')

try:
  module = hub.load(hub_path)
except:
  print(f"""The model {hub_path} did not load. Please verify the model path. It is also worth considering that the model might still be in the process of being uploaded to the designated location. If you have recently uploaded it to a notebook, there could be delays associated with the upload.""")
  raise

#------------------- SETUP TRAINING HEAD -------------------#
key = module(next_batch['image'])

# Attach a trainable linear layer to adapt for the new task.

with tf.variable_scope('head_supervised_new', reuse=tf.AUTO_REUSE):
  # Getting the initial loss
  key = tf.math.reduce_mean(key, axis=[-3, -2])
  logits_t = tf.layers.dense(inputs=key, units=num_classes)
  loss_t = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels=x['label'], logits=logits_t))
  print('initial loss before finetuning:', loss_t)

  # Setup optimizer and training op.
  optimizer = LARSOptimizer(
      LEARNING_RATE,
      momentum=MOMENTUM,
      weight_decay=WEIGHT_DECAY,
      exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
  variables_to_train = tf.trainable_variables()
  train_op = optimizer.minimize(
      loss_t, global_step=tf.train.get_or_create_global_step(),
      var_list=variables_to_train)

for var_initializer in [v.initializer for v in tf.global_variables()]:
  if var_initializer is None:
    print("""Variables have not loaded properly. It is also worth considering that the model might still be in the process of being uploaded to the designated location. If you have recently uploaded it to a notebook, there could be delays associated with the upload. You may also need to restart your runtime.""")
    raise

print('Variables to train:', variables_to_train)
key # The accessible tensor in the return dictionary

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#------------------- PERFORM FINETUNING -------------------#
# We fine-tune the new *linear layer* for just a few iterations.

total_iterations = 10

for it in range(total_iterations):
  _, loss, image, logits, labels = sess.run((train_op, loss_t, x['image'], logits_t, x['label']))
  pred = logits.argmax(-1)
  correct = np.sum(pred == labels)
  total = labels.size
  print("[Iter {}] Loss: {} Top 1: {}".format(it+1, loss, correct/float(total)))

# Plot the images and predictions
fig, axes = plt.subplots(5, 1, figsize=(15, 15))
for i in range(5):
  axes[i].imshow(image[i])
  true_text = str(labels[i])
  pred_text = str(pred[i])
  axes[i].axis('off')
  axes[i].text(256, 128, 'Truth: ' + true_text + '\n' + 'Pred: ' + pred_text)

fname = os.path.join(project_folder, '/out/finetuning', 'finetune_res.png')
plt.savefig(fname)