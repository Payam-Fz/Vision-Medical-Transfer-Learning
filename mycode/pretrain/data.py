# Copyright 2023 The medical_research_foundations Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data pipeline."""

import functools
from typing import Dict, Any, Optional, Callable

from absl import flags
import tensorflow.compat.v1 as tf
from . import data_util

FLAGS = flags.FLAGS


def pad_to_batch(dataset, batch_size):
  """Pad Tensors to specified batch size.

  Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

  Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

  Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
  """

  def _pad_to_batch(*args):
    """Given Tensors yielded by a Dataset, pads all to the batch size."""
    flat_args = tf.nest.flatten(args)

    for tensor in flat_args:
      if tensor.shape.ndims is None:
        print('flat_args:',flat_args)
        print('tensor:',tensor)
        print('tensor_shape:',tensor.shape)
        raise ValueError(
            'Unknown number of dimensions for tensor %s.' % tensor.name
        )
      if tensor.shape.ndims == 0:
        raise ValueError('Tensor %s is a scalar.' % tensor.name)

    # This will throw if flat_args is empty. However, as of this writing,
    # tf.data.Dataset.map will throw first with an internal error, so we do
    # not check this case explicitly.
    first_tensor = flat_args[0]
    first_tensor_shape = tf.shape(first_tensor)
    first_tensor_batch_size = first_tensor_shape[0]
    difference = batch_size - first_tensor_batch_size

    for i, tensor in enumerate(flat_args):
      control_deps = []
      if i != 0:
        # Check that leading dimensions of this tensor matches the first,
        # either statically or dynamically. (If the first dimensions of both
        # tensors are statically known, the we have to check the static
        # shapes at graph construction time or else we will never get to the
        # dynamic assertion.)
        if (first_tensor.shape[:1].is_fully_defined() and
            tensor.shape[:1].is_fully_defined()):
          if first_tensor.shape[0] != tensor.shape[0]:
            raise ValueError(
                'Batch size of dataset tensors does not match. %s '
                'has shape %s, but %s has shape %s'
                % (
                    first_tensor.name,
                    first_tensor.shape,
                    tensor.name,
                    tensor.shape,
                )
            )
        else:
          curr_shape = tf.shape(tensor)
          control_deps = [
              tf.Assert(
                  tf.equal(curr_shape[0], first_tensor_batch_size),
                  [
                      'Batch size of dataset tensors %s and %s do not match. '
                      'Shapes are' % (tensor.name, first_tensor.name),
                      curr_shape,
                      first_tensor_shape,
                  ],
              )
          ]

      with tf.control_dependencies(control_deps):
        # Pad to batch_size along leading dimension.
        flat_args[i] = tf.pad(
            tensor, [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1)
        )
      flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

    return tf.nest.pack_sequence_as(args, flat_args)

  return dataset.map(_pad_to_batch)

# def build_input_fn_for_builder(
#     builder,
#     is_training,
#     cache_dataset=False,
#     image_size=224,
#     rotation_range=0,
#     color_jitter_strength=1.0,
#     options=data_util.DistortionOptions(),
# ):
#   """Build input function for TFDS builder.

#   Args:
#     builder: TFDS builder for specified dataset.
#     is_training: (bool) Whether to build in training mode.
#     cache_dataset: (bool) whether to cache the entire dataset in memory.
#     image_size: (int) input image size, assumes image is square.
#     rotation_range: If 0 no rotation, for x, rotation in range (-x, x) degree.
#     color_jitter_strength: (float) The strength of color jittering.
#     options: Distortion Options, used to keep track of data augmentation options

#   Returns:
#     A function that accepts a dict of params and returns a tuple of images and
#     features, to be used as the input_fn in TPUEstimator. Params must include
#     batch_size.
#   """

#   def _input_fn(params):
#     """Generates TF Dataset from `params`."""
#     preprocess_fn_pretrain = get_preprocess_fn(
#         is_training,
#         is_pretrain=True,
#         image_size=image_size,
#         color_jitter_strength=color_jitter_strength,
#         options=options,
#     )
#     preprocess_fn_finetune = get_preprocess_fn(
#         is_training,
#         is_pretrain=False,
#         image_size=image_size,
#         rotation_range=rotation_range,
#         color_jitter_strength=color_jitter_strength,
#         options=options,
#     )
#     num_classes = builder.info.features['label'].num_classes

#     def _map_fn(image, label):
#       """Produces multiple transformations of the same batch."""
#       if FLAGS.train_mode == 'pretrain':
#         xs = []
#         for _ in range(2):  # Two transformations
#           xs.append(preprocess_fn_pretrain(image))
#         image = tf.concat(xs, -1)
#         label = tf.zeros([num_classes])
#       else:
#         image = preprocess_fn_finetune(image)
#         label = tf.one_hot(label, num_classes)
#       return image, label, 1.0

#     dataset = builder.as_dataset(
#         split=FLAGS.train_split if is_training else FLAGS.eval_split,
#         shuffle_files=is_training,
#         as_supervised=True,
#     )
#     if cache_dataset:
#       dataset = dataset.cache()
#     if is_training:
#       # Number of random elements to be shuffled in the batch.
#       buffer_multiplier = 50 if image_size <= 32 else 10
#       dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
#       # Repeat to iterate infinitely
#       dataset = dataset.repeat(count=-1)
#     dataset = dataset.map(
#         _map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )
#     dataset = dataset.batch(params['batch_size'], drop_remainder=is_training)
#     dataset = pad_to_batch(dataset, params['batch_size'])
#     images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()

#     return images, {'labels': labels, 'mask': mask}

#   return _input_fn

def build_input_fn_for_mimic(
    loader,
    is_training,
    cache_dataset=False,
    image_size=224,
    rotation_range=0,
    color_jitter_strength=1.0,
    options=data_util.DistortionOptions(),
):
  """Build input function for mimic_cxr.

  Args:
    is_training: (bool) Whether to build in training mode.
    cache_dataset: (bool) whether to cache the entire dataset in memory.
    image_size: (int) input image size, assumes image is square.
    rotation_range: If 0 no rotation, for x, rotation in range (-x, x) degree.
    color_jitter_strength: (float) The strength of color jittering.
    options: Distortion Options, used to keep track of data augmentation options

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator. Params must include
    batch_size.
  """

  def _input_fn(params):
    """Generates TF Dataset from `params`."""
    preprocess_fn_pretrain = get_preprocess_fn(
        is_training,
        is_pretrain=True,
        image_size=image_size,
        color_jitter_strength=color_jitter_strength,
        options=options,
    )
    preprocess_fn_finetune = get_preprocess_fn(
        is_training,
        is_pretrain=False,
        image_size=image_size,
        rotation_range=rotation_range,
        color_jitter_strength=color_jitter_strength,
        options=options,
    )

    train_tfds, val_tfds, test_tfds = loader.load()
    num_classes = loader.metadata['num_classes']

    def _map_fn(image, label, info=None):
      print('___________in _map_fn')
      print('___________image', image)
      print('__________label', label)
      print('__________info', info)
      """Produces multiple transformations of the same batch."""
      if FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
        label = tf.zeros([num_classes]) # self-supervision requires no label
      else:
        image = preprocess_fn_finetune(image)
        # indices = tf.argmax(label, axis=1)
        # label = tf.one_hot(indices, num_classes) # MIMIC Dataset labels are already multi-hot
        label = tf.zeros([num_classes])
        
      return image, label, 1.0

    if is_training:
      dataset_split = FLAGS.train_split
    else:
      dataset_split = FLAGS.eval_split
    
    tf.logging.info('_____using dataset: %s', dataset_split)
    if dataset_split == 'validation':
      dataset = val_tfds
    elif dataset_split == 'train':
      dataset = train_tfds
    elif dataset_split == 'test':
      dataset = test_tfds
    else:
      assert False, "invalid dataset_split"
    
    if cache_dataset:
      dataset = dataset.cache()
    if is_training:
      # Number of random elements to be shuffled in the batch.
      buffer_multiplier = 50 if image_size <= 32 else 2
      dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
      print('__________dataset before _map', dataset)
      # Repeat to iterate infinitely
      # dataset = dataset.repeat(count=-1)
    
    dataset = dataset.map(
        _map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    print('__________dataset after _map', dataset)
    dataset = dataset.batch(params['batch_size'], drop_remainder=is_training)
    print('__________dataset after batching', dataset)
    dataset = pad_to_batch(dataset, params['batch_size'])
    images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()

    return images, {'labels': labels, 'mask': mask}

  return _input_fn

def get_preprocess_fn(
    is_training,
    is_pretrain,
    image_size=224,
    rotation_range=0,
    color_jitter_strength=1.0,
    crop=True,
    options=data_util.DistortionOptions(),
):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if image_size <= 32:
    test_crop = False
  else:
    test_crop = crop
  return functools.partial(
      data_util.preprocess_image,
      height=image_size,
      width=image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop,
      rotation_range=rotation_range,
      color_jitter_strength=color_jitter_strength,
      options=options,
  )
