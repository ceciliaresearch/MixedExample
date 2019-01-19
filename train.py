# Based on an older version of CIFAR-10 resnet training code from the
# tensorflow/models github repository.
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import os
import sys
import tensorflow as tf

import mixed_example
import resnet_model


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100'],
                    help='Which dataset to use. Datasets must be downloaded '
                         'before they can be used.')

parser.add_argument('--model_dir', type=str, default='cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=225,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay, implemented as L2 regularization. '
                         'Remember to use 5e-4 for the baseline or CIFAR-100.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--mixed_example_method', type=str, default='vh_mixup',
                    help='Name of mixed-example method to use.')


# Image dimensions for CIFAR.
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def get_num_classes():
  return {'cifar10': 10, 'cifar100': 100}[FLAGS.dataset]


def record_dataset(is_training):
  """Returns a tf.data.Dataset for CIFAR-10/100."""
  if FLAGS.dataset == 'cifar10':
    data_dir = os.path.join('cifar10_data', 'cifar-10-batches-bin')
    if not os.path.exists(data_dir):
      print('Can\'t find data at %s. Make sure it is downloaded.' % data_dir)
      exit()
    if is_training:
      filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                   for i in range(1, 6)]
    else:
      filenames = [os.path.join(data_dir, 'test_batch.bin')]
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  elif FLAGS.dataset == 'cifar100':
    data_dir = os.path.join('cifar100_data', 'cifar-100-binary')
    if not os.path.exists(data_dir):
      print('Can\'t find data at %s. Make sure it is downloaded.' % data_dir)
      exit()
    if is_training:
      filenames = [os.path.join(data_dir, 'train.bin')]
    else:
      filenames = [os.path.join(data_dir, 'test.bin')]
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 2
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def parse_record(raw_record):
  """Parses a CIFAR-10/100 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = {'cifar10': 1, 'cifar100': 1}[FLAGS.dataset]
  label_offset = {'cifar10': 0, 'cifar100': 1}[FLAGS.dataset]
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + label_offset + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[label_offset], tf.int32)
  label = tf.one_hot(label, get_num_classes())

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes+label_offset:record_bytes],
      [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  return image, label


def preprocess_image1(image, is_training):
  """Preprocesses a single image of layout [height, width, depth].

  We do not do any type of random data augmentation in this step so that the
  dataset can be cached, which speeds up training.
  """
  # Goal: uint8 -> float, normalized by std
  if mixed_example.should_zero_mean(FLAGS.mixed_example_method):
    # Certain methods remove a per-image mean first, which also causes a
    # different dataset mean and standard deviation.
    mean = tf.reduce_mean(image)
    image = image - mean
    mean_image = tf.constant([0.01803922,  0.00878431, -0.02682353],
        dtype=tf.float32)
    std_image = tf.constant([0.21921569, 0.21058824, 0.22156863],
        dtype=tf.float32)
    image = tf.cast(image, tf.float32) / 255 - mean_image
    image = image / std_image
  else:
    mean_image = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
    std_image = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
    image = tf.cast(image, tf.float32) / 255 - mean_image
    image = image / std_image

  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
	image, _HEIGHT + 8, _WIDTH + 8)
  return image


def preprocess_image2(image, is_training):
  """Preprocesses a single image of layout [height, width, depth].

  This function takes care of any random data augmentation.
  """
  if is_training:
    # Random crop and flip.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)
  return image


def input_fn(is_training, batch_size, num_epochs=1):
  """Input function for CIFAR-10 or CIFAR-100."""
  dataset = record_dataset(is_training)
  dataset = dataset.map(parse_record)

  # Split the preprocessing in two steps: those without any randomness (i.e.
  # no data augmentation), and those with randomness. We can cache the dataset
  # in between to significantly speed up training.
  dataset = dataset.map(
      lambda image, label: (preprocess_image1(image, is_training), label))
  dataset = dataset.cache()

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    dataset = dataset.map(
        lambda image, label: (preprocess_image2(image, is_training), label))
    # Apply mixed-example data augmentation, two images at a time.
    dataset = dataset.batch(2, drop_remainder=True)
    dataset = dataset.map(functools.partial(
      mixed_example.mixed_example_data_augmentation,
      method_name=FLAGS.mixed_example_method))

  dataset = dataset.repeat(2 * num_epochs if is_training else num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def cifar_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10 and CIFAR-100."""
  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = resnet_model.resnet18(
      inputs, get_num_classes(),
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
  cross_entropy = tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  # Use the L2-regularization form of weight decay.
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  # Metrics
  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}
  accuracy = tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy)

  # Set up the Optimizer.
  train_op = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Our learning rate scheduled is based on a batch size of 128. Adjust to
    # the given batch size using a simple scaling heuristic.
    base_batch = 128.
    lr_vals = [x * (FLAGS.batch_size / base_batch)
        for x in [.01, .1, .01, .001, .0001]]
    lr_boundaries = [int(x * base_batch) for x in [400, 32000, 48000, 70000]]
    learning_rate = tf.train.piecewise_constant(
	tf.train.get_or_create_global_step() * FLAGS.batch_size,
	boundaries=lr_boundaries,
	values=lr_vals,
	name='learning_rate')
    # Note: If a batch size other than 128 is used, momentum might need to be
    # adjusted.
    optimizer = tf.train.MomentumOptimizer(
	learning_rate=learning_rate,
	momentum=0.9)
    tf.summary.scalar('learning_rate', learning_rate)

    # Batch norm requires update ops to be added as a dependency to the
    # train_op.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  if os.path.exists(FLAGS.model_dir):
    print('model_dir already exists, trying to resume...')

  cifar_estimator = tf.estimator.Estimator(
      model_fn=cifar_model_fn, model_dir=FLAGS.model_dir,
      config=tf.estimator.RunConfig(save_checkpoints_secs=1e9))

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    cifar_estimator.train(
        input_fn=lambda: input_fn(
            True, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])

    # Evaluate the model and print results.
    eval_results = cifar_estimator.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.batch_size))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
