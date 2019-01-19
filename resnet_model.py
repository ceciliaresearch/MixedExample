# Based on an older version of CIFAR-10 resnet training code from the
# tensorflow/models github repository. Modified to use the ResNet-18
# architecture.
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
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=0.99, epsilon=1e-5, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def preactivation_block(inputs, num_filters, stride, data_format, is_training):
  input_channels = inputs.get_shape().as_list()[1]

  net = batch_norm_relu(inputs, is_training, data_format)

  residual = conv2d_fixed_padding(
      inputs=net, filters=num_filters, kernel_size=3, strides=stride,
      data_format=data_format)
  residual = batch_norm_relu(residual, is_training, data_format)
  residual = conv2d_fixed_padding(
      inputs=residual, filters=num_filters, kernel_size=3, strides=1,
      data_format=data_format)

  shortcut = inputs
  # Correct for difference in channels.
  if stride != 1 or input_channels != num_filters:
    shortcut = conv2d_fixed_padding(
        inputs=net, filters=num_filters, kernel_size=1, strides=stride,
        data_format=data_format)
  return shortcut + residual


def resnet18(inputs, num_classes, is_training=False):
  data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=64, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')

  block_num_filters = [64, 128, 256, 512]
  for block_num, num_filters in enumerate(block_num_filters):
    inputs = preactivation_block(
        inputs, num_filters=num_filters, stride=(1 + (block_num != 0)),
        data_format=data_format, is_training=is_training)
    inputs = preactivation_block(
        inputs, num_filters=num_filters, stride=1,
        data_format=data_format, is_training=is_training)

  pool_kernel_size = inputs.get_shape().as_list()[2]
  inputs = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=pool_kernel_size, strides=1, padding='valid',
      data_format=data_format)
  inputs = tf.layers.flatten(inputs)
  inputs = tf.layers.dense(inputs=inputs, units=num_classes)
  return inputs
