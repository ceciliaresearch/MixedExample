"""Functions for doing mixed-example data augmentation."""

import numpy as np
import tensorflow as tf


def should_zero_mean(method_name):
  return 'bcplus' in method_name


# Each of the methods below takes in a [2, HEIGHT, WIDTH, CHANNELS] tensor of
# images to be mixed together, and a [2, CLASSES] tensor of labels.

def first_example(images, labels):
  return images[0], labels[0]


def mixup(images, labels):
  mix_coeff = tf.random_uniform([])
  image = mix_coeff * images[0] + (1 - mix_coeff) * images[1]
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def bcplus(images, labels):
  mix_coeff = tf.random_uniform([])
  mean0, var0 = tf.nn.moments(images[0], axes=None)
  std0 = tf.sqrt(var0)
  mean1, var1 = tf.nn.moments(images[1], axes=None)
  std1 = tf.sqrt(var1)
  p = 1. / (1. + (std0 / std1) * (1. - mix_coeff) / mix_coeff)
  image = (p * images[0] + (1. - p) * images[1]) / tf.sqrt(
      tf.pow(p, 2) + tf.pow(1-p, 2))
  label = mix_coeff * labels[0] + (1. - mix_coeff) * labels[1]
  return (image, label)


def vert_concat(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  dim_H = tf.random_uniform([], 0, H, dtype=tf.int32)
  lambda_H = tf.cast(dim_H, tf.float32) / tf.constant(H, dtype=tf.float32)
  image = tf.concat([images[0][:dim_H, :, :],
                     images[1][dim_H:, :, :]], 0)
  label = lambda_H * labels[0] + (1 - lambda_H) * labels[1]
  return (image, label)


def horiz_concat(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  dim_W = tf.random_uniform([], 0, W, dtype=tf.int32)
  lambda_W = tf.cast(dim_W, tf.float32) / tf.constant(W, dtype=tf.float32)
  image = tf.concat([images[0][:, :dim_W, :],
                     images[1][:, dim_W:, :]], 1)
  label = lambda_W * labels[0] + (1 - lambda_W) * labels[1]
  return (image, label)


def mixed_concat(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  dim_W = tf.random_uniform([], 0, W, dtype=tf.int32)
  lambda_W = tf.cast(dim_W, tf.float32) / tf.constant(W, dtype=tf.float32)
  image1 = tf.concat([images[0][:dim_W, :, :],
                      images[1][dim_W:, :, :]], 0)
  image2 = tf.concat([images[1][:dim_W, :, :],
                      images[0][dim_W:, :, :]], 0)
  dim_H = tf.random_uniform([], 0, H, dtype=tf.int32)
  lambda_H = tf.cast(dim_H, tf.float32) / tf.constant(H, dtype=tf.float32)
  image = tf.concat([image1[:, :dim_H, :],
                     image2[:, dim_H:, :]], 1)
  mix_coeff0 = lambda_H * lambda_W + (1. - lambda_H) * (1. - lambda_W)
  label = mix_coeff0 * labels[0] + (1. - mix_coeff0) * labels[1]
  return (image, label)


def random_2x2(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  p = 0.5
  h_buff = int(round(.5 * p * H))
  w_buff = int(round(.5 * p * W))
  dim_H = tf.random_uniform([], h_buff, H - h_buff, dtype=tf.int32)
  dim_W = tf.random_uniform([], w_buff, W - w_buff, dtype=tf.int32)
  lambda_H = dim_H / H
  lambda_W = dim_W / W
  rand_vals = tf.to_float(tf.random_uniform([4]) < tf.random_uniform([]))
  mask =  tf.concat([
    tf.concat([tf.fill([dim_H, dim_W], rand_vals[0]),
               tf.fill([dim_H, W-dim_W], rand_vals[1])], axis=1),
    tf.concat([tf.fill([H-dim_H, dim_W], rand_vals[2]),
               tf.fill([H-dim_H, W-dim_W], rand_vals[3])], axis=1)],
    axis=0)
  mask = tf.expand_dims(mask, 2)
  image = mask * images[0] + (1. - mask) * images[1]
  merge_ratio = tf.reduce_sum(mask) / (W * H)
  label = merge_ratio * labels[0] + (1. - merge_ratio) * labels[1]
  return (image, label)


def vh_mixup(images, labels):
  image1, label1 = vert_concat(images, labels)
  image2, label2 = horiz_concat(images, labels)
  return mixup(tf.stack([image1, image2]), tf.stack([label1, label2]))


def vh_bcplus(images, labels):
  image1, label1 = vert_concat(images, labels)
  image2, label2 = horiz_concat(images, labels)
  return bcplus(tf.stack([image1, image2]), tf.stack([label1, label2]))


def random_square(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  def random_square_np(images, labels):
    square_length = 16
    center_r = np.random.randint(0, H)
    center_c = np.random.randint(0, W)
    r1 = int(max(center_r - square_length / 2, 0))
    r2 = int(min(center_r + square_length / 2, H))
    c1 = int(max(center_c - square_length / 2, 0))
    c2 = int(min(center_c + square_length / 2, W))
    image = images[0].copy()
    square_frac = float(r2 - r1) * (c2 - c1) / (H * W)
    r1_2 = np.random.randint(0, H - (r2 - r1))
    r2_2 = r1_2 + (r2 - r1)
    c1_2 = np.random.randint(0, W - (c2 - c1))
    c2_2 = c1_2 + (c2 - c1)
    image[r1:r2,c1:c2,:] = images[1][r1_2:r2_2,c1_2:c2_2,:]
    label = (1 - square_frac) * labels[0] + square_frac * labels[1]
    return image, label
  (image, label) = tf.py_func(random_square_np, [images, labels],
      [tf.float32, tf.float32], name='random_square')
  return (image, label)


def random_column_interval(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  ind1 = tf.random_uniform([], 0, W, dtype=tf.int32)
  ind2 = tf.random_uniform([], ind1, W, dtype=tf.int32)
  length = ind2 - ind1
  ind3 = tf.random_uniform([], 0, W-length, dtype=tf.int32)
  ind4 = ind3 + length
  image = tf.concat([images[0][:, :ind1, :],
                     images[1][:, ind3:ind4, :],
                     images[0][:, ind2:, :]], axis=1)
  mix_coeff = tf.to_float(W - length) / tf.to_float(W)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return image, label


def random_row_interval(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  ind1 = tf.random_uniform([], 0, H, dtype=tf.int32)
  ind2 = tf.random_uniform([], ind1, H, dtype=tf.int32)
  length = ind2 - ind1
  ind3 = tf.random_uniform([], 0, H-length, dtype=tf.int32)
  ind4 = ind3 + length
  image = tf.concat([images[0][:ind1, :, :],
                     images[1][ind3:ind4, :, :],
                     images[0][ind2:, :, :]], axis=0)
  mix_coeff = tf.to_float(H - length) / tf.to_float(H)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return image, label


def random_rows(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  mask = tf.to_float(tf.random_uniform([H, 1, 1]) < tf.random_uniform([]))
  image = mask * images[0] + (1 - mask) * images[1]
  mix_coeff = tf.reduce_mean(mask)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def random_cols(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  mask = tf.to_float(tf.random_uniform([1, W, 1]) < tf.random_uniform([]))
  image = mask * images[0] + (1 - mask) * images[1]
  mix_coeff = tf.reduce_mean(mask)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def random_pixels(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  mask = tf.to_float(tf.random_uniform([H, W, 1]) < tf.random_uniform([]))
  image = mask * images[0] + (1 - mask) * images[1]
  mix_coeff = tf.reduce_mean(mask)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def random_elems(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  mask = tf.to_float(tf.random_uniform([H, W, C]) < tf.random_uniform([]))
  image = mask * images[0] + (1 - mask) * images[1]
  mix_coeff = tf.reduce_mean(mask)
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def noisy_mixup(images, labels):
  (_, H, W, C) = images.get_shape().as_list()
  mix_coeff = tf.random_uniform([])
  mix_mat = tf.random_normal([W, H, 1], mix_coeff, 0.025)
  mix_mat = tf.maximum(tf.minimum(mix_mat, 1.), 0.)
  image = mix_mat * images[0] + (1 - mix_mat) * images[1]
  label = mix_coeff * labels[0] + (1 - mix_coeff) * labels[1]
  return (image, label)


def mixed_example_data_augmentation(images, labels, method_name):
  return {
      'baseline': first_example,
      'mixup': mixup,
      'bcplus': bcplus,
      'vert_concat': vert_concat,
      'horizontal_concat': horiz_concat,
      'mixed_concat': mixed_concat,
      'random_2x2': random_2x2,
      'vh_mixup': vh_mixup,
      'vh_bcplus': vh_bcplus,
      'random_square': random_square,
      'random_column_interval': random_column_interval,
      'random_row_interval': random_row_interval,
      'random_rows': random_rows,
      'random_columns': random_cols,
      'random_pixels': random_pixels,
      'random_elements': random_elems,
      'noisy_mixup': noisy_mixup
  }[method_name](images, labels)
