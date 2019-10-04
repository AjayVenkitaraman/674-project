# Copyright 2019 The Magenta Authors.
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

"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from magenta.models.sketch_rnn import rnn
import numpy as np
import tensorflow as tf


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      data_set=['sketchrnn_cat.npz','sketchrnn_dog.npz','sketchrnn_bear.npz','sketchrnn_airplane.npz',
                'sketchrnn_ant.npz','sketchrnn_banana.npz','sketchrnn_bench.npz','sketchrnn_book.npz',
                'sketchrnn_bottlecap.npz','sketchrnn_bread.npz'],  # Our dataset.
      # data_set=['aaron_sheep/aaron_sheep.npz','kanji/short_kanji.npz','omniglot/omniglot.npz'],
      num_steps=10000,  # Total number of steps of training. Keep large.
      save_every=50,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=256,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True,  # Is model training? Recommend keeping true.
      loss_function='softmax', # Loss function being used for classification.
      num_classes = 10 # Number of classes predictions.
  )
  return hparams


class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')
    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    
    # Removed the decoder part from the actual sketchrnn code 
    # and just returning last_h
    return last_h

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = self.hps.use_recurrent_dropout
    use_input_dropout = self.hps.use_input_dropout
    use_output_dropout = self.hps.use_output_dropout

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])
    self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
    self.y_labels = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])
    print("self.y_labels.shape = ",self.y_labels.shape)
    # The target/expected vectors of strokes
    self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]
    
    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.batch_z = self.encoder(self.output_x, self.sequence_lengths)
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros((self.hps.batch_size, self.hps.z_size), dtype=tf.float32)


    # TODO(deck): Better understand this comment.
    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    n_out = self.hps.num_classes #num_classes

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [2*self.hps.enc_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    output = tf.nn.xw_plus_b(self.batch_z, output_w, output_b)
    self.output = output
    if self.y_labels is not None:
      self.ce_loss = self.lossfunctions(self.hps.loss_function)
    else:
      self.ce_loss = 0
    if self.hps.is_training:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      self.cost = self.ce_loss

      gvs = optimizer.compute_gradients(self.cost)
      g = self.hps.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

  def lossfunctions(self, lossfn):
    if lossfn == 'softmax':
      return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.output,
          labels=self.y_labels
          )
        )
    # elif lossfn == 'sigmoid':
    #   loss_val = tf.constant(0.0)
    #   for i in range(self.hps.num_classes):
    #     loss_val = tf.add(loss_val,tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(
    #       _sentinel=None,
    #       labels=tf.cast(tf.reshape(self.y_labels,[self.y_labels.shape[0],1]),tf.float32),
    #       logits=tf.transpose(tf.gather_nd(
    #         tf.transpose(self.output),
    #         [[i]],
    #         name=None
    #         )
    #       ),
    #       name=None
    #       )
    #     )
    #    )
    #   return loss_val/self.hps.num_classes
    # elif lossfn == 'weighted':
    #   loss_val = tf.constant(0.0)
    #   for i in range(self.hps.num_classes):
    #     loss_val = tf.add(loss_val,tf.reduce_mean(
    #     tf.nn.weighted_cross_entropy_with_logits(
    #       targets=tf.cast(tf.reshape(self.y_labels,[self.y_labels.shape[0],1]),tf.float32),
    #       logits=tf.transpose(tf.gather_nd(
    #         tf.transpose(self.output),
    #         [[i]],
    #         name=None
    #         )
    #       ),
    #       pos_weight=tf.constant(0.9)
    #       )
    #     )
    #     )
    #   return loss_val
    # else:
    #   assert False, 'Please choose from the following lossfunctions:\n \
    #   1. softmax \n 2. sigmoid \n 3. weighted \n'

