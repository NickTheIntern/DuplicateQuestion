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
"""Skip-Thoughts model for learning sentence vectors.

The model is based on the paper:

  "Skip-Thought Vectors"
  Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
  Antonio Torralba, Raquel Urtasun, Sanja Fidler.
  https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Layer normalization is applied based on the paper:

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  https://arxiv.org/abs/1607.06450
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from skip_thoughts.ops import gru_cell
from skip_thoughts.ops import input_ops


def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):  # pylint: disable=unused-argument
  """Variable initializer that produces a random orthonormal matrix."""
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u


class SkipThoughtsModel(object):
  """Skip-thoughts model."""

  def __init__(self, config, mode="train", input_reader=None):
    """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "encode".
      input_reader: Subclass of tf.ReaderBase for reading the input serialized
        tf.Example protocol buffers. Defaults to TFRecordReader.

    Raises:
      ValueError: If mode is invalid.
    """
    if mode not in ["train", "eval", "encode", "test"]:
      raise ValueError("Unrecognized mode: %s" % mode)

    self.config = config
    self.mode = mode
    self.reader = input_reader if input_reader else tf.TFRecordReader()

    # Initializer used for non-recurrent weights.
    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-self.config.uniform_init_scale,
        maxval=self.config.uniform_init_scale)


    # Each is an int64 Tensor with  shape [batch_size, padded_length].
    self.encode_ids1 = None
    self.encode_ids2 = None

    # Boolean masks distinguishing real words (1) from padded words (0).
    # Each is an int32 Tensor with shape [batch_size, padded_length].
    self.encode_mask1 = None
    self.encode_mask2 = None

    # Input sentences represented as sequences of word embeddings.
    # Each is a float32 Tensor with shape [batch_size, padded_length, emb_dim].
    self.encode_emb1 = None
    self.encode_emb2 = None

    # The output from the sentence encoder.
    # A float32 Tensor with shape [batch_size, num_gru_units].
    self.thought_vectors1 = None
    self.thought_vectors2 = None

    self.label = None
    self.feature = None
    # The cross entropy losses and corresponding weights of the decoders. Used
    # for evaluation.
    self.target_cross_entropy_losses = []
    self.accuracy = []
    self.logits = []

    # The total loss to optimize.
    self.total_loss = None

  def build_inputs(self):
    """Builds the ops for reading input data.

    Outputs:
      self.encode_ids
      self.encode_mask
      
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_ids1 = None
      encode_ids2 = None
      encode_mask1 = tf.placeholder(tf.int8, (None, None), name="encode_mask1")
      encode_mask2 = tf.placeholder(tf.int8, (None, None), name="encode_mask2")
      label = None

    elif self.mode == "test":
      encode_ids1 = None
      encode_ids2 = None
      encode_mask1 = tf.placeholder(tf.int8, (None, None), name="encode_mask1")
      encode_mask2 = tf.placeholder(tf.int8, (None, None), name="encode_mask2")
      label = None
      
    else:
      # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          shuffle=self.config.shuffle_input_data,
          capacity=self.config.input_queue_capacity,
          num_reader_threads=self.config.num_input_reader_threads)

      # Deserialize a batch.
      serialized = input_queue.dequeue_many(self.config.batch_size)
      s1, s2, label = input_ops.parse_example_batch(
          serialized)

      encode_ids1 = s1.ids
      encode_ids2 = s2.ids

      encode_mask1 = s1.mask
      encode_mask2 = s2.mask
      


    self.encode_ids1 = encode_ids1
    self.encode_ids2 = encode_ids2

    self.encode_mask1 = encode_mask1
    self.encode_mask2 = encode_mask2

    self.label = label

  def build_word_embeddings(self):
    """Builds the word embeddings.

    Inputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids

    Outputs:
      self.encode_emb
      self.decode_pre_emb
      self.decode_post_emb
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_emb1 = tf.placeholder(tf.float32, (
          None, None, self.config.word_embedding_dim), "encode_emb1")
      # No sequences to decode.
      encode_emb2 = tf.placeholder(tf.float32, (
          None, None, self.config.word_embedding_dim), "encode_emb2")
    elif self.mode == "test":
      encode_emb1 = tf.placeholder(tf.float32, (
          None, None, self.config.word_embedding_dim), "encode_emb1")
      # No sequences to decode.
      encode_emb2 = tf.placeholder(tf.float32, (
          None, None, self.config.word_embedding_dim), "encode_emb2")
    else:
      word_emb = tf.get_variable(
          name="word_embedding",
          shape=[self.config.vocab_size, self.config.word_embedding_dim],
          initializer=self.uniform_initializer)

      encode_emb1 = tf.nn.embedding_lookup(word_emb, self.encode_ids1)
      encode_emb2 = tf.nn.embedding_lookup(word_emb, self.encode_ids2)


    self.encode_emb1 = encode_emb1
    self.encode_emb2 = encode_emb2


  def _initialize_gru_cell(self, num_units):
    """Initializes a GRU cell.

    The Variables of the GRU cell are initialized in a way that exactly matches
    the skip-thoughts paper: recurrent weights are initialized from random
    orthonormal matrices and non-recurrent weights are initialized from random
    uniform matrices.

    Args:
      num_units: Number of output units.

    Returns:
      cell: An instance of RNNCell with variable initializers that match the
        skip-thoughts paper.
    """
    return gru_cell.LayerNormGRUCell(
        num_units,
        w_initializer=self.uniform_initializer,
        u_initializer=random_orthonormal_initializer,
        b_initializer=tf.constant_initializer(0.0))

  def build_encoder(self):
    """Builds the sentence encoder.

    Inputs:
      self.encode_emb
      self.encode_mask

    Outputs:
      self.thought_vectors

    Raises:
      ValueError: if config.bidirectional_encoder is True and config.encoder_dim
        is odd.
    """
    with tf.variable_scope("encoder") as scope:
      length1 = tf.to_int32(tf.reduce_sum(self.encode_mask1, 1), name="length1")

      if self.config.bidirectional_encoder:
        if self.config.encoder_dim % 2:
          raise ValueError(
              "encoder_dim must be even when using a bidirectional encoder.")
        num_units = self.config.encoder_dim // 2
        cell_fw = self._initialize_gru_cell(num_units)  # Forward encoder
        cell_bw = self._initialize_gru_cell(num_units)  # Backward encoder
        _, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=self.encode_emb1,
            sequence_length=length1,
            dtype=tf.float32,
            scope=scope)
        thought_vectors1 = tf.concat(states, 1, name="thought_vectors1")
      else:
        cell = self._initialize_gru_cell(self.config.encoder_dim)
        _, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.encode_emb1,
            sequence_length=length1,
            dtype=tf.float32,
            scope=scope)
        # Use an identity operation to name the Tensor in the Graph.
        thought_vectors1 = tf.identity(state, name="thought_vectors1")
       
      scope.reuse_variables()

      length2 = tf.to_int32(tf.reduce_sum(self.encode_mask2, 1), name="length2")

      if self.config.bidirectional_encoder:
        if self.config.encoder_dim % 2:
          raise ValueError(
            "encoder_dim must be even when using a bidirectional encoder.")
        num_units = self.config.encoder_dim // 2
        cell_fw = self._initialize_gru_cell(num_units)  # Forward encoder
        cell_bw = self._initialize_gru_cell(num_units)  # Backward encoder
        _, states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=self.encode_emb2,
          sequence_length=length2,
          dtype=tf.float32,
          scope=scope)
        thought_vectors2 = tf.concat(states, 1, name="thought_vectors2")
      else:
        cell = self._initialize_gru_cell(self.config.encoder_dim)
        _, state = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=self.encode_emb2,
          sequence_length=length2,
          dtype=tf.float32,
          scope=scope)
        # Use an identity operation to name the Tensor in the Graph.
        thought_vectors2 = tf.identity(state, name="thought_vectors2")

    self.thought_vectors1 = thought_vectors1
    self.thought_vectors2 = thought_vectors2

  def build_fc(self):
    """Builds the sentence decoders.

    Inputs:
      self.decode_pre_emb
      self.decode_post_emb
      self.decode_pre_ids
      self.decode_post_ids
      self.decode_pre_mask
      self.decode_post_mask
      self.thought_vectors

    Outputs:
      self.target_cross_entropy_losses
      self.target_cross_entropy_loss_weights
    """

    joint_thought = tf.nn.relu(tf.concat([self.thought_vectors1, self.thought_vectors2, tf.multiply(self.thought_vectors1, self.thought_vectors2), 
            tf.square(tf.subtract(self.thought_vectors1, self.thought_vectors2))], 1))
    if self.mode == "encode":
      return
    with tf.variable_scope("fc1") as scope:
      fc1 = tf.contrib.layers.fully_connected(
          inputs=joint_thought,
          num_outputs=512,
          weights_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
          weights_initializer=self.uniform_initializer,
          scope=scope,
          biases_initializer= self.uniform_initializer)
      self.feature = tf.identity(fc1, name="feature")

    if self.mode != "encode":
      with tf.variable_scope("fc2") as scope:
        logits = tf.contrib.layers.fully_connected(
          inputs = self.feature,
          num_outputs=2,
          activation_fn=None,
          weights_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
          weights_initializer=self.uniform_initializer,
          scope = scope,
          biases_initializer= self.uniform_initializer)
      self.logits.append(tf.nn.softmax(logits, name="logits"))
        
      #try more fully connected layers
      losses = tf.nn.softmax_cross_entropy_with_logits(
      labels=self.label, logits=logits)
      batch_loss = tf.reduce_sum(losses)
      tf.losses.add_loss(batch_loss)

      tf.summary.scalar("losses", batch_loss)

      self.target_cross_entropy_losses.append(losses)

      self.prediction = tf.argmax(logits,1)
      
      correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(self.label,1))
      
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
      self.accuracy.append(accuracy)

  def build_loss(self):
    """Builds the loss Tensor.

    Outputs:
      self.total_loss
    """
    if self.mode != "encode":
      total_loss = tf.losses.get_total_loss()
      tf.summary.scalar("losses/total", total_loss)

      self.total_loss = total_loss

  def build_global_step(self):
    """Builds the global step Tensor.

    Outputs:
      self.global_step
    """
    self.global_step = tf.contrib.framework.create_global_step()

  def build(self):
    """Creates all ops for training, evaluation or encoding."""
    self.build_inputs()
    self.build_word_embeddings()
    self.build_encoder()
    self.build_fc()
    self.build_loss()
    self.build_global_step()
