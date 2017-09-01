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
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos.")
tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")

tf.logging.set_verbosity(tf.logging.INFO)


def _setup_learning_rate(config, global_step):
  """Sets up the learning rate with optional exponential decay.

  Args:
    config: Object containing learning rate configuration parameters.
    global_step: Tensor; the global step.

  Returns:
    learning_rate: Tensor; the learning rate with exponential decay.
  """
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(config.learning_rate),
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=False)
  else:
    learning_rate = tf.constant(config.learning_rate)
  return learning_rate


def main(unused_argv):
  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.train_dir:
    raise ValueError("--train_dir is required.")

  model_config = configuration.model_config(
      input_file_pattern=FLAGS.input_file_pattern, bidirectional_encoder=True)
  training_config = configuration.training_config()

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="train")
    model.build()
     
    encoder_variables = [v for v in tf.global_variables()
                    if v.name.startswith("encoder") and "Adam" not in v.name]
    embedding_variables = [v for v in tf.global_variables()
                           if v.name.startswith("word_embedding") and "Adam" not in v.name]
    print([v.name for v in (encoder_variables+embedding_variables)])

    learning_rate = _setup_learning_rate(training_config, model.global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    
    encoder_mult = 0.1
    embedding_mult = 0.01
    multiply = dict([(v, encoder_mult) for v in encoder_variables] + [(v, embedding_mult) for v in embedding_variables])

    train_tensor = tf.contrib.slim.learning.create_train_op(
        total_loss=model.total_loss,
        optimizer=optimizer,
        gradient_multipliers=multiply,
        global_step=model.global_step,
        clip_gradient_norm=training_config.clip_gradient_norm)

    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    
    pretrain_saver = tf.train.Saver(encoder_variables+embedding_variables)

  print(model_path)
  if model_path:
    def restore_fn(sess):
       tf.logging.info(
      "Restoring SA&T variables from checkpoint file")
       saver.restore(sess, model_path)
  else:
    def restore_fn(sess):
      tf.logging.info(
        "Restoring SA&T variables from pretrained model")
      #saver.restore(sess, "/home/ubuntu/code/A_skip_thoughts_2/skip_thoughts/model/backup/run1/model.ckpt-2111")
      pretrain_saver.restore(sess, "/home/ubuntu/code/pretrained/bi/model.ckpt-500008")
  
  tf.contrib.slim.learning.train(
      train_op=train_tensor,
      logdir=FLAGS.train_dir,
      graph=g,
      global_step=model.global_step,
      number_of_steps=training_config.number_of_steps,
      save_summaries_secs=training_config.save_summaries_secs,
      saver=saver,
      save_interval_secs=training_config.save_model_secs,
      init_fn = restore_fn)


if __name__ == "__main__":
  tf.app.run()
