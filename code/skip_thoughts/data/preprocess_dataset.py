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
"""Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os


import numpy as np
import pickle
import tensorflow as tf

from skip_thoughts.data import special_words

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("vocab_file", "",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 5000,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 50,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.flags.DEFINE_boolean("add_eos", True,
                        "Whether to add end-of-sentence ids to the output.")

tf.logging.set_verbosity(tf.logging.INFO)


def _build_vocabulary():
  """Loads or builds the model vocabulary.

  Args:
    input_files: List of pre-tokenized input .txt files.

  Returns:
    vocab: A dictionary of word to id.
  """
  if FLAGS.vocab_file:
    tf.logging.info("Loading existing vocab file.")
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
      for i, line in enumerate(f):
        word = line.decode("utf-8").strip()
        assert word not in vocab, "Attempting to add word twice: %s" % word
        vocab[word] = i
    tf.logging.info("Read vocab of size %d from %s",
                    len(vocab), FLAGS.vocab_file)
    return vocab

  assert False, "should use existing vocabulary" 

  return vocab


def _int64_feature(value):
  """Helper for creating an Int64 Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=[int(v) for v in value]))


def _sentence_to_ids(sentence, vocab):
  """Helper for converting a sentence (list of words) to a list of ids."""
  ids = [vocab.get(w, special_words.UNK_ID) for w in sentence]
  if FLAGS.add_eos:
    ids.append(special_words.EOS_ID)
  return ids


def _create_serialized_example(s1, s2, label, vocab):
  """Helper for creating a serialized Example proto."""
  example = tf.train.Example(features=tf.train.Features(feature={
      "s1": _int64_feature(_sentence_to_ids(s1, vocab)),
      "s2": _int64_feature(_sentence_to_ids(s2, vocab)),
      "label": _int64_feature(label),
  }))

  return example.SerializeToString()


def _process_input_file(filename, vocab, stats):
  """Processes the sentences in an input file.

  Args:
    filename: pickle file that stores s1(list), s2(list), label(np array in one hot encoding)
    vocab: A dictionary of word to id.
    stats: A Counter object for statistics.

  Returns:
    processed: A list of serialized Example protos
  """
  tf.logging.info("Processing input file: %s", filename)
  processed = []

  f = open(filename,'rb')
  sentence1List, sentence2List, labelArray = pickle.load(f)  


  for i,(sentence1,sentence2,label) in enumerate(zip(sentence1List, sentence2List, labelArray)):
    stats.update(["sentences_seen"])
    s1 = sentence1.split()
    s2 = sentence2.split()
    if(len(s2) > FLAGS.max_sentence_length):
      s2 = s2[0:(FLAGS.max_sentence_length-1)]
    if(len(s1) > FLAGS.max_sentence_length):
      s1 = s1[0:(FLAGS.max_sentence_length-1)]
    
    serialized = _create_serialized_example(s1, s2, label,
                                                vocab)
    processed.append(serialized)

  return processed


def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.python_io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
  tf.logging.info("Writing dataset %s", name)
  borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
  for i in range(num_shards):
    filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                   num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    _write_shard(filename, dataset, shard_indices)
    tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                    borders[i], borders[i + 1], filename)
  tf.logging.info("Finished writing %d sentences in dataset %s.",
                  len(indices), name)


def main(unused_argv):
  if not FLAGS.input_file:
    raise ValueError("--input_file is required.")
  if not FLAGS.output_dir:
    raise ValueError("--output_dir is required.")
  if not FLAGS.vocab_file:
    raise ValueError("--vocab_file is required.")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)


  vocab = _build_vocabulary()

  tf.logging.info("Generating dataset.")
  stats = collections.Counter()
  dataset = []

  dataset.extend(_process_input_file(FLAGS.input_file, vocab, stats))


  tf.logging.info("Generated dataset with %d sentences.", len(dataset))
  for k, v in stats.items():
    tf.logging.info("%s: %d", k, v)

  tf.logging.info("Shuffling dataset.")
  np.random.seed(123)
  shuffled_indices = np.random.permutation(len(dataset))
  val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
  train_indices = shuffled_indices[FLAGS.num_validation_sentences:]

  _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
  _write_dataset("validation", dataset, val_indices,
                 FLAGS.validation_output_shards)


if __name__ == "__main__":
  tf.app.run()
