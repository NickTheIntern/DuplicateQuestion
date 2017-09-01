from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import pickle
# Set paths to the model.
VOCAB_FILE = "./unexpanded/vocab.txt"
EMBEDDING_MATRIX_FILE = "./initial/embeddings.npy"
CHECKPOINT_PATH = "../pretrained/bi/model.ckpt-500008"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_FILE = "../QuoraData/train_split2_V1.pickle"

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder = True),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)


# Load the movie review dataset.

f = open(MR_DATA_FILE, 'rb')
(data1, data2, label) = pickle.load(f)


# Generate Skip-Thought Vectors for each sentence in the dataset.
encodings = encoder.encode((data1, data2))
np.save("./s1s2_initial_encoding_split2", encodings)

