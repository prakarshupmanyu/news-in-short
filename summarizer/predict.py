import os
from six.moves import cPickle as pickle
from six.moves import xrange
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

maxlend = 25
maxlenh = 25
maxlen = maxlend + maxlenh
rnn_size = 512 
rnn_layers = 3  
batch_norm = False

#Number of nodes from the top LSTM layer used for activation 
activation_rnn_size = 40 if maxlend else 0

#Training parameters
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
batch_size = 64

nb_train_samples = 30000
nb_val_samples = 3000


############################## Read Word embedding ####################################

with open('data/vocabulary-embedding.pkl', 'rb') as fp:
	embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

nb_unknown_words = 10

for i in range(nb_unknown_words):
	idx2word[vocab_size - 1 - i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i] + 

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'