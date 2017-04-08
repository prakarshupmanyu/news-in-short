import os
from six.moves import cPickle as pickle
from six.moves import xrange
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K

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


###################################### Model ###########################################

random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

################################### RNN Model ##########################################

"""
Starting with stacked LSTM; identical to the bottom layer in training model
"""

rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_size,
	input_length=maxlen,
	W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
	name='embedding_1'))

for i in range(rnn_layers):
	lstm = LSTM(rnn_size, return_sequences=True,
		W_regularizer=regularizer, U_regularizer=regularizer,
		b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
		name='lstm_%d'%(i+1)
		)
	rnn_model.add(lstm)
	rnn_model.add(Dropout(p_dense, name='dropout_%d'%(i+1)))

