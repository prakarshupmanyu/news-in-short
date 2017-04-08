import os
from six.moves import cPickle as pickle
from six.moves import xrange
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, SpatialDropout1D
from keras.layers import Merge, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
import h5py

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

with open('../DataExtractor/art/vocabEmbeddings.pkl', 'rb') as fp:
	#embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
	embedding, idx2word, word2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

nb_unknown_words = 10

for i in range(nb_unknown_words):
	idx2word[vocab_size - 1 - i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i] + '^	'

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

"""#Keras1
rnn_model.add(Embedding(vocab_size, embedding_size,
	input_length=maxlen,
	W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
	name='embedding_1'))
"""
#Keras 2
rnn_model.add(Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_size,
    input_length=maxlen,
    embeddings_regularizer=regularizer, 
    weights=[embedding], 
    mask_zero=True,
    name='embedding_1'))
rnn_model.add(SpatialDropout1D(rate=p_emb))

for i in range(rnn_layers):
	"""
	lstm = LSTM(rnn_size, return_sequences=True,
		W_regularizer=regularizer, U_regularizer=regularizer,
		b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
		name='lstm_%d'%(i+1)
		)
	"""

	#Keras2
	lstm = LSTM(rnn_size, 
        recurrent_regularizer=regularizer, 
        name='lstm_%d'%(i+1),
        bias_regularizer=regularizer, 
        dropout=p_W, 
        recurrent_dropout=p_U,
        kernel_regularizer=regularizer, 
        return_sequences=True)

	rnn_model.add(lstm)
	rnn_model.add(Dropout(p_dense, name='dropout_%d'%(i+1)))


###################################### Load ###########################################

"""
Use bottom weights from the trained model, and save the top weights for later
"""

def str_shape(x):
	return 'x'.join(map(str,x.shape))

def inspect_model(model):
	print model.name
	for i,l in enumerate(model.layers):
		print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
		weights = l.get_weights()
		for weight in weights:
			print str_shape(weight),
		print

"""
Modified version of keras load_weights that loads as much as it can
if there is a mismatch between file and model. It returns the weights
of the first layer in which the mismatch has happened
"""
def load_weights(model, filepath):
	print 'Loading', filepath, 'to', model.name
	flattened_layers = model.layers
	with h5py.File(filepath, mode='r') as f:
		#new file format
		layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

		#Batch weight value assignments in a single backend call which provides a speedup in TensorFlow.
		weight_value_tuples = []
		for name in layer_names:
			print name
			g = f[name]
			weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
			if len(weight_names):
				weight_values = [g[weight_name] for weight_name in weight_names]
				try:
					layer = model.get_layer(name=name)
				except:
					layer = None
				if not layer:
					print 'failed to find layer', name, 'in model'
					print 'weights', ' '.join(str_shape(w) for w in weight_values)
					print 'stopping to load all other layers'
					weight_values = [np.array(w) for w in weight_values]
					break
				symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
				weight_value_tuples += zip(symbolic_weights, weight_values)
				weight_values = None
		K.batch_set_value(weight_value_tuples)
	return weight_values