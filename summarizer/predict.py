import os, random, sys
import cPickle as pickle
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
import h5py


#maxlend = 50 # 0 - if we dont want to use description at all
maxlend = 50
maxlenh = 25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3  # match FN1
batch_norm = False

activation_rnn_size = 50 if maxlend else 0

# training parameters
seed = 42
p_W, p_U, p_dense, weight_decay = 0, 0, 0, 0
optimizer = 'adam'
batch_size = 64

nb_train_samples = 30000
nb_val_samples = 3000


DEBUG = True


################################## Read Word Embedding ##################################

with open('../DataExtractor/art/vocabEmbeddings.pkl', 'rb') as fp:
	embedding, idx2word, word2idx = pickle.load(fp)

vocab_size, embedding_size = embedding.shape
nb_unknown_words = 10

for i in range(nb_unknown_words):
	idx2word[vocab_size-1-i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

#PRINTING
def prt(label, x):
	print label+':',
	for w in x:
		print idx2word[w],
	print


################################## Model ##################################

# seed weight initialization
random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None


################################## RNN Model ##################################

rnn_model = Sequential()
"""
if DEBUG:
	from keras.layers import InputLayer
	model_input = InputLayer(input_shape=(maxlen,))
	rnn_model.add(model_input)
	rnn_model.output.tag.test_value = np.random.randint(vocab_size,size=(batch_size,maxlen)).astype('float32')
"""
rnn_model.add(
	Embedding(
		vocab_size,
		embedding_size,
		input_length=maxlen,
		#batch_input_shape=(batch_size,maxlen),
		embeddings_regularizer=regularizer,
		weights=[embedding],
		mask_zero=True,
		name='embedding_1'))

for i in range(rnn_layers):
	lstm = LSTM(
		rnn_size,
		return_sequences=True, # batch_norm=batch_norm,
		kernel_regularizer=regularizer,
		recurrent_regularizer=regularizer,
		bias_regularizer=regularizer,
		dropout=p_W,
		recurrent_dropout=p_U,
		name='lstm_%d'%(i+1))
	rnn_model.add(lstm)
	rnn_model.add(Dropout(p_dense, name='dropout_%d'%(i+1)))
"""
if DEBUG:
	print rnn_model.output.tag.test_value.shape
"""

################################## Load ##################################

rnn_model.load_weights('../DataExtractor/art/train.hdf5', by_name=True)

with h5py.File('../DataExtractor/art/train.hdf5', mode='r') as f:
	if 'layer_names' not in f.attrs and 'model_weights' in f:
		f = f['model_weights']
	weights = [np.copy(v) for v in f['time_distributed_1'].itervalues()]


################################## Summary Model ##################################

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
	desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

	# activation for every head word and every desc word
	print head_activations
	print desc_activations
	activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
	# make sure we dont use description words that are masked out
	activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)

	# for every head word compute weights for every desc word
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

	# for every head word compute weighted average of desc words
	desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=([2],[1]))
	"""
	if DEBUG:
		print desc_avg_word.tag.test_value.shape
		print head_words.tag.test_value.shape
	"""
	return K.concatenate((desc_avg_word, head_words))

model = Sequential()
model.add(rnn_model)

if activation_rnn_size:
	model.add(Lambda(
		simple_context,
		mask = lambda inputs,
		mask: mask[:,maxlend:],
		output_shape = lambda input_shape: (input_shape[0], maxlenh, 2*(rnn_size - activation_rnn_size)),
		name='simplecontext_1'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# out very own softmax
def output2probs(output):
	output = np.dot(output, weights[0]) + weights[1]
	output -= output.max()
	output = np.exp(output)
	output /= output.sum()
	return output


################################## Test ##################################


def lpadd(x, maxlend=maxlend, eos=eos):
	"""left (pre) pad a description to maxlend and then add eos.
	The eos is the input to predicting the first word in the headline
	"""
	if maxlend == 0:
		return [eos]
	n = len(x)
	if n > maxlend:
		x = x[-maxlend:]
		n = maxlend
	return [empty]*(maxlend-n) + x + [eos]

samples = [lpadd([3]*26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')

probs = model.predict(data, verbose=0, batch_size=1)

print probs