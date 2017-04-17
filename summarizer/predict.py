#!/usr/local/bin/python
# coding: utf-8
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
#maxlenh = 25 #shows an error with anything other than 50
maxlenh = 50
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
	print f['time_distributed_1'].itervalues()
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
	# Prevent error
	# TypeError: Cannot cast array data from dtype('float32') to dtype('<U32') according to the rule 'safe'
	output = map(lambda x: float(x),output)
	#weights[0] = map(lambda x: float(x),weights[0])
	output = np.asarray(output)
	weights[0] = np.asarray(weights[0])
	
	print weights

	output = (np.dot(output, weights[0]) + weights[1])
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


# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
def beamsearch(predict, start=[empty]*maxlend + [eos], avoid=None, avoid_score=1,
               k=1, maxsample=maxlen, use_unk=True, oov=vocab_size-1, empty=empty, eos=eos, temperature=1.0):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """
    def sample(energy, n, temperature=temperature):
        """sample at most n different elements according to their energy"""
        n = min(n,len(energy))
        prb = np.exp(-np.array(energy) / temperature )
        res = []
        for i in xrange(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb/z, 1))
            res.append(r)
            prb[r] = 0. # make sure we select each element only once
        return res

    dead_samples = []
    dead_scores = []
    live_samples = [list(start)]
    live_scores = [0]

    while live_samples:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples, empty=empty)
        assert vocab_size == probs.shape[1]

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        cand_scores[:,empty] = 1e20
        if not use_unk and oov is not None:
            cand_scores[:,oov] = 1e20
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i,a[n]] += avoid_score
        live_scores = list(cand_scores.flatten())
        

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        dead_scores = [dead_scores[r] for r in ranks if r < n]
        dead_samples = [dead_samples[r] for r in ranks if r < n]
        
        live_scores = [live_scores[r-n] for r in ranks if r >= n]
        live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        def is_zombie(s):
            return s[-1] == eos or len(s) > maxsample
        
        # add zombies to the dead
        dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
        dead_samples += [s for s in live_samples if is_zombie(s)]
        
        # remove zombies from the living 
        live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
        live_samples = [s for s in live_samples if not is_zombie(s)]

    return dead_samples, dead_scores

def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    """for every sample, calculate probability for every possible label
    you need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = map(len, samples)
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([output2probs(prob[sample_length-maxlend-1]) for prob, sample_length in zip(probs, sample_lengths)])


def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < vocab_size-nb_unknown_words else x for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def vocab_unfold(desc,xs):
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= vocab_size-nb_unknown_words:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]

def gensamples(X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    if X is None or isinstance(X,int):
        if X is None:
            i = random.randint(0,len(X_test)-1)
        else:
            i = X
        print 'HEAD %d:'%i,' '.join(idx2word[w] for w in Y_test[i])
        print 'DESC:',' '.join(idx2word[w] for w in X_test[i])
        sys.stdout.flush()
        x = X_test[i]
    else:
        #x = [word2idx[w.rstrip('^')] for w in X.split()]
        x = [word2idx.get(w.rstrip('^'),0) for w in X.split()]
        
    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid,str) or isinstance(avoid[0], int):
            avoid = [avoid]
        avoid = [a.split() if isinstance(a,str) else a for a in avoid]
        avoid = [vocab_fold([w if isinstance(w,int) else word2idx[w] for w in a])
                 for a in avoid]

    print 'HEADS:'
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
                                   k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                print score, ' '.join(words)
        #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
        else:
                print score, ' '.join(words)
        codes.append(code)
    return samples

seed = 8
random.seed(seed)
np.random.seed(seed)


X = "PARIS (Dow Jones)--Le groupe immobilier Icade a fait part lundi d'un résultat net récurrent en hausse de 1,2% sur l'exercice 2015, soutenu par les revenus de sa foncière santé. Le résultat net récurrent s'est inscrit à 273 milions d'euros en 2015 contre 269,9 millions d'euros en 2014. L'excédent brut opérationnel a pour sa part atteint 501,5 millions d'euros, en repli de 3,5%."
samples = gensamples(X=X, skips=2, batch_size=batch_size, k=10, temperature=1.)
