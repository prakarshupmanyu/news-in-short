import os, random, sys
import cPickle as pickle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import Adam, RMSprop 


maxlend = 25 # 0 - if we dont want to use description at all
maxlenh = 25
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm = False

activation_rnn_size = 40 if maxlend else 0

# training parameters
seed = 42
p_W, p_U, p_dense, weight_decay = 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size = 64
nflips = 10

nb_train_samples = 30000
nb_val_samples = 100 #For training on system


################################################# Embedding #################################################
wordEmbeddingFile = '../DataExtractor/art/vocabEmbeddings.pkl'
wordEmbeddingFile = '/home/prakarsh/Desktop/vocabEmbeddings.pkl'
with open(wordEmbeddingFile, 'rb') as fp:
    embedding, idx2word, word2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

trainingDataFile = '../DataExtractor/art/train_data.pkl'
trainingDataFile = '/home/prakarsh/Desktop/train_data.pkl'
with open(trainingDataFile, 'rb') as fp:
    X, Y = pickle.load(fp)

nb_unknown_words = 10

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

oov0 = vocab_size - nb_unknown_words

for i in range(oov0, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'


################################################# Define Model #################################################

# seed weight initialization
random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

model = Sequential()
model.add(Embedding(
    vocab_size, 
    embedding_size,
    input_length=maxlen,
    embeddings_regularizer=regularizer,
    weights=[embedding],
    mask_zero=True,
    name='embedding_1'))

for i in range(rnn_layers):
    lstm = LSTM(rnn_size, 
        return_sequences=True, # batch_norm=batch_norm,
        kernel_regularizer=regularizer, 
        recurrent_regularizer=regularizer,
        bias_regularizer=regularizer,
        dropout=p_W,
        recurrent_dropout=p_U,
        name='lstm_%d'%(i+1))
    model.add(lstm)
    model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))

if activation_rnn_size:
    model.add(Lambda(
        simple_context,
        mask = lambda inputs,
        mask: mask[:,maxlend:],
        output_shape = lambda input_shape: (input_shape[0], maxlenh, 2*(rnn_size - activation_rnn_size)),
        name='simplecontext_1'))

model.add(TimeDistributed(Dense(vocab_size,
    kernel_regularizer=regularizer,
    bias_regularizer=regularizer,
    name = 'time_distributed_1')))
model.add(Activation('softmax', name='activation_1'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
K.set_value(model.optimizer.lr,np.float32(LR))

if os.path.exists('../DataExtractor/art/train.hdf5'):
    model.load_weights('../DataExtractor/art/train.hdf5')


################################################# Test #################################################

def lpadd(x, maxlend=maxlend, eos=eos):
    """
    left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

samples = [lpadd([3] * 26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')


print np.all(data[:,maxlend] == eos)
print data.shape,map(len, samples)

probs = model.predict(data, verbose=0, batch_size=1)
print probs.shape


############################# Data Generator ###########################

"""
1. Data generator generates batches of inputs and outputs/labels for training. The inputs are each made from two parts.
   The first maxlen_desc words are the original description, followed by eos followed by the headline which we want to predict,
   except for the last word in the headline which is always eos and then empty padding until maxlen words.

2. For each, input, the output is the headline words (without the start eos but with the ending eos)
   padded with empty words up to maxlen_heading words. The output is also expanded to be y-hot encoding of each word.
"""


def flip_headline(x, nflips=None, model=None, debug=False):
    """
    given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline) with words predicted by the model
    """
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(xrange(maxlend+1,maxlen), nflips))
        if debug and b < debug:
            print b,
        for input_idx in flips:
            if x[b,input_idx] == empty or x[b,input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend+1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print '%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]),
            x_out[b,input_idx] = w
        if debug and b < debug:
            print
    return x_out


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """
    description and hedlines are converted to padded input vectors. headlines are one-hot to label
    """
    batch_size = len(xhs)

    x = [(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = xh + [eos] + [empty]*maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)

    return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    
    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxint)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])
            
            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)


r = next(gen(X_train, Y_train, batch_size=batch_size))
print (r[0].shape, r[1].shape, len(r))

def prt(label, x):
    print label+':',
    for w in x:
        print idx2word[w],
    print


def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy)
        prt('H',y)
        if maxlend:
            prt('D',x)

test_gen(gen(X_train, Y_train, batch_size=batch_size))

test_gen(gen(X_train, Y_train, nflips=6, model=model, debug=False, batch_size=batch_size))

valgen = gen(X_test, Y_test,nb_batches=3, batch_size=batch_size)


#check that valgen repeats itself after nb_batches
for i in range(4):
    test_gen(valgen, n=1)

###################### Train the Model #####################################

history = {}

traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

r = next(traingen)
print (r[0].shape, r[1].shape, len(r))

#for iteration in range(500):
for iteration in range(40):
    print 'Iteration', iteration
    h = model.fit_generator(traingen, 
        #steps_per_epoch=nb_train_samples//batch_size,
        steps_per_epoch=10,
        epochs=1,
        validation_data=valgen,
        validation_steps=nb_val_samples
        )
    for k,v in h.history.iteritems():
        history[k] = history.get(k,[]) + v
    with open('../DataExtractor/art/train.history.pkl','wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('../DataExtractor/art/train.hdf5', overwrite=True)
    #gensamples(batch_size=batch_size)
