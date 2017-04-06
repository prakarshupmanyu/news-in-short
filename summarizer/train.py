import os
from six.moves import cPickle as pickle
from six.moves import xrange
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam, RMSprop
import gc

import keras
keras.__version__


maxlen_desc = 25
maxlen_heading = 25
maxlen = maxlen_desc + maxlen_heading

rnn_size = 512
rnn_layers = 3
batch_norm = False

activation_rnn_size = 40 if maxlen_desc else 0

seed=42
prob_W, prob_U, prob_dense, prob_emb, weight_decay = 0, 0, 0, 0, 0

LR = 1e-4
batch_size=64

with open('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/vocabEmbeddings.pkl', 'rb') as pkfile:
    embedding,id_to_word, word_to_id = pickle.load(pkfile)
embedding_size, embedding_dim = embedding.shape

with open('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/train_data.pkl', 'rb') as f:
    train_data, train_labels = pickle.load(f)


print('number of examples article:',len(train_data),'  and its headings :' , len(train_labels))
print('dimension of embedding space for words',embedding_dim)
print("no of words in dictionary for which embeddings are calculated  : ", len(id_to_word), len(word_to_id))


# split the training data into Train and Cross Validation

nb_train_samples =  800
nb_val_samples = 200

X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_labels, test_size=nb_val_samples, random_state=seed)

"""
1. Input data (X) is made from maxlend description words followed by eos followed by headline words followed by eos
   if description is shorter than maxlend it will be left padded with "_"  eg _ _ ... _ 4 321 842 11 10
   if entire data is longer than maxlen it will be clipped
   if it is shorter it will be right padded with empty.    4 83 1948 3818 38381 381 11 13 49 ... _ _ _ _ _

2. labels (Y) are the headline words followed by eos and clipped or padded to maxlenh

"""

empty = 0
eos = 1
id_to_word[empty] = '_'
word_to_id[eos] = '<eos>'


# seed weight initialization
random.seed(seed)
np.random.seed(seed)

############################## Defining Model ####################################

'''
Adding the first layer to out model which the is Embeddings learned from the embeddings_generator.py
embedding_size = # of words for the embeddings have been computed
embedding_dim = # dim of each embeddings

'''
regularizer = l2(weight_decay) if weight_decay else None

model = Sequential()
model.add(Embedding(embedding_size, embedding_dim,
                    input_length=maxlen,
                    W_regularizer=regularizer, dropout=prob_emb, weights=[embedding], mask_zero=True,
                    name='embedding_1'))

for i in range(rnn_layers):
    lstm = LSTM(rnn_size, return_sequences=True,
                W_regularizer=regularizer, U_regularizer=regularizer,
                b_regularizer=regularizer, dropout_W=prob_W, dropout_U=prob_U,
                name='lstm_%d'%(i+1)
                  )
    model.add(lstm)
    model.add(Dropout(prob_dense,name='dropout_%d'%(i+1)))



from keras.layers.core import Lambda
import keras.backend as K


def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlen_desc, maxlenh=maxlen_heading):
    desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
    head_activations, head_words = head[:, :, :n], head[:, :, n:]
    desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]


    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :maxlend], 'float32'), 1)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies, (-1, maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self, **kwargs):
        super(SimpleContext, self).__init__(simple_context, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlen_desc:]

    def get_output_shape_for(self, input_shape):   # what does it do ????????????????????????
        nb_samples = input_shape[0]
        n = 2 * (rnn_size - activation_rnn_size)
        return (nb_samples, maxlen_heading, n)

if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))

model.add(TimeDistributed(Dense(embedding_size,
                                W_regularizer=regularizer, b_regularizer=regularizer,
                                name = 'timedistributed_1')))

model.add(Activation('softmax', name='activation_1'))



model.compile(loss='categorical_crossentropy', optimizer='adam')

K.set_value(model.optimizer.lr,np.float32(LR))

gc.collect()

def str_shape(x):
    return 'x'.join(map(str, x.shape))


def inspect_model(model):
    for i, l in enumerate(model.layers):
        print(i, 'cls=%s name=%s' % (type(l).__name__, l.name))
        weights = l.get_weights()
        for weight in weights:
            print (str_shape(weight))
        print("\n")

inspect_model(model)

if 'train' and os.path.exists('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/train.hdf5'):
    model.load_weights('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/train.hdf5')


###################### Test ########################################

def leftPadd(x, maxlend=maxlen_desc, eos=eos):
    """
    we know if description is shorter than maxlen_desc it will be left padded with "_"  eg _ _ ... _ 4 321 842 11 10
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]     # returning a left padded version of the desc


samples = [leftPadd([3]*26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')


print(np.all(data[:,maxlen_desc] == eos))

print(data.shape,map(len, samples))

probs = model.predict(data, verbose=0, batch_size=1)
print(probs.shape)


############################# Data Generator ###########################

"""
1. Data generator generates batches of inputs and outputs/labels for training. The inputs are each made from two parts.
   The first maxlen_desc words are the original description, followed by eos followed by the headline which we want to predict,
   except for the last word in the headline which is always eos and then empty padding until maxlen words.

2. For each, input, the output is the headline words (without the start eos but with the ending eos)
   padded with empty words up to maxlen_heading words. The output is also expanded to be y-hot encoding of each word.
"""

nflips=10

def flip_headline(x, nflips=None, model=None, debug=False):
    """
    given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
    with words predicted by the model
    """
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    assert np.all(x[:, maxlen_desc] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(xrange(maxlen_desc + 1, maxlen), nflips))
        if debug and b < debug:
            print(b)
        for input_idx in flips:
            if x[b, input_idx] == empty or x[b, input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlen_desc + 1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = 0
            if debug and b < debug:
                print("")
                #print('%s => %s' % (idx2word[x_out[b, input_idx]], idx2word[w]))
            x_out[b, input_idx] = w
        if debug and b < debug:
            print
    return x_out



def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [(leftPadd(xd) + xh) for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)

    y = np.zeros((batch_size, maxlen_heading, embedding_size))
    for i, xh in enumerate(xhs):
        xh = xh + [eos] + [empty] * maxlen_heading  # output does have a eos at end
        xh = xh[:maxlen_heading]
        y[i, :, :] = np_utils.to_categorical(xh, embedding_size)

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
        new_seed = random.randint(0, sys.maxsize)
        random.seed(c + 123456789 + seed)
        for b in range(batch_size):
            t = random.randint(0, len(Xd) - 1)

            xd = Xd[t]
            s = random.randint(min(maxlen_desc, len(xd)), max(maxlen_desc, len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlen_heading, len(xh)), max(maxlen_heading, len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c += 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)

r = next(gen(X_train, Y_train, batch_size=batch_size))
print(r[0].shape, r[1].shape, len(r))

def prt(label, x):
    print(label+" :")
    for w in x:
        if(w in id_to_word):
            print(id_to_word[w], end='  ')
        else:
            print("____")

def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlen_desc] == eos
        x = Xtr[i,:maxlen_desc]
        y = Xtr[i,maxlen_desc:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy)
        print("\n")
        prt('H',y)
        print("\n")
        if maxlen_desc:
            prt('D',x)
        print("\n\n")
gc.collect()


test_gen(gen(X_train, Y_train, batch_size=batch_size))

test_gen(gen(X_train, Y_train, nflips=6, model=model, debug=False, batch_size=batch_size))

valgen = gen(X_val, Y_val,nb_batches=3, batch_size=batch_size)


#check that valgen repeats itself after nb_batches
for i in range(4):
    test_gen(valgen, n=1)


###################### Train the Model #####################################

history = {}

traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
valgen = gen(X_val, Y_val, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

r = next(traingen)
print(r[0].shape, r[1].shape, len(r))

for iteration in range(500):
    print ('Iteration', iteration)
    h = model.fit_generator(traingen, samples_per_epoch=nb_train_samples,
                        nb_epoch=1, validation_data=valgen, nb_val_samples=nb_val_samples
                           )
    for k,v in h.history.iteritems():
        history[k] = history.get(k,[]) + v
    with open('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/train.history.pkl','wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/train.hdf5', overwrite=True)