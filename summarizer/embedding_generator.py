from six.moves import cPickle as pickle
import collections
import os
import numpy as np
import  math
import random
import tensorflow as tf
from six.moves import xrange

pickleFileName = '/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/article_and_heading_data.pickle'
pickleFileName = '/home/melvin/Documents/USC/news-in-short/DataExtractor/art/lesechos-fr-spider-data_6445-8935_politique.pickle'
pickleFileName = '/home/prakarsh_upmanyu23/latribune-fr-Spider-data_0-5848_politique.pickle'
pickleFileName = '/home/prakarsh_upmanyu23/output_files/concatenated.pickle'

with open(pickleFileName, 'r') as fp:
    data = pickle.load(fp)
    content = data['content']
    headings = data['heading']

print("no of heading : ", len(headings))
print("no of articles : ", len(content))


vocab_size = 0

def get_vocab(totaldata_list):

    """
        Uses the lst to create data dictionaries and vocab

        Args:
          totaldata_list: A concatenated list form of the [headings]+ [article]. This is our entire data set

        Returns:
        1. count: a list of lists where each sub list is of the form [<word>, <word occurrence count>]
        2. word_to_id: a dictionary of size <vocab_size>  most common words where key==>word  value==>some integer index
        3. id_to_word: reversed from the word_to_id dictionary where key==> some integer  and value==>word
        4. data : replacing the words in text corpus by their indexes in the word_to_id dict if word is present in the dictionary, else by 0

    """
    # Note : UNK is for words out of vocab of most common words. word_to_id['UNK'] = 0 and id_to_word[0] = 'UNK'

    count = [['UNK', -1]]
    all_content = ''
    for current_item in totaldata_list:
        #current_item = str(current_item).decode('iso-8859-9').encode('US-ASCII', 'replace')
        #current_item = str(current_item).decode('utf-8')
        all_content = all_content + str(current_item).lower().replace("\n", "")


    words= all_content.split(" ")
    print("no of unique words :", len(collections.Counter(words)))

    vocab_size = int(.50*len(collections.Counter(words)))  # set vocab size depending on the number of total unique words in the corpus
    print("Considering a vocab size of :", vocab_size)
    # from the list from of data, find the the top <vocab size> words
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    collections.Counter(words)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)

    data = list()
    unk_count = 0
    for word in words:

        if word in word_to_id:
            index = word_to_id[word]
        else:
            index = 0    # or index = word_to_id['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return data, count, word_to_id, id_to_word, vocab_size


data, count, word_to_id, id_to_word, vocab_size = get_vocab(content+headings)
#print id_to_word

# Generating the training batch for skip gram model

'''
For a skip-gram model the context is defined as the words to the left & right [window size ]of the target word
NOTE : skip gram tries to predict the 'EACH' context word from the target word

'''

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0    # has to be true for the program to proceed
    assert num_skips <= 2*skip_window     # has to be true for the program to proceed

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)

    span = 2* skip_window +1 # 2*1+1 = 3 ie (context left, target word, content right)

    buffer = collections.deque(maxlen=span)    # buffer is out little window(size =3) that spans through entire data list linearly

    for _  in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)
    '''
    buffer: deque([3084, 12, 6], maxlen=3)
    buffer: deque([12, 6, 195], maxlen=3)
    buffer: deque([6, 195, 2], maxlen=3)
    buffer: deque([195, 2, 3135], maxlen=3)

    '''
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])  # since deque is defined to be of max len 3 , it pops front and appends to next data[data_index]

        data_index = (data_index + 1) % len(data)

    return batch, labels

batch, labels = generate_batch(batch_size = 8 , num_skips=2, skip_window=1)


################## Training Word2Vec ####################################

batch_size = 128
embedding_size = 128
skip_window = 1  # how many words to consider left and right
num_skips = 2    # how many times to re-use an input to generate a label ie go one step left and one step right of the target

'''
We pick a random validation set to sample nearest neighbors. Here we limit the
validation samples to the words that have a low numeric ID, which by
construction are also the most frequent.
'''

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample. This is the value of K according to nce

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/gpu:0'):

    # embedding tensor of the shape <vocab size> X <embedding_dimension>
    # Note: we are generating embeddings only for the top <vocab_size> ie most common words and not all words in the corpus
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocab_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      print(embeddings)
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = id_to_word[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = id_to_word[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


wordEmbeddingFile = '/home/melvin/Documents/USC/news-in-short/DataExtractor/art/vocabEmbeddings.pkl'
wordEmbeddingFile = '/home/prakarsh_upmanyu23/vocabEmbeddings.pkl'
wordEmbeddingFile = '/home/prakarsh_upmanyu23/output_files/concatenated_vocabEmbeddings.pkl'
with open(wordEmbeddingFile, 'wb') as f:
    pickle.dump((final_embeddings, id_to_word, word_to_id),f)


train_labels = []
for headline in headings:
    tokenized_headline = str(headline).split()
    current_headline = []
    for word in tokenized_headline:
        if(word in word_to_id):
            current_headline.append(word_to_id[word])
        else:
            current_headline.append(word_to_id['UNK'])

    train_labels.append(current_headline)


train_data = []
for article in content:
    tokenized_article = str(article).split()
    current_article = []
    for word in tokenized_article:
        if (word in word_to_id):
            current_article.append(word_to_id[word])
        else:
            current_article.append(word_to_id['UNK'])

    train_data.append(current_article)

trainingDataFile = '/home/melvin/Documents/USC/news-in-short/DataExtractor/art/train_data.pkl'
trainingDataFile = '/home/prakarsh_upmanyu23/train_data.pkl'
trainingDataFile = '/home/prakarsh_upmanyu23/output_files/concatenated_train_data.pkl'
with open(trainingDataFile, 'wb') as f:
    pickle.dump((train_data,train_labels),f)
