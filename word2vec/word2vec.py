# Kobee Raveendran
# from the Learn Word2Vec by Implementing it in Tensorflow article
# I've modified explanations to make them easier to understand for me (and possibly others)

import numpy as np
import tensorflow as tf
# for getting training time
from timeit import default_timer as timer

corpus_raw = 'He is king . The king is royal . She is the royal queen '

# conver to lower
corpus_raw = corpus_raw.lower()

# dictionary to translate words to integers and vice versa
words = []

for word in corpus_raw.split():
    # append only words (no periods)
    if word != '.':
        words.append(word)

# remove duplicates
words = set(words)

word2int = {}
int2word = {}

vocab_size = len(words)

# go through list of UNIQUE words and basically
# assigns a number to each (see python's enumerate() for clarity)
# i is the number, word is the element in the list words
for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# convert list of sentences into a list of words
# by splitting the sentences using . as a delimiter
raw_sentences = corpus_raw.split('.')

# will hold each sentence, with each element of this
# list being a sentence represented as a list of words
sentences = []

# for each sentence, split into list of words and append
# that list to the sentences list
for sentence in raw_sentences:
    sentences.append(sentence.split())

data = []

WINDOW_SIZE = 2

# creates a list of word - word pairs
# the max() and min() functions are used to ensure that we don't go 
# out of bounds (look before or after the sentence) when trying to predict words
# note that we go from the bottom of the range (index - window size) to the top of the range
# which is (index + window size)
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])

# see for yourself what this does!
# print(data)

#####################################################################################

# ONE-HOT ENCODING FOR THE VOCABULARY!

# sample vocabulary: pen, pineapple, apple
# word2int['pen'] = 0 --> becomes --> [1 0 0]
# word2int['pineapple'] = 1 --> becomes --> [0 1 0]
#word2int['apple'] = 2 --> becomes --> [0 0 1]

# note that the value of word2int[word] is the index in the one-hot vector

#####################################################################################

# function to convert index numbers in a vocabulary into one-hot vectors for it
def to_one_hot(hot_index, vocab_size):

    temp = np.zeros(vocab_size)
    temp[hot_index] = 1

    return temp

x_train = []        # input word
y_train = []        # output word

# this'll create a 2-D matrix of one-hot representations for every word
# if you look at word[] above, you'll see that they are the word pairs we generated
# earlier; you'll see that word[0] was one word from the sentence, and word[1] held 
# the words that came before and after word[0] (as two separate elements of course)
# so, x_train has the target word, and y_train has the words that can come before and 
# after the target word
for word in data:
    x_train.append(to_one_hot(word2int[word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[word[1]], vocab_size))

# convert to numpy arrays (helps for determining things like shape, and is formatted better)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# print them to confirm the claim above
# print(x_train)
# print(y_train)

# print shapes of each (useful for matrix operations): both should be (30,7)
# print(x_train.shape)
# print(y_train.shape)

#####################################################################################

# tensorflow stuff!

# create the placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape = (None, vocab_size))
y_label = tf.placeholder(tf.float32, shape = (None, vocab_size))

# convert the training data into the embedded representation

# embedding dimension
EMBEDDING_DIM = 7
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))

# this is the familiar A = Wx + b
hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))

prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

# now train!

# to time training
start = timer()
# initialize variables
init = tf.global_variables_initializer()

# create tensorflow session
with tf.Session() as sess:
    sess.run(init)

    # loss function
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices = [1]))

    # training step
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

    n_iters = 10000

    # apply the cross_entropy_loss function to determine the losses
    for _ in range(n_iters):
        sess.run(train_step, feed_dict = {x: x_train, y_label: y_train})

        print("Loss is: ", sess.run(cross_entropy_loss, feed_dict = {x: x_train, y_label: y_train}))

    # after training, check out the values of the weights and biases (W1 and b1)
    print('----------------------')
    print("W1 is: ", sess.run(W1))
    print('----------------------')
    print("b1 is: ", sess.run(b1))

end = timer()
elapsed = end - start
print("TRAINING TIME: " + str(elapsed))