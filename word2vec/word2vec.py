# Kobee Raveendran
# from the Learn Word2Vec by Implementing it in Tensorflow article
# I've modified explanations to make them easier to understand for me (and possibly others)

import numpy as np
import tensorflow as tf

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

