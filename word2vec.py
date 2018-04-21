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

for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word