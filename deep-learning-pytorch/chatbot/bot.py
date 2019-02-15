import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import os
import re
import unicodedata
import codecs
from io import open
import itertools
import math

from data_prep import printLines, createFormattedFile
from vocabulary import loadPrepareData, trimRareWords


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


# load and preprocess data
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)


print('\nCorpus preview: \n\n----------------\n')
printLines(os.path.join(corpus, 'movie_lines.txt'))


# create data file
if os.path.exists(os.path.join(corpus, 'formatted_movie_lines.txt')):
    pass
else:
    createFormattedFile(os.path.join(corpus, 'formatted_movie_lines.txt'))

datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

save_dir = os.path.join('data', 'save')
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

print('\npairs: ')

for pair in pairs[:10]:
    print(pair)

print('\n')

pairs = trimRareWords(voc, pairs)