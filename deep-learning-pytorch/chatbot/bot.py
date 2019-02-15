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

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


# load and preprocess data
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)

def printLines(file, n = 10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()

    for line in lines[:n]:
        print(line)

print('\nCorpus preview: \n\n----------------\n')
printLines(os.path.join(corpus, 'movie_lines.txt'))
