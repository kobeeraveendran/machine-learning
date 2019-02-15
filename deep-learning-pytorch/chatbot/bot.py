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


# create formatted data file
# - file is of the format: lineID +++$+++ characterID +++$+++ movieID +++$+++ character +++$+++ text
# turn each line into a dict following the same pattern

def loadLines(filename, fields):
    lines = {}

    with open(filename, 'r', encoding = 'iso-8859-1') as f:
        for line in f:
            values = line.split('+++$+++')
            lineObj = {}

            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines