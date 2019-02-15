import unicodedata
import re

pad_token = 0
sos_token = 1
eos_token = 2

class Voc:
    
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {pad_token: 'PAD', sos_token: 'SOS', eos_token: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return

        self.trimmed = True

        keep_words = []

        for key, value in self.word2count.items():
            if value >= min_count:
                keep_words.append(key)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {pad_token: "PAD", sos_token: "SOS", eos_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

MAX_LENGTH = 10

# additional text processing before use
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()

    return s

def readVocs(datafile, corpus_name):
    print('Reading lines...')
    lines = open(datafile, encoding = 'utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)

    return voc, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print('\nStarted preparing training data...\n')
    voc, pairs = readVocs(datafile, corpus_name)

    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filterPairs(pairs)

    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('\nCounting words...\n')
    
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    print('Counted words: ', voc.num_words)

    return voc, pairs

MIN_COUNT = 3

def trimRareWords(voc, pairs, MIN_COUNT = MIN_COUNT):

    voc.trim(MIN_COUNT)

    keep_pairs = []

    for pair in pairs: 
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
            
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        
        if keep_input and keep_output:
            keep_pairs.append(pair)


    print("Trimmed from {} pairs to {} pairs, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs