import os
import csv
import re
import codecs
import unicodedata

# create formatted data file
# - file is of the format: lineID +++$+++ characterID +++$+++ movieID +++$+++ character +++$+++ text

corpus = os.path.join('data', 'cornell movie-dialogs corpus')

def printLines(file, n = 10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()

    for line in lines[:n]:
        print(line)

# turn each line into a dict following the same pattern
def loadLines(filename, fields):
    lines = {}

    with open(filename, 'r', encoding = 'iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {}

            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines

# group lines by conversation (as specified by movie_conversations.txt)
def loadConversations(filename, lines, fields):
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# turn conversation lines into question + answer pairs, ignoring any mod 2 remainder lines 
# (since they have no response)
def extractSentencePairs(conversations):
    qa_pairs = []

    for conversation in conversations:

        for i in range(len(conversation['lines']) - 1):
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i + 1]['text'].strip()

            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs

def createFormattedFile(datafile):
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
    MOVIE_CONVERSATIONS_FIELDS = ['characterID', 'character2ID', 'movieID', 'utteranceIDs']

    print('\nProcessing corpus...\n')
    lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)

    print('\nLoading conversations...\n')
    conversations = loadConversations(os.path.join(corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)

    # write to new csv
    print('\nWriting newly formatted file...\n')
    with open(datafile, 'w', encoding = 'utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter = delimiter, lineterminator = '\n')

        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    print('\nSample lines from file:')
    printLines(datafile)

    return datafile