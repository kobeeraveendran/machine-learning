import os

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

# group lines by conversation (as specified by movie_conversations.txt)
def loadConversations(filename, lines, fields):
    conversations = []
    with open(filename, 'r', encoding = 'iso-8859-1') as f:
        for line in f:
            values = line.split('+++$+++')

            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            lineIds = eval(convObj['utteranceIDs'])

            convObj['lines'] = []
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])

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
