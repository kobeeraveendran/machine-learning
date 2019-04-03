import nltk
import numpy as np
import random
import string

# document similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('chatbot_corpus.txt', 'r', errors = 'ignore')
raw = f.read()
raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

# create lists of sentence and word tokens
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# lemmatization

lemmer = nltk.stem.WordNetLemmatizer()

def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# greetings
GREETING_INPUTS = ('hello', 'hi', 'greetings', 'sup', "what's up", 'hey')

GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'Hiya there']

def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# generating the response for general queries

def response(user_response):
    bot_response = ''
    sent_tokens.append(user_response)

    tfid_vec = TfidfVectorizer(tokenizer = lemNormalize, stop_words = 'english')
    tf_idf = tfid_vec.fit_transform(sent_tokens)
    vals = cosine_similarity(tf_idf[-1], tf_idf)
    index = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        bot_response += "I'm sorry, I don't understand you. :("

        return bot_response
    else:
        bot_response += sent_tokens[index]

        return bot_response

flag = True
print("BOT: Hi! I'll answer any questions you may have about chatbots. To exit, type 'Bye!'")

while flag:
    user_response = input()
    user_response = user_response.lower()
    
    if 'bye' not in user_response:
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("BOT: You're welcome!")
        else:
            if greeting(user_response) != None:
                print("BOT: ", greeting(user_response))

            else:
                print("BOT: ", end = '')
                print(response(user_response))
                sent_tokens.remove(user_response)

    else:
        flag = False
        print("BOT: See ya, and take care!")