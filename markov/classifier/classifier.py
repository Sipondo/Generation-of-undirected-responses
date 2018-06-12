#import unicodecsv as csv
import csv
import wikipedia
import pprint
import urllib
import urllib.request
import time
from tqdm import tqdm
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.classify import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

pret = "./classifier"
model = KeyedVectors.load_word2vec_format(pret+'/GoogleNews-vectors-negative300.bin', binary=True)

binding_list = {}
with open(pret+"/pol_bindings.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        democrat = int(row["democrat"])
        republican = int(row["republican"])
        binding_list[int(row["id"])] = 2 if democrat>republican else 1 if democrat<republican else 0

# not_suited = [(binding_list[key], key) for key in binding_list if binding_list[key] == 0]
# suited = [(binding_list[key], key) for key in binding_list if binding_list[key] > 0]
class_list = {key: binding_list[key] for key in binding_list if binding_list[key] > 0}

tweets = []
with open(pret+"/pol_tweets.csv", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=";")
    for row in tqdm(reader):
        tweets.append([row["user_id"], row["tweet_text"]])

label = {1: "Republican", 2:"Democrat"}

def check_listed(tweet):
    return int(tweet[0]) in class_list

def class_tweet(tweet):
    return class_list[int(tweet[0])]

def return_tweet_text(tweet):
    return str(label[class_tweet(tweet)]) + ": " + str(tweet[1])

def process_tweet_text(tweet_text):
    return re.sub(r'[^\w]', ' ', tweet_text).lower()

import numpy as np
def average_vector(tweet_text):
    vectorlist = [[0 for i in range(300)]]
    for word in process_tweet_text(tweet_text).split():
        if word in model:
            if not word in stopw:
                vectorlist.append(model.get_vector(word))
    return np.mean(np.asarray(vectorlist),axis=0)

stopw = stopwords.words('english')
# def build_vocab(tweet_text, all_words, stopw):
#     for word in word_tokenize(process_tweet_text(tweet_text)):
#         if not word in all_words:
#             if not word in stopw:
#                 all_words.append(word)
#     return all_words
#
# all_words = []
# for i in tqdm(range(len(tweets)-1,-1,-1)):
#     all_words = build_vocab(tweets[i][1],all_words,stopw)

def return_features(tweet_text):
    vector = average_vector(tweet_text)
    return {i: vector[i] for i in range(len(vector))}

labelled_tweets = []
for i in tqdm(range(len(tweets)-1,-1,-1)):
    if check_listed(tweets[i]):
        labelled_tweets.append([return_features(tweets[i][1]),class_tweet(tweets[i])])
    tweets.pop()

classifier = NaiveBayesClassifier.train(labelled_tweets)

categories = [1, 2]
#stopWords = stopwords.words('english')
vectorizer = CountVectorizer()#stop_words = stopWords)
transformer = TfidfTransformer()

# all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
# t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

classifier = NaiveBayesClassifier.train(labelled_tweets[:120000])

#def key_tweet(tweet):
#    return [{"text:": tweet[0]}]

#labelled_tweets[500]
# print(accuracy(classifier,labelled_tweets[120000:140000]))

def classify(text):
    vector = return_features(text)
    return classifier.classify(text)


#classifier.classify(labelled_tweets[500][0])
# test_tweet = tweets[500000]
# print(return_tweet(test_tweet))
#
# democounter = 0
# repcounter = 0
# for tweet in tweets:
#     suited_list[int(test_tweet[0])]
