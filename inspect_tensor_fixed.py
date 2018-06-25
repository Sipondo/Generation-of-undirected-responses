################
#Pytorch dataset processing and language generation
#Filters all the subreddit comments and generates two dataset files:
#<datasetname>.trc
#<datasetname>.txt
#The txt is used by the markov chain. The torch tensors are used
#by the RNN and the GAN (as this is much faster).
#Subreddit comments are stripped from any symbols etc. and are
#filtered by dictionary.
#The dictionary is based on https://github.com/dwyl/english-words
#which is based on a dataset by Infochimps, copyright belongs to them.
#
#The language module is based on the pytorch tutorial:
#Translation with a Sequence to Sequence Network and Attention
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py
#
#We heavily altered the example, creating a system that allows us to
#translate sentences in a sequence of embeddings.
#These sequences are then processed sequence-by-sequence (seqGan) and iteratively
#(RNN).
#
#All other code is strictly our work; Bauke Brenninkmeijer and Ties Robroek.

import sys
sys.path.append("")
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import re, string
import random

dataset_filename = "thedonald"
dataset_exportname = "donald"

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            if "http" not in word:
                if word != '':
                    self.word2index[word] = self.n_words
                    self.word2count[word] = 1
                    self.index2word[self.n_words] = word
                    self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def prepareData(set, name, reverse=False):
    language = Lang(name)

    print("Counting words...")
    for sentence in set:
        try:
            language.addSentence(sentence)
        except Exception:
            print("Ignored: " + str(sentence))

    print("Counted words:")
    print(language.name, language.n_words)
    return set, language

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word != '' and not "http" in word]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def load_words():
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

def buildLang():
    english_words = load_words()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df1 = pd.read_csv(os.path.join("csvs", "export_"+dataset_filename+"_jan.csv"))
    df2 = pd.read_csv(os.path.join("csvs", "export_"+dataset_filename+"_feb.csv"))
    df3 = pd.read_csv(os.path.join("csvs", "export_"+dataset_filename+"_mar.csv"))
    csv_total = pd.concat([df1, df2, df3])

    #
    #
    # oracle_samples = torch.load(os.path.join("seqGAN-master", "oracle_samples.trc"))

    text_input_data = csv_total['body']

    text_input_noempty = text_input_data[text_input_data != "[deleted]"]
    text_input_noempty = text_input_noempty[text_input_noempty != "[removed]"]
    text_input_noempty = text_input_noempty[text_input_noempty != "nan"]
    text_input_noempty = text_input_noempty[text_input_noempty != None]

    SOS_token = 0
    EOS_token = 1


    MAX_LENGTH = 10

    pattern = re.compile('[^a-zA-Z0-9 :]')
    p_spaces = re.compile(' +')

    t_input_filtered = text_input_noempty.replace(pattern,"").str.lower().replace(p_spaces," ").to_frame()
    t_input_filtered = t_input_filtered[t_input_filtered.body.apply(type) != float]['body']
    t_input_bound = t_input_filtered[t_input_filtered.map(lambda x: len(x.split())) == 10]
    t_input_bound = t_input_bound[t_input_bound.map(lambda x: np.all([y in english_words for y in x.split()]))].reset_index(drop=True)

    with open(dataset_exportname+".txt",'w')  as file:
        for line in t_input_bound:
            file.write(line)
            file.write(". ")

    resulting_set, language = prepareData(t_input_bound,"donald")


    encoded_set = resulting_set.map(lambda x: indexesFromSentence(language, x))

    npmatrix_out = np.zeros((len(encoded_set.values),np.max([len(x) for x in encoded_set.values])),dtype=np.int32)

    for i in range(npmatrix_out.shape[0]):
        for j in range(len(encoded_set[i])):
            npmatrix_out[i,j] = encoded_set[i][j]


    tensor_out = torch.tensor(npmatrix_out, dtype = torch.long, device=device)

    torch.save(tensor_out,dataset_exportname+".trc")
    return language#, t_input_bound
