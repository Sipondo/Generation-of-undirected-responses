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

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(set, name, reverse=False):
    language = Lang(name)
    # set = filterPairs(set)
    # print("Trimmed to %s sentence pairs" % len(pairs))
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

def buildLang():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_donald = pd.read_csv(os.path.join("csvs", "export_thedonald.csv"))

    #
    #
    # oracle_samples = torch.load(os.path.join("seqGAN-master", "oracle_samples.trc"))

    text_input_data = csv_donald['body']

    text_input_data

    text_input_noempty = text_input_data[text_input_data != "[deleted]"]
    text_input_noempty = text_input_noempty[text_input_noempty != "[removed]"]
    text_input_noempty = text_input_noempty[text_input_noempty != "nan"]
    text_input_noempty = text_input_noempty[text_input_noempty != None]


    SOS_token = 0
    EOS_token = 1


    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    pattern = re.compile('[^a-zA-Z0-9 :]')
    p_spaces = re.compile(' +')

    t_input_filtered = text_input_noempty.replace(pattern,"").str.lower().replace(p_spaces," ").to_frame()
    t_input_filtered = t_input_filtered[t_input_filtered.body.apply(type) != float]['body']
    t_input_bound = t_input_filtered[t_input_filtered.map(lambda x: len(x.split())) == 20].reset_index(drop=True)

    # with open("donald.txt",'w')  as file:
    #     for line in t_input_bound:
    #         file.write(line)
    #         file.write(". ")

    resulting_set, language = prepareData(t_input_bound,"donald")

    return language

    # encoded_set = resulting_set.map(lambda x: indexesFromSentence(language, x))
    #
    # npmatrix_out = np.zeros((len(encoded_set.values),np.max([len(x) for x in encoded_set.values])),dtype=np.int32)
    #
    # for i in range(npmatrix_out.shape[0]):
    #     for j in range(len(encoded_set[i])):z
    #         npmatrix_out[i,j] = encoded_set[i][j]
    #
    #
    # tensor_out = torch.tensor(npmatrix_out, dtype = torch.long, device=device)
    #
    # torch.save(tensor_out,"donald.trc")
    #
