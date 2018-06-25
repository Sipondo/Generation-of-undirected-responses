################
#Pytorch RNN for sentence generation
#Iterates over sequences learning to predict successors
#Requires .trc file that is generated (on import) by inspect_tensor_fixed
#
#The network is heavily based on the pytorch tutorial:
#Generating Names with a Character-Level RNN
#https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
#
#We heavily altered the example, switching it from Character generation to
#sentence generation (with a similar representation system).
#The original network also included support for multiple languages. In
#our research this was not necessary and was thus ommitted.
#
#You can also find a GPU version in the same repo called sentence_rnn_gpu.py
#
#All other code is strictly our work; Bauke Brenninkmeijer and Ties Robroek.

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import sys
sys.path.append("")
import torch
import torch.nn as nn
import inspect_tensor_fixed
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


import random
lang = inspect_tensor_fixed.buildLang()
n_letters = lang.n_words
oracle_samples_path = './donald.trc'
input_data = torch.load(oracle_samples_path).type(torch.LongTensor)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)].data

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][letter] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [line[li] for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample():
    line = randomChoice(input_data)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn = RNN(n_letters, 128, n_letters)

n_iters = 50
print_every = 5
plot_every = 5
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


plt.figure()
plt.plot(all_losses)



max_length = 10


def sample(start_letter='they'):
    with torch.no_grad():  # no need to track history in sampling
        start_letter = [lang.word2index[start_letter]]
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

sample()
