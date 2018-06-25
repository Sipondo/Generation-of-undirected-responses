################
#Pytorch seqGAN for sentence generation
#Processes sequences and attempts to fool itself by producing new sequences.
#Requires .trc file that is generated (on import) by inspect_tensor_fixed
#
#The network is heavily based on work by:
#suragnair, https://github.com/suragnair/seqGAN
#
#The work has been heavily altered, switching from oracle to dataset.
#The original work was on synthetic data and multiple changes were required
#to make it fit for the reddit data.
#Original work by suragnair heavily penalized the genrator as it was
#very powerful on the synthetic data. We had the opposite problem and had
#to greatly increase the potential of the generator with respect to the
#discriminator.
#
#The system can take a rather long while to train especially when supplied
#a large dataset. There is no CPU version as that would be unreasonable.
#(though by changing the CUDA flag it should somewhat work on just a CPU).
#
#All other code is strictly our work; Bauke Brenninkmeijer and Ties Robroek.

from __future__ import print_function
from math import ceil
import numpy as np
import sys
sys.path.append("")
sys.path.append("..")
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

import inspect_tensor_fixed

language = inspect_tensor_fixed.buildLang()
CUDA = True
VOCAB_SIZE = 16000#5000
MAX_SEQ_LEN = 10
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 150#100
DIS_PRETRAIN_EPOCHS = 7#50
ADV_TRAIN_EPOCHS = 150 #50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 128
GEN_HIDDEN_DIM = 256
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './donald.trc'

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, start_letter=START_LETTER, gpu=CUDA)
        #
        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # # sample from generator and compute oracle NLL
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                                start_letter=START_LETTER, gpu=CUDA)
    #
    # print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    #pos_val = oracle.sample(100)
    random_sample = torch.randperm(oracle.shape[0])[:1000]
    pos_val = oracle[random_sample]
    neg_val = generator.sample(1000)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).item()/2000.))

# MAIN
if __name__ == '__main__':
    # oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    # # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    #
    oracle = torch.load(oracle_samples_path).type(torch.LongTensor)
    oracle_samples = oracle[:10000]
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        oracle_samples = oracle_samples.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=0.005)#e-2)
    train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters(), lr=0.005)
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, DIS_PRETRAIN_EPOCHS, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                            start_letter=START_LETTER, gpu=CUDA)
    # print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 15)
        testsamp = gen.sample(3).tolist()
        for sentence in testsamp:
            print(" ".join([language.index2word[word] for word in sentence]))

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 1, 3)
