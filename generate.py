###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import time
import math
import random
import numpy as np
import itertools
import sys
import os
import pickle
import torch
import torch.nn as nn
import tools.process as process
import tools.parser as parser
from tools.rnn_language_training import data
from torch.autograd import Variable

sys.path.append(".")

def run(p=None, args_dic=None, plaintext=None):
    
    if p:
        print("Running code from terminal " + "-" * 100)
        ABS_PATH = os.path.abspath(".") + '/'
        args = p.parse_args()
        args_dic = parser.args_to_dic(args)
        if args.temperature < 1e-3:
            p.error("--temperature has to be greater or equal 1e-3")
        with open(args_dic['secret_file'], 'r') as myfile:
            secret_text = myfile.read() 

    elif args_dic:
        
        print("Running code from code " + "+" * 100)
        ABS_PATH = "/home/ballet/steganography-nn/"
        if args_dic['temperature'] < 1e-3:
            print("temperature has to be greater or equal 1e-3")
        secret_text = plaintext

    # Couldn't find a way to decode it afterwards so simply removing it.
    secret_text = secret_text.replace('\r', '')
    print("Secret text is '{}'".format(secret_text))

    #starting the counter
    epoch_start_time = time.time()

    torch.nn.Module.dump_patches = True

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args_dic['seed'])
    # Get the same bins ordering as in the generting part
    random.seed(args_dic['seed'])
    if torch.cuda.is_available():
        if not args_dic['cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args_dic['seed'])

    #loading the pre-trained model
    with open(args_dic['checkpoint'], 'rb') as f:
        model = torch.load(f)

    if args_dic['cuda']:
        model.cuda(0)
    else:
        model.cpu()

    #if we want to save the corpus and load the model if it exists
    if args_dic['save_corpus']:
        corpus = process.load_corpus_and_save(args_dic['corpus_name'], ABS_PATH, args_dic['data'])
    else:
        corpus = data.Corpus(args_dic['data'])

    #building the input model
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    #initializing the neural network to a new tweet
    previous_word = '<eos>'
    word_idx= corpus.dictionary.word2idx[previous_word]
    input.data.fill_(word_idx)

    if args_dic['cuda']:
        input.data = input.data.cuda(0)

    # Secret Text Modification
    if args_dic['random']:
        secret_text = process.get_random_string(args_dic['bins'], args_dic['words'])
    else:
        secret_text = process.get_secret_text(secret_text, args_dic['bins'])

    if args_dic['bins'] > 1:
        #if we want to load the bins and save it if it exists
        bins, zero, common_tokens = process.generating_bins(ABS_PATH, args_dic['corpus_name'], args_dic['bins'], args_dic['common_bin_factor'], args_dic['replication_factor'], args_dic['seed'],
                        args_dic['num_tokens'], args_dic['save_bins'], corpus)


        print('Finished Generating Bins')
        print('Time: {:5.2f}s'.format(time.time() - epoch_start_time))
        ###############################################################################

        # Generation of stegotext
        print('Generating Stegotext')

        stegotext = []
        w = 0 
        i = 1
        bin_sequence_length = len(secret_text[:]) # 85

        while i <= bin_sequence_length:
            output, hidden = model(input, hidden)

            #building the Tensor with our bins
            zero_index = zero[secret_text[:][i-1]]
            zero_index = torch.LongTensor(zero_index)

            word_weights = output.squeeze().data.div(args_dic['temperature']).exp().cpu()

            #in case the bin contains all the words, don't constrain
            if(len(zero_index)>0):
                word_weights.index_fill_(0, zero_index, 0)

            #get the next word
            word = process.get_next_word(input, word_weights, corpus)

            word = word.encode('ascii', 'ignore').decode('ascii')
            stegotext.append(word)

            if word not in common_tokens:
                i += 1
            w += 1

            if i % args_dic['log_interval'] == 0:
                print("Total number of words", w)
                print("Total length of secret", i)
                print('| Generated {}/{} words'.format(i, len(secret_text)))

        stegotext_str = ' '.join(stegotext)
        with open(args_dic['outf'], 'w') as outf:
            outf.write(stegotext_str)

        print('Time: {:5.2f}s'.format(time.time() - epoch_start_time))
        print('Finished Generating Stegotext')
        print('-' * 89)
        return stegotext_str

#Ran from terminal
if __name__ == '__main__':
    p = parser.get_parser()
    decoded = run(p=p)
    print("Generated stegotext : {}".format(decoded))