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

from tools.rnn_language_training import data
from torch.autograd import Variable

sys.path.append(".")

def run(parser=None, args_dic=None, plaintext=None):
    
    if parser:
        args = parser.parse_args()
        print("Running code from terminal " + "-" * 100)
        ABS_PATH = os.path.abspath(".") + '/'
        a_data = args.data
        a_checkpoint = args.checkpoint
        a_cuda = args.cuda 
        a_words = args.words
        a_temperature =  args.temperature
        a_bins = args.bins
        a_common_bin_factor = args.common_bin_factor
        a_num_tokens = args.num_tokens
        a_secret_file = args.secret_file
        a_outf = args.outf
        a_replication_factor = args.replication_factor
        a_seed = args.seed
        a_random = args.random
        a_log_interval = args.log_interval
        a_save_corpus = args.save_corpus
        a_save_bins = args.save_bins
        a_corpus_name = args.corpus_name
        if args.temperature < 1e-3:
            parser.error("--temperature has to be greater or equal 1e-3")

    elif args_dic:
        
        print("Running code from code " + "+" * 100)
        ABS_PATH = "/home/ballet/steganography-nn/"
        a_data = args_dic['data']
        a_checkpoint = args_dic['checkpoint']
        a_cuda = args_dic['cuda']
        a_words = args_dic['words']
        a_temperature =  args_dic['temperature']
        a_bins = args_dic['bins']
        a_common_bin_factor = args_dic['common_bin_factor']
        a_num_tokens = args_dic['num_tokens']
        a_secret_file = args_dic['secret_file']
        a_outf = args_dic['outf']
        a_replication_factor = args_dic['replication_factor']
        a_seed = args_dic['seed']
        a_random = args_dic['random']
        a_log_interval = args_dic['log_interval']
        a_save_corpus = args_dic['save_corpus']
        a_save_bins = args_dic['save_bins']
        a_corpus_name = args_dic['corpus_name']

        if a_temperature < 1e-3:
            print("temperature has to be greater or equal 1e-3")

        with open(a_secret_file, 'w') as myfile:
            myfile.write(plaintext)  # python will convert \n to os.linesep

     
    #starting the counter
    epoch_start_time = time.time()

    torch.nn.Module.dump_patches = True

    # Set the random seed manually for reproducibility.
    torch.manual_seed(a_seed)
    # Get the same bins ordering as in the generting part
    random.seed(a_seed)
    if torch.cuda.is_available():
        if not a_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(a_seed)

    #loading the pre-trained model
    with open(a_checkpoint, 'rb') as f:
        model = torch.load(f)

    if a_cuda:
        model.cuda(0)
    else:
        model.cpu()

    #if we want to save the corpus and load the model if it exists
    if a_save_corpus:
        corpus = process.load_corpus_and_save(a_corpus_name, ABS_PATH, a_data)
    else:
        corpus = data.Corpus(a_data)

    #building the input model
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

    if a_cuda:
        input.data = input.data.cuda(0)

    # Secret Text Modification
    if a_random:
        secret_text = process.get_random_string(a_bins, a_words)
    else:
        secret_text = process.get_secret_text(a_secret_file, a_bins)

    if a_bins > 1:
        #if we want to load the bins and save it if it exists
        bins, zero, common_tokens = process.generating_bins(ABS_PATH, a_corpus_name, a_bins, a_common_bin_factor, a_replication_factor, a_seed,
                        a_num_tokens, a_save_bins, corpus)


        print('Finished Generating Bins')
        print('Time: {:5.2f}s'.format(time.time() - epoch_start_time))
        ###############################################################################

        # Generation of stegotext
        print('Generating Stegotext')

        with open(a_outf, 'w') as outf:
            w = 0 
            i = 1
            bin_sequence_length = len(secret_text[:]) # 85

            while i <= bin_sequence_length:
                output, hidden = model(input, hidden)

                #building the Tensor with our bins
                zero_index = zero[secret_text[:][i-1]]
                zero_index = torch.LongTensor(zero_index)

                word_weights = output.squeeze().data.div(a_temperature).exp().cpu()

                #in case the bin contains all the words, don't constrain
                if(len(zero_index)>0):
                    word_weights.index_fill_(0, zero_index, 0)

                #get the next word
                word = process.get_next_word(input, word_weights, corpus)

                word = word.encode('ascii', 'ignore').decode('ascii')
                if w > 0: outf.write(' ')
                outf.write(word)

                if word not in common_tokens:
                    i += 1
                w += 1

                if i % a_log_interval == 0:
                    print("Total number of words", w)
                    print("Total length of secret", i)
                    print('| Generated {}/{} words'.format(i, len(secret_text)))


        with open(a_outf, 'r') as out:
            print(out.read())
    
        print('Time: {:5.2f}s'.format(time.time() - epoch_start_time))
        print('Finished Generating Stegotext')
        print('-' * 89)


#Ran from terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    parser.add_argument('--secret_file', type=str, default='./demo/secret_file.txt',
                        help='location of the secret text file')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    parser.add_argument('--bins', type=int, default=2,
                        help='number of word bins')
    parser.add_argument('--replication_factor', type=int, default=1,
                        help='number of bins each word appears')
    parser.add_argument('--common_bin_factor', type=int, default=0,
                        help='how many bins to add common words')
    parser.add_argument('--num_tokens', type=int, default=0,
                        help='adding top freq number of words to each bin')
    parser.add_argument('--random', action='store_true',
                        help='use randomly generated sequence')
    parser.add_argument('--corpus_name', default='big_tweets')
    parser.add_argument('--save_corpus', action='store_true', default=False)
    parser.add_argument('--save_bins', action='store_true', default=False)

    run(parser=parser)