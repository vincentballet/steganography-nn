import numpy as np
import random 
import math
import itertools
import os
import data
import pickle
import torch
import torch.nn as nn
import binascii
from fractions import gcd


#return true is word is ascii
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

#helper to load a pre-existing corpus and saving it
def load_corpus_and_save(a_corpus_name, ABS_PATH, a_data):
    if os.path.isfile(ABS_PATH + "save_corpus/corpus_{}.obj".format(a_corpus_name)):
        print("Loading the corpus")
        with open(ABS_PATH + "save_corpus/corpus_{}.obj".format(a_corpus_name), 'rb') as pickle_file:
            corpus = pickle.load(pickle_file)
    else:
        print("Creating the corpus")
        corpus = data.Corpus(a_data)
        filehandler = open(ABS_PATH + "save_corpus/corpus_{}.obj".format(a_corpus_name), 'wb')
        pickle.dump(corpus, filehandler)
    return corpus

#bin generator, load existing bin if exists and save_bin, and generates and saves it if doesn't
def generating_bins(ABS_PATH,  a_corpus_name, a_bins, a_common_bin_factor, a_replication_factor, a_seed,
                    a_num_tokens, a_save_bins, corpus):
    # if we want to load the bins and save it if it exists
    if os.path.isfile(
                    ABS_PATH + 'save_bins/corpus_name{}bin{}common_bin_factor{}replication_factor{}seed{}num_tokens{}.npz'.format(
                    a_corpus_name, a_bins, a_common_bin_factor, a_replication_factor, a_seed,
                    a_num_tokens)) and a_save_bins:
        print("Loading an existing bins model")
        loaded_model = np.load(
            ABS_PATH + 'save_bins/corpus_name{}bin{}common_bin_factor{}replication_factor{}seed{}num_tokens{}.npz'.format(
                a_corpus_name, a_bins, a_common_bin_factor, a_replication_factor, a_seed, a_num_tokens))
        bins = loaded_model['np_bins'].tolist()
        zero = loaded_model["np_zero"].tolist()
        common_tokens = loaded_model["np_common_tokens"].tolist()

        ###############################################################################
        # Generation of bins
    else:
        print("Creating the bins model")
        bins, zero, common_tokens = generate_bins(
            corpus=corpus,
            nbr_bins=a_bins,
            num_tokens=a_num_tokens,
            common_bin_factor=a_common_bin_factor,
            replication_factor=a_replication_factor,
            seed=a_seed,
            save_bins=a_save_bins,
            corpus_name=a_corpus_name,
            ABS_PATH = ABS_PATH
        )
    return bins, zero, common_tokens

def get_ordered_tokens(corpus):
    dictionary = corpus.dictionary.word_count
    d = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return d

def compute_bins_for_replicated_words(corpus, nbr_bins, nbr_tokens, replication_factor, seed):
    #Not sure if it's a good practice to put it here
    np.random.seed(seed)

    d = get_ordered_tokens(corpus)
    ordered_tokens = [item[0] for item in d]

    #Words already appear once, hence we subtract 1 to the replaction_factor
    rep_factor = replication_factor - 1

    non_common_tokens = ordered_tokens[nbr_tokens:]
    rep_factor_per_token = np.rint(np.linspace(0, rep_factor, len(non_common_tokens))).astype(int)

    return non_common_tokens, [np.random.choice(range(nbr_bins), size=rep, replace=False) for rep in rep_factor_per_token]

def get_common_tokens(corpus, n):
    d = get_ordered_tokens(corpus)
    common_tokens = [item[0] for item in d]
    common_tokens = common_tokens[0:n]
    return common_tokens + [':',",",'.','"','to','a','the','in','of','and','is']

#create the bins
def string2bins(bit_string, n_bins):
    n_bits = int(math.log(n_bins, 2))
    return [bit_string[i:i+n_bits] for i in range(0, len(bit_string), n_bits)]


def generate_bins(corpus, nbr_bins, num_tokens, common_bin_factor, replication_factor, seed, save_bins, corpus_name, ABS_PATH):
    if nbr_bins >= 2:
        np.random.seed(seed)
        sub_bin_indices = np.random.choice(range(nbr_bins), size=replication_factor, replace=False)
        common_bin_indices = np.random.choice(range(nbr_bins), size=common_bin_factor, replace=False)

        ntokens = len(corpus.dictionary)
        tokens = list(range(ntokens))

        random.shuffle(tokens)
        words_in_bin = math.ceil(len(tokens) / nbr_bins)

        # common words
        common_tokens = get_common_tokens(corpus, num_tokens)
        remove_words = ['<user>', 'rt','<eos>']
        common_tokens = list(set(common_tokens) - set(remove_words))
        common_tokens_idx = [corpus.dictionary.word2idx[word] for word in common_tokens]
        bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)] # words to keep in each bin...

        sub_bins = [bins[index] for index in sub_bin_indices]
        replicated_bin = list(itertools.chain(*sub_bins))  # just one bin

        bins = [replicated_bin if bins.index(bin_) in sub_bin_indices else bin_ for bin_ in bins]
        bins = [list(set(bin_) - set(common_tokens_idx)) if bins.index(bin_) in common_bin_indices else bin_ for bin_ in
                bins]
        zero = [list(set(tokens) - set(bin_)) for bin_ in bins]

    if save_bins:
        np_bins = np.asarray(bins)
        np_zero = np.asarray(zero)
        np_common_tokens = np.asarray(common_tokens)
        np.savez(ABS_PATH + 'save_bins/corpus_name{}bin{}common_bin_factor{}replication_factor{}seed{}num_tokens{}.npz'.format(corpus_name, nbr_bins, common_bin_factor, replication_factor, seed, num_tokens), np_bins=np_bins, np_zero=np_zero,np_common_tokens=np_common_tokens)
        print("Bins model saved")
    return bins, zero, common_tokens

def get_random_string(nbr_bins, nbr_words):
    return np.random.choice(range(nbr_bins), nbr_words)

def get_secret_text(filename, nbr_bins):
    secret_file = open(filename, 'r')
    secret_data = secret_file.read()
    #creating the string with the encoding of the letters and padding
    bit_string = ''.join(bin(ord(letter))[2:].zfill(8) for letter in secret_data)
    # secret_text = np.random.choice(range(args.bins), args.words)
    #an integer object from the given number in base 2
    secret_text = [int(i,2) for i in string2bins(bit_string, nbr_bins)]

    return secret_text

#get the next word output of the neural network
def get_next_word(input, word_weights, corpus):
    word_idx = torch.multinomial(word_weights, 1)[0]

    input.data.fill_(word_idx)
    word = corpus.dictionary.idx2word[word_idx]
    valid_word = is_ascii(word)

        #iterating to get a valid word
    while not valid_word:
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        valid_word = is_ascii(word)

    return word

#return the least commum multiple
def lcm(a, b):
    return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

#convert some ascii text in corresponding bitstring
def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def get_removal_len(all_decoded_strings,bin_len,step):
    length = len(all_decoded_strings[0]) * int(bin_len)
    to_remove = int(length - (math.floor(length / step) * step))
    return length, to_remove

def remove_padding(bin_len, to_remove, bitstring, a_bins):
    to_readd = int(bin_len) - to_remove
    to_save = bitstring[-a_bins:]
    bitstring = bitstring[:- int(bin_len)]
    bitstring = bitstring + to_save[-to_readd:]
    return bitstring

def join_character_from_bitstring(bitstring,idx,step):
    return chr(int(''.join(bitstring)[idx:idx+step], base=2))

def pre_process_received_string(encoded_data):
    return encoded_data[:-1]