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
import os
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools.rnn_language_training import data

import zlib
import io
from tools import process
import tools.parser as parser
from tools.rnn_char import helpers
from tools.rnn_char import possible_strings
from tools.rnn_char import generate
from tools.rnn_char.helpers import *
from tools.rnn_char.model_char.model import *


def onlyascii(char):
    return (ord(char) > 0 and ord(char) < 127) or char==' ' or char=='?'

def run(p=None, args_dic=None, encoded_text=None):
    epoch_start_time = time.time()
    
    if p:
        print("Running code from terminal " + "-" * 100)
        ABS_PATH = os.path.abspath(".") + '/'
        args = p.parse_args()
        args_dic = parser.args_to_dic(args)
        if args.temperature < 1e-3:
            p.error("--temperature has to be greater or equal 1e-3")
        with open(args_dic['encoded_file'], 'r') as myfile:
            encoded_data = myfile.read()

    elif args_dic:
        print("Running code from code " + "+" * 100)
        ABS_PATH = "/home/ballet/steganography-nn/"
        if args_dic['temperature'] < 1e-3:
            print("temperature has to be greater or equal 1e-3")
        encoded_data = encoded_text
        
    print("Stegotext is '{}'".format(encoded_data))

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


    with open(args_dic['checkpoint'], 'rb') as f:
        model = torch.load(f)

    if args_dic['cuda']:
        model.cuda()
    else:
        model.cpu()

    #if we want to save the corpus and load the model if it exists
    if args_dic['save_corpus']:
        corpus = process.load_corpus_and_save(args_dic['corpus_name'], ABS_PATH, args_dic['data'])
    else:
        corpus = data.Corpus(args_dic['data'])

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

    if args_dic['cuda']:
        input.data = input.data.cuda()


    ###############################################################################


    ###############################################################################
    # Stegotext decoding
    def read_and_decompress(b_io):
        d = zlib.decompressobj(16+zlib.MAX_WBITS)
        buffer = b_io.getbuffer()
        return [d.decompress(bytes([b])) for b in buffer]

    #method to decode when replication_factor is 1
    def decode_simple(all_decoded_strings):
        #removing padding
        total_len, to_remove = process.get_removal_len(all_decoded_strings, bin_len, step)

        for tup in all_decoded_strings:
            output_text = ""
            bitstring = ''.join(tup)

            #handling padding
            if to_remove > 0:
                to_re_add = int_bin_len - to_remove
                to_save = bitstring[-args_dic['bins']:]
                bitstring = bitstring[:- int_bin_len]
                bitstring = bitstring + to_save[-to_re_add:]

            #appending the decoded char to get the final string
            for i in range(0, len(bitstring), step):
                char_bits = ''.join(bitstring[i:i+step]) # e.g. 01001101
                char_bytes = chr(int(char_bits, base=2))
                output_text = output_text + char_bytes

        return output_text

    #method to decode when replication factor > 1
    def decode_replicated(all_decoded_strings,step,previous,start):

        def get_ascii_set(idx, all_decoded_strings_tmp):
            possible_letters = set([process.join_character_from_bitstring(tup,idx,step) for tup in all_decoded_strings_tmp])
            # keeping only ascii
            possible_letters = [l for l in possible_letters if onlyascii(l)]
            return possible_letters
        
        #handling padding
        length, to_remove = process.get_removal_len(all_decoded_strings, bin_len, step)
        length_limit_before_fixing_padding = length - step - to_remove
        finishing = False


        #computing the number of tupples needed to represent a char
        index_tupple = math.ceil(step/int(np.log2(args_dic['bins'])))

        sets_of_letters = []

        #as we use a recursive algo, we need to rebuilt the previous string to give it as initialization for the nn char
        if len(previous) > 0:
            for char in list(previous):
                sets_of_letters.append(list(char))
            init=0

        else:
            #Getting the first chars set to initialize the nn
            possible_letters = get_ascii_set(0,all_decoded_strings)
            sets_of_letters.append(list(possible_letters))
            init = step

        #handing the general cases before padding
        for idx in range(init,length-to_remove,step):
            if idx >= length_limit_before_fixing_padding:
                finishing = True
                set_char = set()
                for tup in all_decoded_strings:
                    bitstring = ''.join(tup)
                    if to_remove > 0:
                        bitstring = process.remove_padding(bin_len, to_remove, bitstring, args_dic['bins'])
                    set_char.add(process.join_character_from_bitstring(bitstring,idx,step))
                sets_of_letters.append(list(set_char))
            else:
                #generating the possible char for this index
                possible_letters = get_ascii_set(idx,all_decoded_strings)
                #if the wrong path was taken and the decoding is not ascii, return with error
                if len(possible_letters)== 0:
                    return "error_decoding"
                sets_of_letters.append(list(possible_letters))
                #incrementing the index for the tupples
                index_tupple = math.ceil((idx+step)/int_bin_len)

            #generating the cartesian product of the possible letters combinations
            combinaison_letter = list(itertools.product(*sets_of_letters))

            #getting the index of  most probable combination if there is more than one combinations
            if len(combinaison_letter) > 1:
                index = possible_strings.next_letters_table(args_dic['model_char_nn'],combinaison_letter,args_dic['next_character'], False)[0]
            else:
                index = 0

            if finishing :
                return ''.join(combinaison_letter[index])
            else:
                #appending the founded letters
                sets_of_letters = [list(l) for l in combinaison_letter[index]]

                #keeping only the string starting with the right tupples
                possible_letters_in_ascci = (process.text_to_bits(''.join(combinaison_letter[index])))[(start*step):]
                #keeping only the bit string starting with the right chars
                all_decoded_strings = [tup for tup in all_decoded_strings if (''.join(tup[0:index_tupple])[:idx+step]) == possible_letters_in_ascci]


    def decode_simple_gzipped(all_decoded_strings):
        # Number of stegowords needed to encode a character in ascii
        LEN_HEADER = 11
        NBR_STEGOWORDS_PER_CHAR = (8 / np.log2(args_dic['bins']))
        NBR_IGNORED_STEGOWORDS = int(LEN_HEADER * NBR_STEGOWORDS_PER_CHAR)

        # This is a generic header for our machine. See http://www.onicos.com/staff/iz/formats/gzip.html
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03s'

        # Behaves like a classic input/output file buffer
        zipped_io = io.BytesIO()

        # We don't need to spend time decoding the 11 first steogwords because it is gonna be the header of the zipped file
        zipped_io.write(header)

        # Iterate over all possible decoded stegotexts
        for tup in all_decoded_strings:
            bitstring = ''.join(tup)
            #Lowerbound : as explained 5 lines above
            for i in range(NBR_IGNORED_STEGOWORDS, len(tup), step):
                char_bits = ''.join(bitstring[i:i+step]) # e.g. 01001101
                char_bytes = bytes(int(char_bits, base=2).to_bytes(len(char_bits) // 8, byteorder='big'))  # e.g. b'\xbc'
                zipped_io.write(char_bytes)
                # DEBUG print("DATA {0:03d} {char_bits} => Up To Now => {character}".format(int(i), char_bits = char_bits, character=char_bytes))


                dec = read_and_decompress(zipped_io) # e.g. b'Kiwis'

        print("End of the program. The original message was : {}".format(b''.join(read_and_decompress(zipped_io))))
        zipped_io.close() # discard buffer memory


    def uncompress_zipped(zipped_io, combinations, LEN_HEADER):
        #DEBUG print("\tleft = {}, right = {}".format(combinations[0], combinations[1]))

        # Deep copy of the header to a new buffer
        tmp_zipped_io = io.BytesIO(zipped_io.getvalue())

        # Generate all the possible combinations
        comb = list(itertools.product(*combinations))
        comb = [b''.join(tup) for tup in comb]

        #DEBUG  print("\tCombinations : {}".format(comb))
        dec_comb = []
        for c in comb:
            #Skips header
            tmp_zipped_io.seek(11)
            tmp_zipped_io.write(c)
            dec = read_and_decompress(tmp_zipped_io)[LEN_HEADER:]
            #DEBUG print("\t{} goes with {}".format(c, dec))
            dec_comb.append(dec)

        tmp_zipped_io.close()
        return comb, dec_comb


    def decode_replicated_gzipped(all_decoded_strings):
        #Number of stegowords needed to encode a character in ascii
        LEN_HEADER = 11
        LEN_FOOTER = 8
        NBR_STEGOWORDS_PER_CHAR = (8 / np.log2(args_dic['bins']))
        NBR_IGNORED_STEGOWORDS = int(LEN_HEADER * NBR_STEGOWORDS_PER_CHAR)
        NBR_IGNORED_BITS = LEN_HEADER * int(np.log2(args_dic['bins']))
        #This is a generic header for our machine. See http://www.onicos.com/staff/iz/formats/gzip.html
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03s'

        #Behaves like a classic input/output file buffer
        zipped_io = io.BytesIO()

        #We don't need to spend time decoding the 11 first steogwords because it is gonna be the header of the zipped file
        zipped_io.write(header)

        LENGTH = len(all_decoded_strings[0]) * int(np.log2(args_dic['bins']))

        sets_of_letters = []
        print("{}, {}".format(LENGTH, step))

        #Todo handle header in range
        possible_letters = []
        for idx in range(0, LENGTH, step):
            # Note : If bitstring length is not a multiple of 8 (len ascii char) then 8 // 8 should be replaced by len(''.join(tup)[idx:idx+step] // 8)
            possible_letters = set(
                [bytes(int(''.join(tup)[idx:idx + step], base=2).to_bytes(8 // 8, byteorder='big')) for tup in
                all_decoded_strings])  # e.g. b'\xbc'
            sets_of_letters.append(list(possible_letters))

        # Removed header
        sets_of_letters = sets_of_letters[11:]
        print("{}, {}\n\n".format(len(sets_of_letters), sets_of_letters))

        best_combination = []
        try:
            while len(sets_of_letters) > LEN_FOOTER + 3:
                # Unzip together the first two sets of letters
                compr_combs, combinations = uncompress_zipped(zipped_io, sets_of_letters[0:2], LEN_HEADER)
                combinations = [(b''.join(tup)).decode() for tup in combinations]
                print("Combinations => {}".format(combinations))

                # If there is more than one possibility, ask the nn to output the best one
                if len(combinations) > 1:
                    oredered_combinations = possible_strings.next_letters_table(args_dic['model_char_nn'], combinations, args_dic['next_character'])
                    if len(oredered_combinations) > 0:
                        best_combination = oredered_combinations[0]
                    else:
                        raise StopIteration
                else:
                    best_combination = combinations[0]
                print("Most likey = {}".format(best_combination))

                last = compr_combs[combinations.index(best_combination)]
                sets_of_letters = sets_of_letters[2:]
                sets_of_letters.insert(0, [last])
                # DEBUG print("Removed first and prepend solution  {}".format(sets_of_letters))

        except StopIteration:
            print("Aborting. Couldn't predict next letter")

        print("End of the program. The original message was : {}".format(best_combination))
        zipped_io.close()  # discard buffer memory

    def recursion_decode(encoded_words_bins, previous, start):
        lcm_step_bin = (int(process.lcm(step, bin_len) / bin_len)) * 2

        if args_dic['replication_factor'] > 1:
            if not args_dic['compressed']:
                # We need to split recursively in order for the cartesian product to be possible
                if len(encoded_words_bins) > (lcm_step_bin + math.ceil(step / bin_len)):
                    #decode the first part
                    decode_replicated_string = decode_replicated(list(itertools.product(*encoded_words_bins[:lcm_step_bin])), step, previous, start)
                    if decode_replicated_string == "error_decoding":
                        return decode_replicated_string
                    #call recursively the remaining part
                    to_return = recursion_decode(encoded_words_bins[lcm_step_bin:],decode_replicated_string,int((start+((lcm_step_bin*bin_len)/step))))
                else:
                    to_return = decode_replicated(list(itertools.product(*encoded_words_bins)), step, previous, start)
            else:
                decode_replicated_gzipped(all_decoded_strings)
        else:
            if not args_dic['compressed']:
                to_return = decode_simple(list(itertools.product(*encoded_words_bins)))
            else:
                decode_simple_gzipped(all_decoded_strings)
        return to_return

    if args_dic['bins'] > 1:
        bins, zero, common_tokens = process.generating_bins(ABS_PATH, args_dic['corpus_name'], args_dic['bins'], args_dic['common_bin_factor'], args_dic['replication_factor'], args_dic['seed'],
                        args_dic['num_tokens'], args_dic['save_bins'], corpus)

        #extracting the data tokens
        encoded_data = encoded_data.replace("rt <user> : "," ")
        encoded_data_words = encoded_data.split()
        encoded_data_tokens = [corpus.dictionary.word2idx[w] for w in encoded_data_words]

        #Should be removed along with common tokens
        common_tokens_idx = [corpus.dictionary.word2idx[word] for word in common_tokens]

        # Len of ascii character
        step = 8
        # Len of bin in bits
        bin_len = math.log(args_dic['bins'], 2)
        # integer bin len
        int_bin_len = int(bin_len)

        # Infer the bins that we used during the generate part of the algorithm
        encoded_words_bins = []
        for token in encoded_data_tokens:
            bins_for_token = []
            for idx, bin_ in enumerate(bins):
                if token in bin_ and token not in common_tokens_idx:
                    bins_for_token.append("{0:0{bit_len}b}".format(idx, bit_len=int_bin_len))
            if len(bins_for_token) > 0:
                encoded_words_bins.append(bins_for_token)



        start_time = time.time()
        #recursively trying to decode
        final_solution= recursion_decode(encoded_words_bins,'',0)

        #while the wrong path is taken, start again
        while final_solution == "error_decoding":
            final_solution = recursion_decode(encoded_words_bins, '', 0)

        print("final solution : {}".format(final_solution))

        print('Time: {:5.2f}s'.format(time.time() - epoch_start_time))
        return final_solution
        ###############################################################################


#Ran from terminal
if __name__ == '__main__':
    p = parser.get_parser()
    decoded = run(p=p)
    print("Decoded text : {}".format(decoded))