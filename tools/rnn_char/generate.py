#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse
import numpy
import warnings
import sys
import torch.nn as nn
from tools.rnn_char import helpers
from model_char import model
from torch.autograd import Variable

warnings.simplefilter("ignore", UserWarning)

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False, letters_number=10, reaching_end=False):
    global output_dist
    hidden = decoder.init_hidden(1)
    prime_input = Variable(helpers.char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()

        possible_char = set([])
        found = 0
        while found < letters_number:
            tmp = torch.multinomial(output_dist, 1, replacement=False)[0]
            # Add predicted character to string and use as next input
            predicted_char = helpers.all_characters[tmp]
            if(predicted_char not in possible_char):
                possible_char.add(predicted_char)
            found += 1
            inp = Variable(helpers.char_tensor(predicted_char).unsqueeze(0))
            if cuda:
                inp = inp.cuda()
        if reaching_end:
            possible_char.add(".")
            possible_char.add("\n")
        return possible_char

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--letters_number', type=int, default=10)
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))

