#!/usr/bin/env python

import torch
from tools.rnn_char import generate
import os

#construct a table that give the potential next char for the input depending on the nn
def next_letters_table(model,combinations, number_of_next_letters, reaching_end):
    global output_dist
    possible_combinations = []
    next_letter_for_letters = {}

    for idx, comb in enumerate(combinations):
        #extracting the initializing part of the string
        string_begining = (comb)[:-1]

        possible_letters = next_letter_for_letters.get(string_begining)
        # if we have already generated the next char for this initialization value, retrieve it. If not, we generate it
        if possible_letters is None:
            with open(model, 'rb') as f:
                decoder = torch.load(f)
            possible_letters = generate.generate(decoder, list(string_begining), 100, 0.8, True, number_of_next_letters, reaching_end)
            next_letter_for_letters[string_begining] = possible_letters
        #if this combinations is given as probable by the nn, append it to the possible combinations
        last_char = list(comb)[-1]
        if last_char in possible_letters:
            possible_combinations.append(idx)
    #returning the list of possible combinations
    if len(possible_combinations) > 0:
        return possible_combinations
    else :
        if number_of_next_letters > 100:
            reaching_end = True
        return next_letters_table(model, combinations, number_of_next_letters * 2, reaching_end)

#if __name__ == '__main__':
#    next_letters_table("./tools/rnn_char/models/tinyshakespeare.pt",['This is life','This is lifr'],2)