# Steganography LSTM

Code for the paper [Generating Steganographic Text with LSTMs](https://arxiv.org/abs/1705.10742). The LSTM is based on the [Word Language Model](https://github.com/pytorch/examples/tree/master/word_language_model) example from PyTorch (http://pytorch.org/).

## Requirements

- Latest [NVIDIA driver](http://www.nvidia.com/Download/index.aspx)
- [CUDA 8 Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [PyTorch](https://github.com/pytorch/pytorch#installation)

## Data
A small sample of Penn Treebank and Tweets. `pre-process.py` is tokenization of punctuation.

## Training
` python main.py --cuda --nhid 600 --nlayers 3 --epochs 6 --data './data/tweets' --save './models/twitter-model.pt' `
For the full list of arguments, check the [PyTorch example README](https://github.com/pytorch/examples/tree/master/word_language_model).

## Text Generation
One of our key original contributions. After we train our model, we generate words and restrict the output based on the secret text. `generate.py` is modified such that it takes the secret text and modifies the probabilities based on the "bins" as described in our paper.
If replication factor is bigger than 1, the encoding/decoding is non-deterministic.

Example generation with 4 bins: 
` python generate.py --data './data/tweets' --checkpoint './models/twitter-model.pt' --cuda --words 1000 --temperature 0.8 --bins 8 --common_bin_factor 4 --num_tokens 20 --secret_file './demo/secret_file.txt' --outf './outputs/stegotext.txt' --replication_factor 3 --save_bins --save_corpus `
add --random if non existing './demo/secret_file.txt'

See the arguments in `generate.py` or refer to the [PyTorch example README](https://github.com/pytorch/examples/tree/master/word_language_model).

## Stegotext decoding
Given that we know the seed that was used to generate to stegotext, we can decode it.

` decoder.py --data './data/tweets' --checkpoint './models/twitter-model.pt' --cuda --bins 8 --model_char_nn './tools/rnn_char/models/tinyshakespeare.pt' --encoded_file './outputs/stegotext.txt' --replication_factor 3 --save_corpus --next_character 10 --save_bins --num_tokens 20 --common_bin_factor 4 `

See the arguments in `decoder.py`

## Credits
This code is based on tbfang/steganography-lstm.
The character-level rnn is based on [Practical PyTorch: Generating Shakespeare with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)