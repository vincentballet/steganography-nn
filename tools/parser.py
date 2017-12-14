import argparse

def get_parser():
    p = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    p.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    p.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    p.add_argument('--encoded_file', type=str, default='./results/stegotweets.txt',
                        help='location of the encoded text file')
    p.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    p.add_argument('--secret_file', type=str, default='./demo/secret_file.txt',
                            help='location of the secret text file')
    p.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    p.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    p.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    p.add_argument('--temperature', type=float, default=0.8,
                        help='temperature - higher will increase diversity')
    p.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    p.add_argument('--bins', type=int, default=2,
                        help='number of word bins')
    p.add_argument('--replication_factor', type=int, default=1,
                        help='number of bins each word appears')
    p.add_argument('--common_bin_factor', type=int, default=0,
                        help='how many bins to add common words')
    p.add_argument('--num_tokens', type=int, default=0,
                        help='adding top freq number of words to each bin')
    p.add_argument('--random', action='store_true',
                        help='use randomly generated sequence')
    p.add_argument('--compressed', action='store_true',
                        help='stegotext has been compressed using gzip')
    p.add_argument('--model_char_nn', type=str,
                        help='which model the character neural network should be train on')
    p.add_argument('--corpus_name', default='big_tweets')
    p.add_argument('--save_corpus', action='store_true', default=False)
    p.add_argument('--save_bins', action='store_true', default=False)
    p.add_argument('--next_character', type=int, default=10)
    p.add_argument('--ascii_only', action='store_true',default=True)
    p.add_argument('--lower_case_only', action='store_true',default=True)
    p.add_argument('--spellcheck', action='store_true',default=True)

    return p

def args_to_dic(args_from_main):
    args_dic = {'data': args_from_main.data,
        'checkpoint': args_from_main.checkpoint,
        'model_char_nn': args_from_main.model_char_nn,
        'cuda': args_from_main.cuda,
        'words': args_from_main.words,
        'temperature': args_from_main.temperature,
        'bins': args_from_main.bins,
        'common_bin_factor': args_from_main.common_bin_factor,
        'num_tokens': args_from_main.num_tokens,
        'outf': args_from_main.outf,
        'replication_factor': args_from_main.replication_factor,
        'encoded_file': args_from_main.encoded_file,
        'seed': args_from_main.seed,
        'secret_file': args_from_main.secret_file,
        'random': args_from_main.random,
        'log_interval': args_from_main.log_interval,
        'save_corpus': args_from_main.save_corpus,
        'save_bins': args_from_main.save_bins,
        'corpus_name': args_from_main.corpus_name,
        'compressed': args_from_main.compressed,
        'next_character': args_from_main.next_character,
        'lower_case_only' : args_from_main.lower_case_only,
        'ascii_only': args_from_main.ascii_only,
        'spellcheck':args_from_main.spellcheck}

    if args_from_main.temperature < 1e-3:
            parser.error("--temperature has to be greater or equal 1e-3")

    return args_dic