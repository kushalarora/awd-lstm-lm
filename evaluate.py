###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
import Levenshtein as Lev
from torch.autograd import Variable

import data
import sys
import os
import math

from utils import batchify, get_batch, repackage_hidden
from Bleu import score_corpus
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--distorted_file', type=str, default='test.txt.deformed-80.0-10.0-10.0',
                    help='distorted file for test.txt.')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--max_vocab', type=int, default=sys.maxsize, help='Maximum Vocab to consider.')
parser.add_argument('--nbest_in', type=str, default='data/nbest_list_example_caps.txt',
                    help='NBest input list.')
parser.add_argument('--nbest_out', type=str, default='data/nbest_list_example_output.txt',
                    help='NBest output list.')
parser.add_argument('--nbest', type=int, default=200, help='Number of nbest to consider.')
parser.add_argument('--lower', action='store_true', help='Lower the sentences.')
parser.add_argument('--report_wer', action='store_true', help='Report WER.')
parser.add_argument('--report_bleu', action='store_true', help='Report BLEU.')
parser.add_argument('--oov_penalty', action='store_true', help='OOV Penality for nbest list scoring.')
args = parser.parse_args()

args.bptt = 140

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

global model, criterion, optimizer
with open(args.checkpoint, 'rb') as f:
    model, criterion, optimizer = torch.load(f)
print("Using Model: %s\n" % args.checkpoint)
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

import hashlib
print('Using Dataset: (%s) with %s vocab' % (args.data, ('FULL' if args.max_vocab == sys.maxsize else args.max_vocab)))
fn = 'corpus.{}.data'.format(hashlib.md5((args.data + '-' + str(args.max_vocab)).encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data, args.max_vocab)
    torch.save(corpus, fn)

print('Dataset: %s' % args.data)
print('Vocab Size: %d' % len(corpus.dictionary))

ntokens = len(corpus.dictionary)
test_batch_size = 1

def sent_score(sent):
    model.eval()
    if args.model == 'QRNN': model.reset()
    hidden = model.init_hidden(1)
    data = Variable(sent[:-1].unsqueeze(1), volatile=True)
    target = Variable(sent[1:].view(-1))
    output, hidden = model(data, hidden)
    loss = float(criterion(model.decoder.weight, model.decoder.bias, output, target).data)
    if args.oov_penalty:
        loss += sum(map(lambda x: 1 if corpus.dictionary.word2idx[corpus.dictionary.unk_token] == x else 0, sent.tolist()))
    return loss

def entropy(data_source, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0]

def log_perplexity(data_source, batch_size=1):
    return entropy(data_source, batch_size)/len(data_source)

def contrastive_log_perplexity(test, test_distorted, true_entropy=None):
    true_entropy = true_entropy or entropy(test)
    distorted_entropy = entropy(test_distorted)
    return (distorted_entropy - true_entropy)/len(test)

def nbest_score(nbest_list_input, nbest_list_output,
                lower=False, report_wer=False, report_bleu=False):
    nbest_all=[]
    with open(nbest_list_input, 'r') as input, \
            open(nbest_list_output, 'w') as output:

        size=len(input.readlines())
        input.seek(0)
        for sentences in input: # tqdm(input, total=size):
            nbest_all.append([])
            for idx, sentence in enumerate(sentences.split('\t')[:args.nbest]):
                if len(sentence.split()) < 2:
                    if idx == 0:
                        del nbest_all[-1]
                        break

                    continue

                if lower:
                    sentence = sentence.lower()
                tokenize_sent = corpus.tokenize_sent(sentence)
                if args.cuda:
                    tokenize_sent = tokenize_sent.cuda()
                score = sent_score(tokenize_sent)
                nbest_all[-1].append((idx, sentence, score))

        nbest_sorted = []
        for nbest_batch in nbest_all:
            nbest_sorted.append([nbest_batch[0]] + sorted(nbest_batch[1:], key=lambda x : x[2]))

        for nbest_batch in nbest_sorted:
            for hypo in nbest_batch:
                output.write('%d %s %6.3f\n' % hypo)

        if report_wer:
            original_total_wer = 0.0
            original_num_tokens = 0
            for nbest_batch in nbest_all:
                 original_total_wer += sent_wer(nbest_batch[0][1], nbest_batch[1][1])
                 original_num_tokens += len(nbest_batch[0][1].split())

            total_wer = 0.0
            num_tokens = 0.0
            for nbest_batch in nbest_sorted:
                 total_wer += sent_wer(nbest_batch[0][1], nbest_batch[1][1])
                 num_tokens += len(nbest_batch[0][1].split())

            print('$' * 89)
            print('Original WER: %5.2f || Reranked Average WER: %5.2f' % (original_total_wer*100/original_num_tokens, total_wer*100/num_tokens))
            print('$' * 89)

        if report_bleu:
            original_bleu = score_corpus([nbest_all[x][0][1] for x in range(len(nbest_all))],
                                         [nbest_all[x][1][1] for x in range(len(nbest_all))], 4)

            bleu = score_corpus([nbest_sorted[x][0][1] for x in range(len(nbest_sorted))],
                                [nbest_sorted[x][1][1] for x in range(len(nbest_sorted))], 4)

            print('$' * 89)
            print("Original BLEU Score: %5.2f || Reranked BLEU WER: %5.2f " % (original_bleu, bleu))
            print('$' * 89)

def sent_wer(s1, s2):
    """ Computes the Word Error Rate, defined as the edit distance between the
	two provided sentences after tokenizing to words.
	Arguments:
	    s1 (string): space-separated sentence
	    s2 (string): space-separated sentence
    """
    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

test_data = batchify(corpus.test, 1, args)
test_data_distorted = batchify(corpus.tokenize(os.path.join(args.data, args.distorted_file)), 1, args)

original_ent = log_perplexity(test_data, test_batch_size)
print('=' * 100)
print('|  Cross Entropy {:5.2f} | Perplexity {:8.2f}'.format(original_ent, math.exp(original_ent)))
print('=' * 100)

contrastive_ent = contrastive_log_perplexity(test_data, test_data_distorted, original_ent)
print('=' * 100)
print('|  Contrastive Cross Entropy {:5.2f} | Contrastive Perplexity {:8.2f}'.format(contrastive_ent, math.exp(contrastive_ent)))
print('=' * 100)

nbest_score(args.nbest_in, args.nbest_out, lower=args.lower,
            report_wer=args.report_wer, report_bleu=args.report_bleu)
