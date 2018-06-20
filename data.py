import os
import torch
import sys
import codecs
from collections import Counter


class Dictionary(object):
    def __init__(self, max_vocab_size=sys.maxsize, unk_token='<unk>'):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        self.unk_token = unk_token
        self.max_vocab_size = max_vocab_size

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def build(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with codecs.open(path, 'r', 'utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.add_word(word)

    def __len__(self):
        return len(self.idx2word)

    def prune(self):
        top_n_idx2word = []
        top_n_word2idx = {}
        top_n_counter = Counter()
        self.total = 0

        for new_idx, (old_idx, counter) in enumerate(self.counter.most_common(self.max_vocab_size)):
            word = self.idx2word[old_idx]
            top_n_idx2word.append(word)
            top_n_word2idx[word] = new_idx
            self.total += 1
            top_n_counter[new_idx] = self.counter[old_idx]

        self.counter = top_n_counter
        self.idx2word = top_n_idx2word
        self.word2idx = top_n_word2idx

        if self.unk_token not in self.word2idx:
            self.word2idx[self.unk_token] = len(self.idx2word) - 1

class Corpus(object):
    def __init__(self, path, max_vocab_size=sys.maxsize, unk_token='<unk>'):
        self.dictionary = Dictionary(max_vocab_size, unk_token)
        self.dictionary.build(os.path.join(path, 'train.txt'))
        if len(self.dictionary) > self.dictionary.max_vocab_size:
            self.dictionary.prune()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""

        assert os.path.exists(path)
        # Add words to the dictionary
        with codecs.open(path, 'r', 'utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        unk_id = self.dictionary.word2idx.get(self.dictionary.unk_token, -1)

        # Tokenize file content
        with codecs.open(path, 'r', 'utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, unk_id)
                    token += 1
        return ids

    def tokenize_sent(self, sentence):
        unk_id = self.dictionary.word2idx.get(self.dictionary.unk_token, -1)

        words=sentence.split()
        ids = torch.LongTensor(len(words))
        token = 0
        for word in words:
            ids[token] = self.dictionary.word2idx.get(word, unk_id)
            token += 1
        return ids
