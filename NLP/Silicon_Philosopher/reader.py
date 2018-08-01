import os
import torch
import torch.tensor as tensor

#     def __init__(self, rnn_type, token_size, embedding_size, hidden_size, layers_size):


class Corpus(object):

    def __init__(self, file_path):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = self.tokenize(file_path)

    def tokenize(self, file_path):

        assert os.path.exists(file_path)

        with open(file_path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split('\t')[1].lower().split() + ['<eos>']

                for word in words:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word[self.word2idx[word]] = word

                    tokens += 1

        # TODO pad the tensor with different size of sentences.


path = "/Users/rl/PycharmProjects/pytorch_examples/NLP/Silicon_Philosopher/data/quotes_dataset.txt"
c = Corpus(path)


