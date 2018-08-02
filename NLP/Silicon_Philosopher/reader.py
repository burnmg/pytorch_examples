import os
import torch
import torch.nn as nn

class Corpus(object):

    def __init__(self, file_path):
        self.word2idx = {}
        self.idx2word = {}
        self.idxs,self.sentences_lengths  = self.tokenize(file_path)

    def tokenize(self, file_path):

        assert os.path.exists(file_path)

        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>'
        with open(file_path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split('\t')[1].lower().split() + ['<eos>']

                for word in words:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word[self.word2idx[word]] = word

                    tokens += 1


        idxs_list = []
        idxs_lengths = []
        with open(file_path, 'r') as f:
            tokens = 0

            for line in f:

                words = line.split('\t')[1].lower().split() + ['<eos>']
                sentence_idxs = torch.zeros(len(words), dtype=torch.long)

                for i, word in enumerate(words):
                    sentence_idxs[i] = self.word2idx[word]

                idxs_list.append(sentence_idxs)
                idxs_lengths.append(len(words))

        padded_idxs = nn.utils.rnn.pad_sequence(idxs_list, batch_first=True)
        return padded_idxs, idxs_lengths


# path = "/Users/rl/PycharmProjects/pytorch_examples/NLP/Silicon_Philosopher/data/quotes_dataset.txt"
path = '/Users/rl/PycharmProjects/pytorch_examples/NLP/Silicon_Philosopher/data/small_data'
c = Corpus(path)

print(c.idxs)

