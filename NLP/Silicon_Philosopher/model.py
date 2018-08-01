import torch
import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self, rnn_type, token_size, embedding_size, hidden_size, layers_size):

        super(RNNModel, self).__init__(self)
        self.encoder = nn.Embedding(token_size, embedding_size)
        self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, layers_size) # bias can be False
        self.decoder = nn.Linear(hidden_size, token_size)


    def init_weight(self):
        self.encoder