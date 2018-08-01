import torch
import torch.nn as nn
import torch.tensor as tensor


class RNNModel(nn.Module):

    def __init__(self, rnn_type, token_size, embedding_size, hidden_size, layers_size):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(token_size, embedding_size)
        self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, layers_size)  # bias can be False
        self.decoder = nn.Linear(hidden_size, token_size)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.layers_size = layers_size

    def init_weight(self):
        None

    def forward(self, input, hidden):
        embeddings = self.encoder(input)
        output, hidden = self.rnn(embeddings, hidden)

        decoded = self.decoder(output.view(output.shape[0] * output.shape[1], output.shape[2]))
        return decoded.view(output.shape[0], output.shape[1], output.shape[2]), hidden

    def init_hidden(self, batch_size):

        if self.rnn_type == 'LSTM':
            return (torch.zeros((self.layers_size, batch_size, self.hidden_size), dtype=torch.float),
                    torch.zeros((self.layers_size, batch_size, self.hidden_size), dtype=torch.float))
        else:
            return torch.zeros((self.layers_size, batch_size, self.hidden_size), dtype=torch.float)
