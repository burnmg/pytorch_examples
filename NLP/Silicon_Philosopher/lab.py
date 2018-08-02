import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size = 3
max_length = 3
hidden_size = 2
n_layers =1


l = [
    torch.tensor([1,2,3], dtype=torch.long),
torch.tensor([1,2], dtype=torch.long),
torch.tensor([1], dtype=torch.long)

]
packed1 = torch.nn.utils.rnn.pad_sequence(l, batch_first=True)
packed2 = torch.nn.utils.rnn.pad_sequence(l, batch_first=False)

emb = nn.Embedding(4, 5)
output = emb(packed1)

pack = torch.nn.utils.rnn.pack_padded_sequence(t, lengths = [3,2,1], batch_first= True)

embebedd