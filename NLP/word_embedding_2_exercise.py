import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMDEDDING_DIM = 100
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
VOCAB_SIZE = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
# print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):

        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)


    def forward(self, inputs):
        x = sum(self.embedding(inputs)).view(1, -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim = 1)
        return x

    def get_word_embedding(self, word, word_to_idx):
        word = torch.tensor(word_to_ix[word], dtype = torch.long)
        return self.embedding(word).view(1, -1)

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)
model = CBOW(VOCAB_SIZE, EMDEDDING_DIM, CONTEXT_SIZE)
opt = optim.SGD(model.parameters(), lr = 0.01)
loss_function = nn.NLLLoss()

losses = []

for epoch in range(30):
    total_loss = 0
    for train, target in data:
        opt.zero_grad()

        input = [word_to_ix[word] for word in train]

        out = model(torch.tensor(input))
        loss = loss_function(out, torch.tensor([word_to_ix[target]]))

        loss.backward()
        opt.step()
        total_loss += loss.item()

    losses.append(total_loss)

print(losses)
plt.plot(losses)
