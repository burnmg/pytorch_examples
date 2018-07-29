import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1)

word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype = torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence) - 2)]
# print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):

        super(NGramLanguageModeler, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim = 1)

        return x


losses = []

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

print(model)
for epoch in range(100):
    total_loss = 0
    for context, target in trigrams:

        optimizer.zero_grad()
        inputs = torch.tensor(
            [word_to_ix[word] for word in context], dtype = torch.long
        )
        out = model(inputs)
        loss = loss_function(out, torch.tensor([word_to_ix[target]], dtype = torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)


plt.plot(range(1, len(losses) + 1), losses, 'ro')








