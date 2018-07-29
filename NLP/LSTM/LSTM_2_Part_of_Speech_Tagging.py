import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    return torch.tensor(idx, dtype = torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
tag_to_ix = {}
for sentence, tags in training_data:
    for w in sentence:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)
    for t in tags:
        if t not in tag_to_ix:
            tag_to_ix[t] = len(tag_to_ix)

print(word_to_ix)
print(tag_to_ix)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

        self.hidden = self.init_hidden()


    def init_hidden(self):

        hidden = (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

        return hidden

    def forward(self, inputs):

        x = self.word_embeddings(inputs)
        x, self.hidden = self.lstm(x.view(len(inputs), 1, -1))
        x = self.hidden2tag(x.view(len(sentence), -1))
        tag = F.log_softmax(x, dim=1)

        return tag

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(3000):
    for sentence, tags in training_data:
        optimizer.zero_grad()

        sentence = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        out = model(sentence)

        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()

'''
with torch.no_grad():                                
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
'''



