import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)
inputs = [torch.rand(1, 3) for _ in range(5)]
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# print(inputs)

hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
out = lstm(inputs, hidden)
print(out)

print(hidden)
