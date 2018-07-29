from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = f.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]

        num_features = 1

        for s in size:
            num_features *= s

        return num_features


net = Net()

input = torch.randn(1, 1, 32, 32)

target = torch.arange(1, 11)
target = target.view(1, -1)
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())

for i in range(10):
    output = net(input)
    loss = criterion(output, target)
    loss.backward()

    optimizer.step()

