import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 120)
        self.fc2 = nn.Linear(120, action_size)
        self.rl1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl1(x)
        return x

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
