import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn


class ModelCNNMnist(nn.Module):
    def __init__(self):
        super(ModelCNNMnist, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(50, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.fc1.apply(init_weights)

    def forward(self, x):
        fc_ = x.view(-1, 28 * 28)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output





