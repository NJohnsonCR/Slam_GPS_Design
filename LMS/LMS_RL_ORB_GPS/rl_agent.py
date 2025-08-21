import torch.nn as nn

class SimpleRLAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)