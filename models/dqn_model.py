import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)
