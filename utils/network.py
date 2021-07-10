import numpy as np
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()

        self.hparams = hparams
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hparams["n_hidden"][0]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"][0], hparams["n_hidden"][1]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"][1], 1)
        )

    def forward(self, x):
        return self.model(x)


