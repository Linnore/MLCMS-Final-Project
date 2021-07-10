import numpy as np
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    """Fully connected neural network.
    """
    def __init__(self, hparams, input_size, output_size, activation=nn.ReLu):
        """
        Args:
            hparams (dict): contains all hyper-parameters and configuration of this network.
            input_size (int): size of each input data point
            output_size (int): size of output layer
            activation (function, optional): the activation function. Defaults to nn.ReLu.
        """
        super().__init__()

        self.hparams = hparams
        self.input_size = input_size

        self.model = nn.ModuleList()
        numOfLayers = hparams["numOfLayers"]
        layerSize = hparams["layerSize"]

        self.model.append(nn.Linear(input_size, layerSize[0]))
        for i in range(numOfLayers-1):
            self.model.append(nn.Linear(layerSize[i], layerSize[i+1]))
            self.model.append(activation())
        self.model.append(nn.Linear(layerSize[numOfLayers-1], 1))

    def forward(self, x):
        fx = x
        for layer in self.model:
            fx = layer(fx)
        return fx


