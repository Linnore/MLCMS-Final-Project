import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import os


class FullyConnectedNet(pl.LightningModule):
    """Fully connected neural network using pytoch-lightening.
    """

    def __init__(self, hparams, input_size, output_size, criterion=nn.MSELoss(), activation=nn.ReLU):
        """
        Args:
            hparams (dict): contains all hyper-parameters and configuration of this network.
            input_size (int): size of each input data point
            output_size (int): size of output layer
            criterion (function): the loss function 
            activation (function, optional): the activation function. Defaults to nn.ReLu.
        """
        super().__init__()
        self.hparams = hparams
        self.input_size = input_size
        self.criterion = criterion
        self.activation = activation
        self.lr = hparams["learning_rate"]

        self.current_batch_step = 0

        self.model = nn.ModuleList()
        numOfLayers = hparams["numOfLayers"]
        layerSize = hparams["layerSize"]

        self.model.append(nn.Linear(input_size, layerSize[0]))
        for i in range(numOfLayers-1):
            self.model.append(nn.Linear(layerSize[i], layerSize[i+1]))
            self.model.append(activation())
        self.model.append(nn.Linear(layerSize[numOfLayers-1], output_size))

    def forward(self, x):
        fx = x
        for layer in self.model:
            fx = layer(fx)
        return fx

    def training_step(self, batch, batch_idx):
        X, velocity = batch
        self.current_batch_step += 1
        # Perform forward pass
        vhat = self.forward(X)

        # Compute loss
        loss = self.criterion(velocity, vhat)

        self.log("train_loss", loss)
        self.logger.experiment.add_scalars(
            "loss", {'train': loss}, self.current_batch_step)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, velocity = batch

        # Perform forward pass
        vhat = self.forward(X)

        # Compute loss
        loss = self.criterion(velocity, vhat)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the validation loss values to the tensorboard
        self.log('val_loss', avg_loss)
        self.logger.experiment.add_scalars(
            "loss", {'val': avg_loss}, self.current_batch_step)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.model.parameters(), self.hparams["learning_rate"])
        return optim

    def output_summary(self, numOfSamples, logged_matrics, train_dataset_label, val_dataset_label, summary_file_path):
        """Generate the summary of current model, including on which datasets it was trained and validated, and some performance indices.
        The summary will be output into a given file.

        Args:
            numOfSamples (int): number of samples in the validation dataset. I.e., n in the AIC expression.
            logged_matrics (dict): the losses of the model, logged when is was early-stopped.
            train_dataset_label (str): a label to represent the training dataset.
            val_dataset_label (str): a label to represent the validation dataset.
            summary_file_path (str): the path to a summary file for output purpose.
        """

        mse = logged_matrics["val_loss"]
        train_loss = logged_matrics["train_loss"]

        k = np.sum(p.numel() for p in self.parameters())

        aic = 2*k + numOfSamples*np.log(mse) + numOfSamples*(1+np.log(2*np.pi))

        if not os.path.isfile(summary_file_path):
            print("Created file "+summary_file_path)
            with open(summary_file_path, "w") as output:
                output.write(
                    "Model Train_dataset_label Val_dataset_label Train_loss MSE AIC\n")
        else:
            print(summary_file_path +
                  " exists, model summary will be attached to the end of this file.")

        with open(summary_file_path, "a") as output:
            model_name = "%d_" % self.hparams["numOfLayers"]
            for i in range(self.hparams["numOfLayers"]-1):
                model_name = model_name + "%d_" % self.hparams["layerSize"][i]

            model_name = model_name + \
                "%d" % self.hparams["layerSize"][self.hparams["numOfLayers"]-1]

            output.write(model_name + " " + train_dataset_label + " " +
                         val_dataset_label + " %f %f %f\n" % (train_loss, mse, aic))
