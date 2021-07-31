
#%%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .utils import pairwise_value_chain

mean_squared_error = nn.MSELoss()

class Regression3DCNN(pl.LightningModule):

    def __init__(self, conv_channels = [16, 32, 64], fc_features = [1728, 500, 100, 50, 3]):
        super().__init__()

        self.input_channel = 1
        self.conv_channels = conv_channels
        self.output_feature = 3
        self.fc_features = fc_features

        self.conv_layers = self._make_conv_layers()
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm3d(channel) for channel in conv_channels])
        self.maxpool3d = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.fc_layers = self._make_fully_connected_layers()

        self.lr = 0.001

    def _make_conv_layers(self):
        channel_args_itr = pairwise_value_chain(self.input_channel, self.conv_channels)
        conv_layers = nn.ModuleList([nn.Conv3d(*c_in_out, kernel_size = 3, padding = 1) for c_in_out in channel_args_itr])
        return conv_layers

    def _make_fully_connected_layers(self):
        feature_args_itr = pairwise_value_chain(self.fc_features, self.output_feature)
        fc_layers = nn.ModuleList([nn.Linear(*feat_in_out) for feat_in_out in feature_args_itr])
        return fc_layers

    def forward(self, x):
        for batchnorm3d, conv3d in zip(self.batchnorm_layers[:-1], self.conv_layers[:-1]):
            x = self.maxpool3d(torch.relu(batchnorm3d(conv3d(x))))
        x = torch.relu(self.batchnorm_layers[-1](self.conv_layers[-1](x)))
        x = torch.flatten(x, start_dim = 1)
        
        for fc in self.fc_layers[:-1]:
            x = torch.relu(fc(x))
        x = torch.tanh(self.fc_layers[-1](x))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = mean_squared_error(y, y_hat)
        self.log('training loss', train_loss, on_epoch = True, logger = True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = mean_squared_error(y, y_hat)
        self.log('validation loss', val_loss, on_epoch = True, logger = True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
