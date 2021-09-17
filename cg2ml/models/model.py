
#%%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .utils import pairwise_value_chain

mean_squared_error = nn.MSELoss()

class Regression3DCNN(pl.LightningModule):

    def __init__(self, conv_channels = [16, 32, 64], fc_features = [1728, 500, 100, 50, 3], learning_rate = 0.001):
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.input_channel = 1
        self.conv_channels = conv_channels
        self.output_feature = 3
        self.fc_features = fc_features

        self.conv_layers = self._make_conv_layers()
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm3d(channel) for channel in conv_channels])
        self.maxpool3d = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.fc_layers = self._make_fully_connected_layers()

        self.lr = learning_rate

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
        x = self.fc_layers[-1](x)
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
        print(val_loss)
        self.log('validation loss', val_loss, on_epoch = True, logger = True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

# position: 15, 15, 15, 3
# center B, 3
# scale 3
def normalized_radius(position, center, scale):
    return torch.linalg.norm((position-center.view(-1, 1, 1, 1, 3))/scale, dim = -1) # B, 15, 15, 15

# Equivalent to the definition in paper but does not use explicit conditionals
# This should make it play better with autograd
# d: B, 1
# r: B, 15, 15, 15
def disk(r, d):
    d_ = d.view(-1, 1, 1, 1)
    exponent = torch.relu((r-d_)/(1-d_))
    return torch.exp(-exponent**2) # B, 15, 15, 15

def get_coordinates(volume_shape):
    return torch.meshgrid(*[torch.arange(n)-0.5*(n+1) for n in volume_shape])

def ideal_image(coordinates, center, scale, disk_size, center_intensity):
    radii = normalized_radius(coordinates, center, scale) # B, 15, 15, 15
    return disk(radii, disk_size) * center_intensity.view(-1, 1, 1, 1)


class Regression3DCNNv2(pl.LightningModule):

    def __init__(self, conv_channels = [16, 32, 64], fc_features = [1728, 500, 100, 50, 3], learning_rate = 0.001):
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.input_channel = 1
        self.conv_channels = conv_channels
        self.output_feature = 5 # 3 for the coordinates, 1 for disk size, 1 for peak intensity
        self.fc_features = fc_features

        self.conv_layers = self._make_conv_layers()
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm3d(channel) for channel in conv_channels])
        self.maxpool3d = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.fc_layers = self._make_fully_connected_layers()

        vol_coords = torch.stack(get_coordinates((15, 15, 15)), dim = -1)
        self.register_buffer('volume_coordinates', vol_coords)
        self.register_buffer('scale', torch.tensor([2.0, 3.4, 3.4]))

        self.mish = nn.Mish()
        self.lr = learning_rate

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
            x = self.maxpool3d(self.mish(batchnorm3d(conv3d(x))))
        x = self.mish(self.batchnorm_layers[-1](self.conv_layers[-1](x)))
        x = torch.flatten(x, start_dim = 1)
        
        for fc in self.fc_layers[:-1]:
            x = self.mish(fc(x))
        x = self.fc_layers[-1](x)
        center_coord = x[:, 0:3]
        params = torch.sigmoid(x[:, 3:])
        return center_coord, params
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, params_hat = self(x)
        x_hat = ideal_image(self.volume_coordinates, y_hat, self.scale, params_hat[:, 0], params_hat[:, 1])
        volume_intensity_mse = mean_squared_error(torch.squeeze(x), x_hat)
        coord_mse = mean_squared_error(y, y_hat)
        train_loss = 0.001*coord_mse + volume_intensity_mse
        self.log('training loss', train_loss, on_epoch = True, logger = True)
        self.log('training volume intensity mse', volume_intensity_mse, on_epoch = True, logger = True)
        self.log('training center coordinate mse', coord_mse, on_epoch = True, logger = True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, params_hat = self(x)
        x_hat = ideal_image(self.volume_coordinates, y_hat, self.scale, params_hat[:, 0], params_hat[:, 1])
        val_loss = mean_squared_error(torch.squeeze(x), x_hat)
        coord_mse = mean_squared_error(y, y_hat)
        self.log('validation loss', val_loss, on_epoch = True, logger = True)
        self.log('validation center coordinate mse', coord_mse, on_epoch = True, logger = True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)