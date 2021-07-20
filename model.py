#%%
from more_itertools import pairwise, value_chain
import torch
import torch.nn as nn
import pytorch_lightning as pl

def pairwise_value_chain(*args):
    return pairwise(value_chain(*args))

mean_squared_error = nn.MSELoss()

class Regression3DCNN(pl.LightningModule):

    def __init__(self, conv_channels = [16, 32, 64], fc_features = [1728, 500, 100, 50, 3]):
        super().__init__()

        self.input_channel = 1
        self.conv_channels = conv_channels
        self.output_feature = 3
        self.fc_features = fc_features

        self.conv_layers = self._make_conv_layers()
        self.maxpool3d = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.fc_layers = self._make_fully_connected_layers()

        self.lr = 0.005

    def _make_conv_layers(self):
        channel_args_itr = pairwise_value_chain(self.input_channel, self.conv_channels)
        conv_layers = nn.ModuleList([nn.Conv3d(*c_in_out, kernel_size = 3, padding = 1) for c_in_out in channel_args_itr])
        return conv_layers

    def _make_fully_connected_layers(self):
        feature_args_itr = pairwise_value_chain(self.fc_features, self.output_feature)
        fc_layers = nn.ModuleList([nn.Linear(*feat_in_out) for feat_in_out in feature_args_itr])
        return fc_layers

    def forward(self, x):
        for conv3d in self.conv_layers[:-1]:
            x = self.maxpool3d(torch.relu(conv3d(x)))
        x = torch.relu(self.conv_layers[-1](x))
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
#%%
from functools import lru_cache

import numpy as np

@lru_cache(maxsize = 1)
def create_coordinate_array(side, scales = (0.115, 0.115, 0.15)):
    center = (side-1)/2
    pixels = np.arange(side) - center
    
    coords = [pixels*scale for scale in scales]
    return np.stack(np.meshgrid(*coords), axis = -1)


def gaussianND(coord, center, radius):
    Dr2 = np.sum((coord-center)**2, axis = -1)
    return np.exp(-Dr2/radius**2)

#%%
from numpy.random import default_rng
N_data = 5000
rng = default_rng()
means = rng.uniform(size = (N_data, 1, 1, 1, 3))
means = (means-0.5)*0.4 # means will be random vectors in the box bounded by [-0.2, -0.2, -0.2] and [0.2, 0.2, 0.2]
#%%
sigma = 0.2
coords = create_coordinate_array(15)
vols = gaussianND(np.expand_dims(coords, axis = 0), means, sigma).astype(np.float32)
#%%
y_tensor = torch.as_tensor(np.squeeze(means), dtype = torch.float32)
x_tensor = torch.as_tensor(np.expand_dims(vols, axis = 1), dtype = torch.float32)
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
#%%
train_data, val_data = torch.utils.data.random_split(dataset, [4500, 500])
train_dl = torch.utils.data.DataLoader(train_data, batch_size = 256, num_workers = 4)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = 256, num_workers = 4)
#%%
from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger('logs/')
cnn = Regression3DCNN()
trainer = pl.Trainer(logger = tb_logger, max_epochs = 2000)
#%%
trainer.fit(cnn, train_dl, val_dl)
#%%