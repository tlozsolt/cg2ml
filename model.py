#%%
from more_itertools import pairwise, value_chain
import torch
import torch.nn as nn
import pytorch_lightning as pl

def pairwise_value_chain(*args):
    return pairwise(value_chain(*args))

class Regression3DCNN(pl.LightningModule):

    def __init__(self, conv_channels = [16, 32, 64, 64], fc_features = [4096, 1500, 500, 100, 50, 3]):
        super().__init__()

        self.input_channel = 1
        self.conv_channels = conv_channels
        self.output_feature = 3
        self.fc_features = fc_features

        self.conv_layers = self._make_conv_layers()
        self.maxpool3d = nn.MaxPool3d(2)
        self.fc_layers = self._make_fully_connected_layers()

    def _make_conv_layers(self):
        channel_args_itr = pairwise_value_chain(self.input_channel, self.conv_channels)
        conv_layers = nn.ModuleList([nn.Conv3d(*c_in_out, 3) for c_in_out in channel_args_itr])
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
        
#%%
cnn = Regression3DCNN()
# %%
test_img = torch.zeros(1, 1, 32, 32, 32)
test_img.size()

# %%
cnn(test_img)
# %%
