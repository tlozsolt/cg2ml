#%%
from more_itertools import pairwise, value_chain
import torch
import torch.nn as nn
import h5py
import xarray as xr
import time
import pytorch_lightning as pl
from torch.utils.tensorboard  import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def get_chunk_img(n1,index,indexdata):

   # print(np.shape(n1))
   loc=np.array(indexdata.chunk_origin.loc[dict(index=index)])
   # cen=np.array(indexdata.center_coords.loc[dict(index=index)])
   # print(cen)
   # print(index)
   # print(loc+cen)
   # plt.imshow(n1[:,:,int(loc[2]+cen[2])])
   # plt.scatter(loc[0]+cen[0],loc[1]+cen[1])
   # plt.show()
   # plt.imshow(n1[0:50,100:200,int(loc[2]+cen[2])])
   # plt.show()
   # print(int(loc[0]+15),int(loc[0])+15, int(loc[1]), int(loc[2]))

   output = n1[int(loc[1]):int(loc[1]+15),int(loc[0]):int(loc[0]+15),int(loc[2]):int(loc[2]+15)]
   size=np.shape(output)
   #print(size)
   while size != (15,15,15):
       if size[2] != 15:
           #np.append(output,np.zeros((15,15)),axis=2)
           output=np.dstack((output, np.zeros((15,15))))
           size=np.shape(output)
       if size[0] != 15:

           output=np.concatenate((output, [np.zeros((15,15))]),axis=0)

           size=np.shape(output)
       if size[1] != 15:
           print(np.shape([np.zeros((15,15))]))
           output=np.concatenate((output, np.transpose([np.zeros((15,15))],(1,0,2))),axis=1)
           size=np.shape(output)
       #print(size)


   # f=0
   # while f<15:
   #      c=plt.imshow(output[:,:,f])
   #      plt.scatter([cen[0]],[cen[1]])
   #      print(cen[0:2])
   #      plt.show()
   #      f=f+1
        #time.sleep(0.1)
   #print(output)
   return output






#%%
from numpy.random import default_rng
N_data = 5000
rng = default_rng()
xarray_data = xr.open_dataset("saved_chunk_xarray.nc")
hf = h5py.File('finalim.h5', 'r')
hf.keys()
n1 = hf.get('image_dataset')
n1 = np.array(n1)

n1 = np.transpose(n1, (1, 2, 0))
#means = rng.uniform(size = (N_data, 1, 1, 1, 3))
#means = (means-0.5)*0.4 # means will be random vectors in the box bounded by [-0.2, -0.2, -0.2] and [0.2, 0.2, 0.2]
#%%
sigma = 0.2
#coords = create_coordinate_array(15)
#vols = gaussianND(np.expand_dims(coords, axis = 0), means, sigma).astype(np.float32)
#print(np.shape(np.squeeze(means)))
#print(np.shape(np.expand_dims(vols, axis = 1)))
row_count = sum(1 for row in np.array(xarray_data.chunk_origin))
print(row_count)
means = np.zeros((row_count,3))
vols = np.zeros((row_count,15,15,15))
i=0
#print(get_chunk_img("flatfieldim.h5",2,xarray_data))
while i<row_count:
    #print(i)
    means[i,:]=np.array(xarray_data.center_coords.loc[dict(index=i)])
    #print(means[i,:])
    img=get_chunk_img(n1,i,xarray_data)
    #print(img)
    vols[i,:,:,:] =  img
    i=i+1;
#%%
y_tensor = torch.as_tensor(np.squeeze(means), dtype = torch.float32)
x_tensor = torch.as_tensor(np.expand_dims(vols, axis = 1), dtype = torch.float32)
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
#%%
print(dataset[0])
train_data, val_data = torch.utils.data.random_split(dataset, [13506,3371])
train_dl = torch.utils.data.DataLoader(train_data, batch_size = 256, num_workers = 4)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = 256, num_workers = 4)
#%%
from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger('logs/')
cnn = Regression3DCNN()
trainer = pl.Trainer(logger = tb_logger, max_epochs = 200)
print(trainer)
#%%
trainer.fit(cnn, train_dl, val_dl)
print(trainer)
writer = SummaryWriter()
#%%
