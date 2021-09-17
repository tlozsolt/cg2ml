# TODO: Write documentations for the functions below
# TODO: Create script so that the dimensions are auto-caculated
# TODO: Optuna to tune hyperparameters
# TODO: Logging capability for hyperparameters, early stopping
#%%
import h5py
from h5py._hl import dataset
from cg2ml.models.utils import pairwise_value_chain
# %%
import xarray as xr

labels = xr.open_dataset('./saved_chunk_xarray.nc')
# %%
# %%
out = labels['center_coords'][0].to_numpy()[::-1]
out
#%%
labels['chunk_origin'][0].to_numpy()[::-1]
# %%
### Create dataset from the input data files & train the neural network ###
%reload_ext autoreload
%autoreload 2
from cg2ml.data.preprocessing import DatasetFactory
from cg2ml.training import make_dataloaders, make_trainer
from cg2ml.models.model import Regression3DCNN, Regression3DCNNv2

factory = DatasetFactory(volume_data_path = './finalim.h5', label_path = './saved_chunk_xarray.nc')

#%%
dataset = factory.to_TensorDataset()
#%%
train_dl, val_dl = make_dataloaders(dataset, split_ratios = (9, 1), batch_size = 256, num_workers = 4)
nn_model = Regression3DCNNv2(learning_rate = 0.001)
trainer = make_trainer(savedir = './results', max_epochs = 2000, gpus = 1)

#%%
trainer.fit(nn_model, train_dl, val_dl)


# %%
### Inspect the contents of the csv file ###
import pandas as pd
csv = pd.read_csv('../Data/locations/tfrGel10212018A_shearRun10292018f_locations_hv01342_sed_trackPy.csv', delimiter = ' ')
csv.head()
# %%
csv.shape
#%%
names = [item for item in csv.columns if 'std' in item]
names
# %%
import matplotlib.pyplot as plt
plt.hist(csv['background']/(2**16-1), bins = 20)
# %%
import numpy as np
def get_center_intensity(volume, normalized_center_coords):
    center_coords = np.floor_divide(volume.shape, 2) + np.array(normalized_center_coords)
    return volume[tuple(np.int_(center_coords))]
# %%
set(csv['disc_size'])
# %%
get_center_intensity(*factory[0])
# %%
from tqdm import tqdm
center_intensities = np.array([get_center_intensity(*chunk) for chunk in tqdm(factory)])
# %%
import matplotlib.pyplot as plt

plt.hist(center_intensities)
# %%
import torch
def normalized_radius(position, center, scale):
    return torch.linalg.norm((position-center)/scale, dim = -1)



# Equivalent to the definition in paper but does not use explicit conditionals
# This should make it play better with autograd
def disk(r, d):
    exponent = torch.relu((r-d)/(1-d))
    return torch.exp(-exponent**2)

def get_coordinates(volume_shape):
    return torch.meshgrid(*[torch.arange(n)-0.5*(n+1) for n in volume_shape])

def ideal_image(volume_shape, center, scale, disk_size):
    positions = get_coordinates(volume_shape)
    radii = normalized_radius(torch.stack(positions, dim = -1), center, scale)
    return disk(radii, disk_size)
#%%
dataset[0][0].size()
import scipy.ndimage as ndimage

labels, _ = ndimage.label(dataset[0][0].numpy())
# %%
scale = torch.tensor([2.0, 3.4, 3.4])
out = ideal_image((15, 15, 15), dataset[0][1], scale, 0.2)
# %%
out.size()
# %%
import plotly.graph_objects as go
values = labels
#values = out.numpy()
#values = dataset[1][0].numpy()
Z, Y, X = get_coordinates((15, 15, 15))
fig = go.Figure(data=go.Volume(
    x=X.numpy().flatten(),
    y=Y.numpy().flatten(),
    z=Z.numpy().flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=7, # needs to be a large number for good volume rendering
    ))
fig.show()
# %%
