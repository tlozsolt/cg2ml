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
from cg2ml.data.preprocessing import DatasetFactory

factory = DatasetFactory(volume_data_path = './finalim.h5', label_path = './saved_chunk_xarray.nc')
# %%
len(factory)
# %%
factory[0][1]
#%%
dataset = factory.to_TensorDataset()
#%%
import torch
N_data = len(dataset)
N_val = int(0.1*N_data)
N_train = N_data - N_val
train_data, val_data = torch.utils.data.random_split(dataset, [N_train, N_val])
train_dl = torch.utils.data.DataLoader(train_data, batch_size = 256, num_workers = 4)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = 256, num_workers = 4)
#%%
from cg2ml.models.model import Regression3DCNN
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger('logs/')
cnn = Regression3DCNN()
trainer = pl.Trainer(logger = tb_logger, max_epochs = 2000, gpus = 1)
#%%
trainer.fit(cnn, train_dl, val_dl)