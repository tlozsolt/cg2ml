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
%reload_ext autoreload
%autoreload 2
from cg2ml.data.preprocessing import DatasetFactory
from cg2ml.training import make_dataloaders, make_trainer
from cg2ml.models.model import Regression3DCNN

factory = DatasetFactory(volume_data_path = './finalim.h5', label_path = './saved_chunk_xarray.nc')

#%%
dataset = factory.to_TensorDataset()
#%%
train_dl, val_dl = make_dataloaders(dataset, split_ratios = (9, 1), batch_size = 256, num_workers = 4)
nn_model = Regression3DCNN()
trainer = make_trainer(savedir = './results', max_epochs = 2000, gpus = 1)

#%%
trainer.fit(nn_model, train_dl, val_dl)



# %%
