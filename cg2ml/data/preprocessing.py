import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

def read_volume_from_hdf5(volume_data_path):
    hdf_file = h5py.File(volume_data_path)
    volume_data = hdf_file['image_dataset']
    return volume_data

def stack_iterables_to_tensor(iterable):
    return torch.tensor(np.stack(list(iterable)), dtype = torch.float32)

class DatasetFactory():

    def __init__(self, volume_data_path, label_path, chunk_size = 15):
        # Right now, we will implicitly assume our volume data is in hdf5 format
        # and labels are contained in a xarray, but this may not be the best abstraction
        # and may change in the future
        self.chunk_size = chunk_size
        self.volume = read_volume_from_hdf5(volume_data_path)
        self.labels = xr.open_dataset(label_path)
    
    def __len__(self):
        return len(self.labels['center_coords'])

    def __getitem__(self, index):
        chunk = self.get_chunk(index)
        center = self.get_center_coord(index) - 0.5*(self.chunk_size+1) # for chunk size 5, changes index from 0, 1, 2, 3, 4 -> -2, -1, 0, 1, 2
        # The concrete chunk size of 15 is baked into the script that generates the xarrays as well
        # This will also need to be addressed later
        return chunk, center

    def get_center_coord(self, index):
        # will later need to make sure the coordinates are in expected order
        # may be better to utilize the dimension names x, y, z
        # also using the specialized method to_numpy instead of the numpy array constructor
        # as it may have internal optimizations built in
        center_coord_xyz = self.labels['center_coords'][index].to_numpy()
        return center_coord_xyz[::-1] # convert xyz coordinates to zyx
    
    def get_chunk_origin(self, index):
        chunk_origin_xyz = self.labels['chunk_origin'][index].to_numpy()
        return chunk_origin_xyz[::-1] # convert xyz coordinates to zyx

    def get_chunk(self, index):
        chunk_origin = self.get_chunk_origin(index)
        chunk_indices = self._compute_chunk_indices(chunk_origin)
        chunk = self.volume[chunk_indices]

        if any(size != self.chunk_size for size in chunk.shape):
            chunk = self._pad_chunk(chunk)
        
        return chunk
            
    def _compute_chunk_indices(self, chunk_origin):
        return tuple(slice(c, c + self.chunk_size) for c in np.int_(chunk_origin))

    def _pad_chunk(self, chunk):
        pad_width = tuple((0, self.chunk_size - s)for s in chunk.shape)
        return np.pad(chunk, pad_width)

    def to_TensorDataset(self):
        chunks_tensor, labels_tensor = map(stack_iterables_to_tensor, zip(*[value for value in tqdm(self)]))
        tensordataset = TensorDataset(torch.unsqueeze(chunks_tensor, dim = 1), labels_tensor)
        return tensordataset