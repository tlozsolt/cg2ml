#Function to fetch 13 x 13 x 11 image of a particular particle given a particle index
#Ben Thorne, July 2021

import h5py
import numpy as np
import xarray as xr
#returns a 3D numpy array which is the image of a particular particle.
# Pass the image file (hdf5 format) string, the index of the particle, and the xarray of particle coordinates (saved as nc file)

def get_chunk_img(filename,index,indexdata):
  hf = h5py.File(filename, 'r')
  hf.keys()
  n1 = hf.get('image_dataset')
  n1 = np.array(n1)
  ds = xr.open_dataset(indexdata)

  loc=np.array(ds.chunk_origin.loc[dict(index=index)])

  output = n1[int(loc[0]):int(loc[0])+13,int(loc[1]):int(loc[1])+13,int(loc[2]):int(loc[2])+11]

  return output




get_chunk_img("flatfieldim.h5",0,"saved_chunk_xarray.nc")
