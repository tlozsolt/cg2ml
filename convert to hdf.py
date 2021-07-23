#Ben Thorne July 2021
#convert tiff stack to hdf5 file and save

import h5py
from skimage import io
I = io.imread("tfrGel10212018A_shearRun10292018f_hv01342_locInput.tif") #input file
hf = h5py.File('finalim.h5', 'w')  #output file
hf.create_dataset('image_dataset', data=I/65535)  #saves the numpy array into the hdf5 dataset
hf.close()  #close
