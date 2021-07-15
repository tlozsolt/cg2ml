#Ben Thorne July 2021
#convert tiff stack to hdf5 file and save

import h5py
from skimage import io
I = io.imread("tfrGel10212018A_shearRun10292018f_flatField_hv01342.tif") #input file
hf = h5py.File('flatfieldim.h5', 'w')  #output file
hf.create_dataset('image_dataset', data=I)  #saves the numpy array into the hdf5 dataset
hf.close()  #close
