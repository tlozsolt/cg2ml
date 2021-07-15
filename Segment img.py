# Ben Thorne, July 2021

# This script creates training data for a machine learning algorithm that will hopefully be able to identify particle
# centers given a z stack of raw images. The training data that is produced by the script will be the "solutions" to
# the problem.

# This script takes a CSV file with the particle center coordinates and creates a
# new image stack with pixels that are distributed according to a gaussian with a mean of the particle's center
# coordinates.

#packages
import csv
import numpy as np
import math as math

import xarray as xr

# Constants
zscale = 0.33  #z spacing between images in stack in pixels
gaussian_Std = 3  # standard deviation of gaussian used to fit to particle centers
gauss_rad = 2  # the number of pixels deep each gaussian will be represented as
zres= 596  # number of z frames
xres=2048
yres=2048

# INPUT FILES

locs = np.zeros(3)  #initialize np array that holds locations

# we read the csv file and initialize a numpy array that will hold the coordinates of the particle centers
with open('testcsv.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    row_count = sum(1 for row in csv_reader)
    print(row_count)
    #row_count=50
    locs = np.zeros((row_count-1,3))  #numpy array of center coordinates
    indexes = np.zeros(row_count-1)  #numpy array of indexes
    chunk_locs = np.zeros((row_count-1,3))  #numpy array of chunk coordinates to fetch image of particle given 13 x 13 x 11 size


# we read the input csv file and fill in the coordinates into an xarray
with open('testcsv.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if line_count == 0: # this is the header row. Ignore this
            print(row)
            line_count += 1
        else:
            i=0
            while i <= 3:  # we do this weird thing to have the coordinates go in the order x,y,z instead of z,y,x (offensive to me)
                if i==0:
                    indexes[line_count-1]=int(row[i])  #populate index array
                if i==1:
                    locs[line_count-1,2]=float(row[i]) #z coord
                    zloc=math.floor(float(row[i])*.115/.15)-5
                    if zloc <0:
                        zloc=0
                    chunk_locs[line_count-1,2]=int(zloc)
                if i==2:
                    locs[line_count-1,1]=float(row[i]) #y coord
                    chunk_locs[line_count-1,1]=int(math.floor(float(row[i]))-6)
                else:
                    locs[line_count-1,0]=float(row[i]) #x coord

                    chunk_locs[line_count-1,0]=int(math.floor(float(row[i]))-6)
                i+=1
            line_count += 1

    print(f'Processed {line_count} lines.')

    ds=xr.Dataset(
        {
            "center_coords":(["index","dimension"],locs),
            "chunk_origin":(["index","dimension"],chunk_locs),
        },
        coords={
            "dims":(["dimension"],["x","y","z"]),
            "indexes":(["index"],indexes),

        },
    )
print(ds["center_coords"])
print(ds["chunk_origin"])

ds.to_netcdf("saved_chunk_xarray.nc")
