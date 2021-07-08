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
import matplotlib as plt

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
    locs = np.zeros((row_count-1,3))

# we read the input csv file and fill in the coordinates into the numpy array
with open('testcsv.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if line_count == 0: # this is the header row. Ignore this
            print(row)
            line_count += 1
        else:
            i=1
            while i <= 3:
                if i==1:
                    locs[line_count-1,2]=float(row[i])
                if i==2:
                    locs[line_count-1,1]=float(row[i])
                else:
                    locs[line_count-1,0]=float(row[i])
                i+=1
            line_count += 1
        #if line_count==50:
        #    break
    print(f'Processed {line_count} lines.')
    #print(locs)



# PROCESSING STEPS

## we now want to create  an image csv of floats corresponding to
#imitialize the numpy array
gaussim=np.zeros((xres,yres,zres))

#creates a normalized gaussian array
def gaussian(x, mu, sig):
    out=np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return out/out[int((np.size(x)-1)/2)]

tup=np.shape(locs)
i=0
x=np.arange(-1*gauss_rad,gauss_rad+1)
while i<tup[0]:
    #create the gaussian pixels
    xgauss=gaussian(x,locs[i,0]%1,gaussian_Std)
    ygauss=gaussian(x,locs[i,1]%1,gaussian_Std)
    z=np.arange(-1*round(gauss_rad/zscale),round(gauss_rad/zscale)+1)
    zgauss=gaussian(z,locs[i,2]%1,gaussian_Std)

    #paste the gaussian pixels into the image array
    gaussim[int(locs[i,0])-gauss_rad:int(locs[i,0])+gauss_rad+1,int(locs[i,1]),int(locs[i,2]/zscale)]=xgauss
    gaussim[int(locs[i,0]),int(locs[i,1])-gauss_rad:int(locs[i,1])+gauss_rad+1,int(locs[i,2]/zscale)]=ygauss
    gaussim[int(locs[i,0]),int(locs[i,1]),int(locs[i,2]/zscale)-round(gauss_rad/zscale):int(locs[i,2]/zscale)+round(gauss_rad/zscale)+1]=zgauss
    i=i+1





# OUTPUT FILES
