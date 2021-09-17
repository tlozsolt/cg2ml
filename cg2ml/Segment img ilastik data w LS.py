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
import h5py
import trackpy as tp
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default='browser'

import matplotlib.pyplot as plt

# Constants
zscale = 0.33  #z spacing between images in stack in pixels
gaussian_Std = 3  # standard deviation of gaussian used to fit to particle centers
gauss_rad = 2  # the number of pixels deep each gaussian will be represented as
zres= 596  # number of z frames
xres=2048
yres=2048
jitrange=1
chunksize=13  #we assume the chunk size is odd
chunkhalf=(chunksize-1)/2
npad=15
# INPUT FILES

locs = np.zeros(3)  #initialize np array that holds locations

# we read the csv file and initialize a numpy array that will hold the coordinates of the particle centers


def get_chunk_img(n1,index,chunkorigin,centerloc):

   # print(np.shape(n1))
   loc=chunkorigin
   # cen=centerloc
   # print(cen)
   # print(index)
   # print(loc)
   # print(loc+cen)
   # plt.imshow(n1[:,:,int(loc[2]+cen[2])])
   # plt.scatter(loc[0]+cen[0],loc[1]+cen[1])
   # plt.show()
   # plt.imshow(n1[0:50,100:200,int(loc[2]+cen[2])])
   # plt.show()
   # print(int(loc[0]+15),int(loc[0])+15, int(loc[1]), int(loc[2]))

   output = n1[int(loc[1]):int(loc[1]+chunksize),int(loc[0]):int(loc[0]+chunksize),int(loc[2]):int(loc[2]+chunksize)]
   size=np.shape(output)
   #print(size)
   while size != (chunksize,chunksize,chunksize):
       if size[2] != chunksize:
           #np.append(output,np.zeros((15,15)),axis=2)
           output=np.dstack((output, np.zeros((chunksize,chunksize))))
           size=np.shape(output)
       if size[0] != chunksize:

           output=np.concatenate((output, [np.zeros((chunksize,chunksize))]),axis=0)

           size=np.shape(output)
       if size[1] != chunksize:
           print(np.shape([np.zeros((chunksize,chunksize))]))
           output=np.concatenate((output, np.transpose([np.zeros((chunksize,chunksize))],(1,0,2))),axis=1)
           size=np.shape(output)
       #print(size)


   # f=0
   # while f<15:
   #      c=plt.imshow(output[:,:,f])
   #      plt.scatter([cen[0]],[cen[1]])
   #      print(cen[0:2])
   #      plt.show()
   #      f=f+1
   #      #time.sleep(0.1)
   # #print(output)
   return output


def LSQ(df,image):
      #background, image, diameter,
    # image=np.pad(image,(500,500))
    # LSQ_result = tp.refine_leastsq(df, image, diameter=11,
    #                             fit_function="disc",
    #                             param_val={"disc_size":0.2},
    #                             param_mode={"disc_size":"var"})

    # LSQ_result = tp.refine_leastsq(df, image,diameter=(11,11,9),fit_function="disc",bounds={'x':(2+npad,12+npad),'y':(2+npad,12+npad),'z':(2+npad,12+npad)})
    LSQ_result = tp.refine_leastsq(df, image,diameter=(11,11,9),fit_function="disc",bounds={'x':(17,26),'y':(17,26),'z':(17,26)})
    # i=0
    # while i<chunksize:
    #     plt.imshow(image[:,:,i])
    #     plt.scatter(LSQ_result['x'],LSQ_result['y'])
    #     plt.scatter(df['x'],df['y'])
    #     plt.show()
    #     i=i+1

    abba=np.shape(image)

    X, Y, Z = np.mgrid[0:abba[0],0:abba[1], 0:abba[2]]

    im=np.transpose(image,axes=[1,0,2])
    fig = go.Figure(data=go.Volume(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=im.flatten(),isomin=0.05,isomax=0.35,opacity=0.05, surface_count=27))
    fig.add_scatter3d(x=LSQ_result['x'],y=LSQ_result['y'],z=LSQ_result['z'],name="LSQ result")
    fig.add_scatter3d(x=df['x'],y=df['y'],z=df['z'],name='centroid')
    fig.show()
    fig, ax = plt.subplots()
    ax.imshow(image[:,:,npad+5])
    ax.scatter(LSQ_result['x'],LSQ_result['y'], label="LSQ result")
    ax.scatter(df['x'],df['y'],label="centroid")
    ax.legend()
    plt.show()
    return LSQ_result

#%%

# with open('tfrGel10212018A_shearRun10292018f_decon_hv01342_table.csv', mode='r') as csv_file:
with open('tfrGel10212018A_shearRun10292018f_locations_hv01342_sed_trackPy.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    row_count = sum(1 for row in csv_reader)
    print(row_count)
    #row_count=50
    locs = np.zeros((row_count-1,3))  #numpy array of center coordinates
    indexes = np.zeros(row_count-1)  #numpy array of indexes
    chunk_locs = np.zeros((row_count-1,3))  #numpy array of chunk coordinates to fetch image of particle given 13 x 13 x 11 size
    LSQcoords = np.zeros((row_count-1,3))

# we read the input csv file and fill in the coordinates into an xarray
with open('tfrGel10212018A_shearRun10292018f_locations_hv01342_sed_trackPy.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    hf = h5py.File('finalim.h5', 'r')
    hf.keys()
    n1 = hf.get('image_dataset')
    n1 = np.array(n1)

    n1 = np.transpose(n1, (1, 2, 0))
    for row in csv_reader:

        if line_count == 0: # this is the header row. Ignore this
            print(row)
            line_count += 1
        elif float(row[3])<17 or float(row[3])>432:  #skip row if out of x-bound
            print(row)
            #line_count += 1
        elif float(row[2])<17 or float(row[2])>432:  #skip row if out of y-bound
            print(row)
            #line_count += 1
        elif float(row[1])<24 or float(row[1])>175:  #skip row if out of z-bound
            print(row)
            #line_count += 1
        else:
            i=1
            while i <= 9:

                indexes[line_count-1]=line_count  #populate index array
                if i==1:
                    locs[line_count-1,2]=float(row[i])-17 #z coord  in pixels
                    zloc=math.floor(float(row[i])-17)-chunkhalf #in z slices
                    if zloc <0:
                        zloc=0
                    chunk_locs[line_count-1,2]=int(zloc)
                    locs[line_count-1,2]=locs[line_count-1,2] -int(zloc)
                if i==2:
                    locs[line_count-1,1]=float(row[i])-10 #y coord
                    yloc=math.floor(float(row[i])-10)-chunkhalf
                    if yloc<0:
                        yloc=0
                    chunk_locs[line_count-1,1]=int(yloc)
                    locs[line_count-1,1]=locs[line_count-1,1]-int(yloc)
                if i==3:
                    locs[line_count-1,0]=float(row[i])-10 #x coord
                    print(locs[line_count-1,0])
                    xloc=math.floor(float(row[i])-10)-chunkhalf
                    if xloc <0:
                        xloc=0
                    chunk_locs[line_count-1,0]=int(xloc)
                    locs[line_count-1,0]=locs[line_count-1,0]-int(xloc)
                i+=1

            img=get_chunk_img(n1,i,chunk_locs[line_count-1,:],locs[line_count,:])
            maximg=np.mean(img[int(locs[line_count-1,0])-2:int(locs[line_count-1,0])+2,int(locs[line_count-1,1])-2:int(locs[line_count-1,1])+2,int(locs[line_count-1,2])-2:int(locs[line_count-1,2])+2])

            # bb_dict={'x':(0,15),'y':(0,15),'z':(0,15)}
            # bb_dict={'x': (chunk_locs[line_count-1,0], chunk_locs[line_count-1,0]+chunksize), 'y': ((chunk_locs[line_count-1,1], chunk_locs[line_count-1,1]+chunksize)), 'z': ((chunk_locs[line_count-1,2], chunk_locs[line_count-1,2]+chunksize))}
            # d = {'p0_pos':[locs[line_count-1,0],locs[line_count-1,1],locs[line_count-1,2]],'pos_columns': ['x','y','z'], 'background': 0,'signal':maximg,'size':float(row[10])**(1/3)}
            print([[locs[line_count-1,0]],[locs[line_count-1,1]],[locs[line_count-1,2]]])
            df = pd.DataFrame(data=np.transpose(np.array([[locs[line_count-1,0]+npad],[locs[line_count-1,1]+npad],[locs[line_count-1,2]+npad]])),columns=['x','y','z'])
            df['signal'] = maximg
            df['background']=0
            df['disc_size']=0.2
            # df['size_x']=float(row[10])**(1/3)
            # df['size_y']=float(row[10])**(1/3)
            # df['size_z']=float(row[10])**(1/3)
            df['size_x']=3.4
            df['size_y']=3.4
            df['size_z']=2
            # print(bb_dict)
            print(row)
            bubs=LSQ(df, np.pad(img,15))
            print("bubs")
            print(bubs)

            LSQcoords[line_count-1,:]=[bubs['x'],bubs['y'],bubs['z']]
            print(LSQcoords[line_count-1,:])
            # LSQcoords[line_count-1]=LSQ(df, img, diameter=5,fit_function="disc",param_val={"signal":maximg,"background":0, "size":float(row[10])**(1/3)},bounds=bb_dict)
            line_count += 1

    print(f'Processed {line_count} lines.')





    ds=xr.Dataset(
        {
            "center_coords":(["index","dimension"],locs),
            "chunk_origin":(["index","dimension"],chunk_locs),
            # "LSQ_coords"
        },
        coords={
            "dims":(["dimension"],["x","y","z"]),
            "indexes":(["index"],indexes),

        },
    )
print(ds["center_coords"])
print(ds["chunk_origin"])

ds.to_netcdf("saved_chunk_xarray_ilastikv1.nc")
