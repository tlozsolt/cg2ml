## Note all desired operations work row by row
## Therefore, it is better to refactor the logic in terms of rows, which will have and additional benefit of parallelization becoming easier

#%%
import numpy as np
import h5py
import xarray as xr
import plotly.graph_objects as go

def read_volume_from_hdf5(volume_data_path):
    hdf_file = h5py.File(volume_data_path)
    volume_data = hdf_file['image_dataset']
    return volume_data

full_volume = read_volume_from_hdf5('finalim.h5') # z, y, x load images
# %%
import pandas as pd
csv = pd.read_csv('ilastik_predictions_locRefine_111621.csv', delimiter = ',') #load centroid data

csv.head() #remove header

# %%
def subset_coord_columns(dataframe, n):
    return dataframe.iloc[:, n:n+3]

def calculate_chunk_centers(centroid_coord):
    return np.int_(np.floor(centroid_coord))

centroid_coord = subset_coord_columns(csv,13).to_numpy() #get centroid coordinates x y z
# centroid_coord = np.transpose(centroid_coord, axes=[1,0,2])
# centroid_coord = centroid_coord.astype(float)
chunk_centers = calculate_chunk_centers(centroid_coord) #get nearest pixel x,y,z
#%%
chunk_centers.shape

# %%
def is_chunk_inbounds(center_coords, halfwidths,edge, volume_shape):
    inbounds_left = np.all(halfwidths+edge <= center_coords, axis = -1)
    inbounds_right = np.all(center_coords < volume_shape - halfwidths-edge, axis = -1)
    return np.logical_and(inbounds_left, inbounds_right)
halfwidth=6
chunk_halfwidths = np.array([6, 6, 6])
edge_snip_amount = np.array([24,17,17]) #the input to elastic has more pixels than
inbound_inds = is_chunk_inbounds(chunk_centers, chunk_halfwidths, edge_snip_amount,np.array(full_volume.shape))
valid_centers = chunk_centers[inbound_inds] #xyz
valid_centroids = centroid_coord[inbound_inds] #xyz
# %%
def calculate_chunk_indices(center_coords, halfwidths):
    return tuple(slice(s-ds, s+ds+1) for s, ds in zip([center_coords[2],center_coords[1],center_coords[0]], halfwidths))


# %%
def calculate_chunk_centroid(original_centroid_coord, chunk_halfwidths):
    subpixel_coord, _ = np.modf(original_centroid_coord)
    return subpixel_coord+chunk_halfwidths

chunk_centroids = calculate_chunk_centroid(valid_centroids, chunk_halfwidths)
# %%
import trackpy as tp
def fit_disc(chunk, chunk_centroid_coord, pad,halfwidth):
    # print(chunk_centroid_coord.reshape(-1,1)+pad)
    fit_params = pd.DataFrame(chunk_centroid_coord.reshape(1,-1) + pad, columns = ['x', 'y', 'z'])
    fit_params['background'] = 0
    fit_params['signal'] = np.mean(chunk[halfwidth-1:halfwidth+3,halfwidth-1:halfwidth+3,halfwidth-1:halfwidth+3])

    fixed_params = {'disc_size': 0.2, 'size_x': 3.4, 'size_y': 3.4, 'size_z': 2}

    LSQ_result = tp.refine_leastsq(fit_params, np.pad(chunk, pad), diameter = (9,11,11), fit_function = "disc", param_val = fixed_params)

    return LSQ_result


# %%
# chunk_centroids[0]
chunk_indices_x=[]
chunk_indices_y=[]
chunk_indices_z=[]
#%%
list_out=[]
pad=15

for idx, j in enumerate(valid_centers):
    chunk_ind=calculate_chunk_indices(j, chunk_halfwidths)
    print(idx)
    chunk = full_volume[chunk_ind]
    # chunk=np.transpose(chunk,axes=[1,0,2])

    list_out.append(fit_disc(chunk, chunk_centroids[idx], pad,halfwidth))
    chunk_indices_x.append(chunk_ind[2].start)
    chunk_indices_y.append(chunk_ind[1].start)
    chunk_indices_z.append(chunk_ind[0].start)


#%%

LSQ_out=pd.concat(list_out)
LSQ_out["chunk_origin_x"]=chunk_indices_x
LSQ_out["chunk_origin_y"]=chunk_indices_y
LSQ_out["chunk_origin_z"]=chunk_indices_z
LSQ_out["chunk_centroid_x"]=valid_centroids[:,0]
LSQ_out["chunk_centroid_y"]=valid_centroids[:,1]
LSQ_out["chunk_centroid_z"]=valid_centroids[:,2]
# print(LSQ_out)
xr = xr.Dataset.from_dataframe(LSQ_out)
xr.to_netcdf('JoonLSQout.nc')
