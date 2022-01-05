## Note all desired operations work row by row
## Therefore, it is better to refactor the logic in terms of rows, which will have and additional benefit of parallelization becoming easier

#%%
import numpy as np
import h5py

def read_volume_from_hdf5(volume_data_path):
    hdf_file = h5py.File(volume_data_path)
    volume_data = hdf_file['image_dataset']
    return volume_data

full_volume = read_volume_from_hdf5('../finalim.h5') # z, y, x
# %%
#%%
import pandas as pd
csv = pd.read_csv('../../Data/locations/tfrGel10212018A_shearRun10292018f_locations_hv01342_sed_trackPy.csv', delimiter = ' ')
csv.head()
# %%
def subset_coord_columns(dataframe):
    return dataframe.iloc[:, 1:4]

def calculate_chunk_centers(centroid_coord):
    return np.int_(np.floor(centroid_coord))

centroid_coord = subset_coord_columns(csv).to_numpy()
chunk_centers = calculate_chunk_centers(centroid_coord)
#%%
chunk_centers.shape

# %%
def is_chunk_inbounds(center_coords, halfwidths, volume_shape):
    inbounds_left = np.all(halfwidths <= center_coords, axis = -1)
    inbounds_right = np.all(center_coords < volume_shape - halfwidths, axis = -1)
    return np.logical_and(inbounds_left, inbounds_right)

chunk_halfwidths = np.array([11, 8, 8])
inbound_inds = is_chunk_inbounds(chunk_centers, chunk_halfwidths, np.array(full_volume.shape))
valid_centers = chunk_centers[inbound_inds]
valid_centroids = centroid_coord[inbound_inds]
# %%
def calculate_chunk_indices(center_coords, halfwidths):
    return tuple(slice(s-ds, s+ds+1) for s, ds in zip(center_coords, halfwidths))

chunk0 = full_volume[calculate_chunk_indices(valid_centers[0], chunk_halfwidths)]
# %%
def calculate_chunk_centroid(original_centroid_coord, chunk_halfwidths):
    subpixel_coord, _ = np.modf(original_centroid_coord)
    return subpixel_coord+chunk_halfwidths

chunk_centroids = calculate_chunk_centroid(valid_centroids, chunk_halfwidths)
# %%
import trackpy as tp
def fit_disc(chunk, chunk_centroid_coord, pad):
    fit_params = pd.DataFrame(chunk_centroid_coord.reshape(1, -1) + pad, columns = ['z', 'y', 'x'])
    fit_params['background'] = 0
    fit_params['signal'] = 1.0
    
    fixed_params = {'disc_size': 0.2, 'size_x': 3.4, 'size_y': 3.4, 'size_z': 2}
    
    LSQ_result = tp.refine_leastsq(fit_params, np.pad(chunk, pad), diameter = (11,11,9), fit_function = "disc", param_val = fixed_params)
    return LSQ_result


# %%
chunk_centroids[0]
#%%
out = fit_disc(chunk0, chunk_centroids[0], 15)
#%%
out
# %%
csv_lsq = pd.read_csv('../../Data/locations/tfrGel10212018A_shearRun10292018f_locations_hv01342_sed_trackPy_lsqRefine.csv', delimiter = ' ')
csv_lsq.head()
lsq_coord = subset_coord_columns(csv_lsq).to_numpy()
valid_lsq_coord = lsq_coord[inbound_inds]
# %%
valid_lsq_coord[0]
# %%
def create_coordinate_array(side, scales = (1, 1, 1)):
    #center = (side-1)/2
    #pixels = np.arange(side) - center
    pixels = np.arange(side)
    coords = [pixels*scale for scale in scales]
    return np.stack(np.meshgrid(*coords), axis = -1)


def gaussianND(coord, center, radius):
    Dr2 = np.sum((coord-center)**2, axis = -1)
    return np.exp(-Dr2/radius**2)

# %%
test_coord = create_coordinate_array(30)
# %%
test_coord.shape
# %%
true_center = np.array([20.7, 10.3, 8.5])
out_test = gaussianND(test_coord, true_center, radius = 5)
# %%
import plotly.graph_objects as go
values = out_test
#values = out.numpy()
#values = dataset[1][0].numpy()
Z, Y, X = test_coord[..., 0], test_coord[..., 1], test_coord[..., 2]
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=7, # needs to be a large number for good volume rendering
    ))
#fig.add_scatter3d(x = [true_center[2]], y =  [true_center[1]], z = [true_center[0]], name = 'true center')
fig.show()

# %%
dict(zip(('size_x', 'size_y', 'size_z'),(1, 2, 3)))
# %%
