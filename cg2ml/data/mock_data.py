from functools import lru_cache

import numpy as np
from numpy.random import default_rng

# TODO: Maybe make the mock data generator into a class
# Will need to accept volume dimensions, number of random particles, particle radius, center, noise type, blob type


def draw_random_coords_ND(N_coords, bounds):
    return 


from functools import lru_cache

import numpy as np

@lru_cache(maxsize = 1)
def create_coordinate_array(side, scales = (0.115, 0.115, 0.15)):
    center = (side-1)/2
    pixels = np.arange(side) - center
    
    coords = [pixels*scale for scale in scales]
    return np.stack(np.meshgrid(*coords), axis = -1)


def gaussianND(coord, center, radius):
    Dr2 = np.sum((coord-center)**2, axis = -1)
    return np.exp(-Dr2/radius**2)

#%%
from numpy.random import default_rng
N_data = 5000
rng = default_rng()
means = rng.uniform(size = (N_data, 1, 1, 1, 3))
means = (means-0.5)*0.4 # means will be random vectors in the box bounded by [-0.2, -0.2, -0.2] and [0.2, 0.2, 0.2]
#%%
sigma = 0.2
coords = create_coordinate_array(15)
vols = gaussianND(np.expand_dims(coords, axis = 0), means, sigma).astype(np.float32)
#%%
y_tensor = torch.as_tensor(np.squeeze(means), dtype = torch.float32)
x_tensor = torch.as_tensor(np.expand_dims(vols, axis = 1), dtype = torch.float32)
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)