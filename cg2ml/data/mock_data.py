from functools import lru_cache

import numpy as np
from numpy.random import default_rng

# TODO: Maybe make the mock data generator into a class
# Will need to accept volume dimensions, number of random particles, particle radius, center, noise type, blob type


def draw_random_coords_ND(N_coords, bounds):
    return 