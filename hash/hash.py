from typing import Union

import numpy as np
import yaml
import pandas as pd
import os, glob
import math

class hash():
    """
    A python class to create a spatial hash used for chunking an xyzt image data
    into a searchable table of hash parameters.
    This class does not open or load any images, and all image information required
    for creating the hash (size of image, size of hash, etc) is read from a
    parameter file.

    Attributes:
        - self.metaData : parsed yaml file
        - self.metaDataPath: the path to the metaData yaml file when the instance was created
        - self.hashDimensions: dictionary of hash dimensions
        - self.hash : a hash table of positive integers mapping into:
                      coordinate time ,z,y,x and crop parameters for the hash Table
        - self.hash_df: the same hash table as self.hash, but as a pandas array.

    Methods:
        - fetchMeta : returns the metaData information for the queried keyword
        - fetchPaths: return a dictionary (or pandas array) with keywords for each
                      pipeline step and the associated paths to the images.

    **************
    Useage:
    - get the hash indices for all time point t=17
      >> inst.hash_df []

    -zsolt Jul 30, 2021
    """
    def __init__(self, metaDataPath):

        # load yaml metaData file
        self.metaDataPath = metaDataPath
        with open(metaDataPath, 'r') as s:
            self.metaData = yaml.load(s, Loader=yaml.SafeLoader)

        # dimensions of the image
        imgParam = self.metaData['image']
        dim = imgParam['dim']
        timeSteps, zDim, yDim, xDim = dim['t'], dim['z'], dim['y'], dim['x']

        # check that xy dimensions are equal and constant for sed and gel.
        # raise error otherwise
        hash_param = self.metaData['hash']
        dim = set()
        overlap = set()
        for mat in ['sed', 'gel']:
            for coord in ['x','y']:
                dim.add(hash_param[mat]['dim'][coord])
                overlap.add(hash_param[mat]['minOverlap'][coord])

        # are the sets of len 1? They should be
        if len(dim) == 1: hashDim_xy = dim.pop()
        else: raise ValueError("Hash dimensions in x and y must be equal for both sed and gel")

        if len(overlap) == 1: hashOverlap_xy = overlap.pop()
        else: raise ValueError("Hash overlap in x and y must be equal for both sed and gel")

        # compute the number of hashes in x and y
        # [ ] check that this works, july 30 2021
        Nx = math.ceil((xDim-hashOverlap_xy)/(hashDim_xy - hashOverlap_xy))
        Ny = math.ceil((yDim-hashOverlap_xy)/(hashDim_xy - hashOverlap_xy))

        # now deal with the z coordinate
        # z chunks are computed separately for gel and sediment and based on min overlap and z dimension in gel and sed portions
        gelSedLoc = imgParam['gelSedimentLocation']
        sed_overlap_gel = hash_param['sed']['interface_pxOverlap']
        gel_overlap_sed = hash_param['gel']['interface_pxOverlap']

        # Get the z bounds for gel and sediment taking into account the min overlap of gel into sed and vice-versa
        gelZ_dim = (0, gelSedLoc + gel_overlap_sed)
        sedZ_dim = (gelSedLoc - sed_overlap_gel, zDim)

        # how big are the chunks in z for sed and gel?
        hashDim_z_gel = hash_param['gel']['dim']['z']
        hashDim_z_sed = hash_param['sed']['dim']['z']

        # compute the number of chunks
        Nz_sed: Union[float, int] = math.ceil((sedZ_dim[1] - sedZ_dim[0] - sed_overlap_gel )/(hashDim_z_sed - sed_overlap_gel))
        Nz_gel = math.ceil((gelZ_dim[1] - gelZ_dim[0] - gel_overlap_sed )/(hashDim_z_gel - gel_overlap_sed))

        # create an empty dictionary with keys spanning the hash size
        N = timeSteps * (Nz_sed + Nz_gel) * Nx * Ny
        self.metaData['hashDimension'] = {'N': N,
                                          't':timeSteps,
                                          'Nz_sed': Nz_sed,
                                          'Nz_gel': Nz_gel,
                                          'Nx': Nx,
                                          'Ny': Ny}
        # Generate, for each hashValue, the xyz cutoffs...permutations on 3 choices, each, on x and y
        # to generate x points, we start at left (ie 0), \
        # add the cropped dimension until we exceed the full dimension, \
        # and then shift the rightmost point to end at rightmost extent. Same for y
        def hashCenters(blockSize, left, right, N):
            """Returns a list of N evenly spaced points separated by """
            if blockSize*N < (right-left): raise KeyError("hash dimension is too small")
            return np.linspace(np.floor(left + blockSize/2), np.ceil(right - blockSize/2),num=N)

        centerPts = {'x': hashCenters(hashDim_xy,0,xDim,Nx),
                     'y': hashCenters(hashDim_xy,0,yDim,Ny),
                     'z_sed': hashCenters(hashDim_z_sed, sedZ_dim[0], sedZ_dim[1], Nz_sed),
                     'z_gel': hashCenters(hashDim_z_gel, gelZ_dim[0], gelZ_dim[1], Nz_gel)}

        xCrop = [(int(c-hashDim_xy/2),int(c + hashDim_xy/2)) for c in centerPts['x']]

        zCrop = []  # list of all the z-positions for gel and sediment
        material = []  # list of str specifiying whether material is sed or gel

        for c in centerPts['z_gel']:
            zCrop.append((int(c - hashDim_z_gel/2),
                          int(c + hashDim_z_gel/2)))
            material.append('gel')
        for c in centerPts['z_sed']:
            zCrop.append((int(c - hashDim_z_sed/2), \
                          int(c + hashDim_z_sed/2)))
            material.append('sed')

        hashTable = {}
        # Do we need a hashInverse table? Yes because the job flow requires numbers to be passed to the
        # cg2ml refactor: I am going to decide that we do not need an inverse hash table as hash_df can be
        # queried using pandas.
        """
        # job queing. We could in principle make one hash with
        # key:values like '9,3,16':(9,3,16)
        # as opposed to:
        #      hash: '68':(9,3,16) and
        #      inverseHash: '9,3,16':68
        # but we cannot pass '9,3,16' to the job queuing as a hashValue
        # additionally, converting '68' to 68 is simplest possible type conversion \
        # that will most probably work as the default
        # on any level of the job control whether in python, bash or whatever the SLURM is written in.
        # This type conversion is also handled in this class by using the method queryHash
        """
        keyCount = 0
        for t in range(timeSteps):
            for z in range(len(zCrop)):
                for y in range(len(xCrop)):
                    for x in range(len(xCrop)):
                        hashTable[str(keyCount)] = dict([('xyztCropTuples', [xCrop[x], xCrop[y], zCrop[z], t])])
                        hashTable[str(keyCount)]['material'] = material[z]
                        hashTable[str(keyCount)]['index'] = (x, y, z, t)
                        keyCount += 1
        """
    I have to check that 'the bounds are working the way I think they are. \
    In particular the overlaps need to added/subtracted somewhere and this, \
    I think, depends on the direction and start point from which I am counting.
    To generate z, we split case into sediment and gel, propagating from gel bottom \
    up until we past gel/sediment and propagate down from TEM grid until we get past gel/sediment.\
    No shifting at the end necessary since this is just going to influence \
    how much gel/sediment overlap is in each chunk. \
    We should make these into functions along with wrapping the dimension counting into functions as well.\
    Additionally, there is likely a better yaml implementation to put dimensions used into \
    an easier to use data structure. \
    Can we view line comments in yaml after importing? Definately yes if you make the comments \
    part of the data structure. Then everything is a value and text string...
    I feel like the rest of this is a nested for loop over 
    indices:  time, no. of gel and  no of sed chunks, xy chunks, 
    """
        # create hash_df to allow for indexing using pandas.
        # I dont want to touch self.hash as I would likely have to rewrite some of the access functions
        # This is however a better more elegant data structure for query in the hash table
        # what the hashValues for first and last time points?
        # >>> hash_df[(hash_df['t'] == 0) |  (hash_df['t']==89)].index
        hash_df = pd.DataFrame(hashTable).T
        hash_df['x'] = hash_df['index'].apply(lambda x: x[0])
        hash_df['y'] = hash_df['index'].apply(lambda x: x[1])
        hash_df['z'] = hash_df['index'].apply(lambda x: x[2])
        hash_df['t'] = hash_df['index'].apply(lambda x: x[3])

        # type cast hash_df index to integer as currently the index is str from hashTable keys that require
        # nonmutable type.
        _idx = hash_df.index.map(int)
        hash_df.set_index(_idx, inplace=True)

        self.hash_df = hash_df
        self.hash = hashTable

def test(metaPath='../metaData_template.yml'):
    return hash(metaPath)

if __name__ == '__main__':
    inst = test('../metaData_template.yml')
#%%
    inst = test('../metaData_template.yml')
