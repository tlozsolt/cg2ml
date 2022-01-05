import abc
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional

import numpy as np
import pandas as pd


@dataclass(frozen = True)
class AbstractSegmentationResult(Sequence): # Inherit from Sequence to enforce the sequence protocol on all child classes

    @abc.abstractproperty
    def centroids(self) -> np.array: # use numpy for now, later maybe change to xarray or custom class for 3D coordinates
        pass
    
    # Implement part of the sequence protocol that doesn't rely on what format segmentation result is in
    def __getitem__(self, index) -> np.array:
        return self.centroids[index]


@dataclass(frozen = True)
class CSVSegmentationResult(AbstractSegmentationResult):
    filepath: Union[str, Path]
    _data: pd.DataFrame = field(init = False) # So that instance constructions require only filepath

    def __post_init__(self):
        self._data = self._read_delimited_file()

    # Implement __eq__ to make this and all child classes hashable
    # Two CSV segmentation results are equal if they both reference the same file
    def __eq__(self, other) -> bool:
        if isinstance(other, CSVSegmentationResult):
            return self.filepath == other.filepath
        else:
            return False

    # Abstract method for reading the csv file at self.filepath and returning the results as a pandas DataFrame
    @abc.abstractmethod
    def _read_delimited_file(self) -> pd.DataFrame:
        pass

    # Implement the remaining part of the sequence protocol
    def __len__(self) -> int:
        return len(self._data)

    # Implement couple useful pandas functions for improved ergonomics
    def head(self) -> None:
        self._data.head()

    def tail(self) -> None:
        self._data.tail()


@dataclass(frozen = True)
class IlastikSegmentationResult(CSVSegmentationResult):

    # Concrete implementation for the csv file read.
    def _read_delimited_file(self):
        return pd.read_csv(self._data, delimiter = ',')

    # Concrete implementation of the centroids property
    @property
    def centroids(self) -> np.array:
        return self._data.loc[:, [f'Center of the object_{i}' for i in range(3)]] # xyz order


@dataclass(frozen = True)
class TrackPySegmentationResult(CSVSegmentationResult):
    
    # Concrete implementation for the csv file read. Note the subtle difference in the delimiter (different between Ilastik and TrackPy)
    def _read_delimited_file(self):
        return pd.read_csv(self._data, delimiter = ' ')

    # Concrete implementation of the centroids property
    @property
    def centroids(self) -> np.array:
        return self._data.loc[:, 3:-1:0] # xyz order
  