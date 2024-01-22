from typing import Any, Optional

import SimpleITK as sitk
import h5py
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class LoadH5(MapTransform):
    def __init__(self, path_key, keys: KeysCollection):
        super().__init__(keys)
        self.path_key = path_key

    def __call__(self, data):
        d = dict(data)
        h5_file = h5py.File(d[self.path_key])
        for key in self.keys:
            d[key] = h5_file[key][()]
        # d.pop(self.path_key)
        return d


class LoadImageITKd(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = sitk.GetArrayFromImage(sitk.ReadImage(data[key], outputPixelType=sitk.sitkFloat32))
        # d.pop(self.path_key)
        return d


class RandCropOrPad(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection):
        super(Randomizable, self).__init__(keys)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandCropOrPad":
        pass

    def randomize(self, data: Any) -> None:
        if self.R.random() < self.prob:
            pass

    def __call__(self, data):
        self.randomize(data)


class CropOrPad(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        h5_file = h5py.File(d[self.path_key])
        for key in self.keys:
            d[key] = h5_file[key]
        d.pop(self.path_key)
        return d
