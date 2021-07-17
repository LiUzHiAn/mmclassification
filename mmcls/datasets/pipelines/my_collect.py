from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from PIL import Image

from ..builder import PIPELINES


@PIPELINES.register_module()
class MyCollect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.

    Returns:
        dict: The result dict contains the following keys
                - keys in``self.keys``
    """

    def __init__(self,
                 keys):
        self.keys = keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys})'
