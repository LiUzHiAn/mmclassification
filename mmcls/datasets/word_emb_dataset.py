import codecs
import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, master_only

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix


@DATASETS.register_module()
class WordEmbeddingDataset(BaseDataset):
    """纯用word embedding"""

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            # The ann_file is the annotation files we generated. (filename cls_id)
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'emb_prefix': self.data_prefix}
                info['emb_info'] = {'filename': filename.replace(".jpg", ".npy")}  # 同名不同后缀
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


@DATASETS.register_module()
class ImageWordEmbeddingDataset(BaseDataset):
    """图像 + word embedding"""

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            # The ann_file is the annotation files we generated. (filename cls_id)
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                # self.data_prefix == "training_set"
                info = {'emb_prefix': self.data_prefix + "_txt", 'img_prefix': self.data_prefix}
                info['emb_info'] = {'filename': filename.replace(".jpg", ".npy")}  # 同名不同后缀
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]