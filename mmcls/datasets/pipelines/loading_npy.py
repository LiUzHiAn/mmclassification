import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadEmbeddingFromFile(object):
    """Load an embedding from file.

    Required keys are "emb_prefix" and "emb_info" (a dict that must contain the
    key "filename").

    """

    def __init__(self):
        super(LoadEmbeddingFromFile, self).__init__()

    def __call__(self, results):

        if results['emb_prefix'] is not None:
            filename = osp.join(results['emb_prefix'],
                                results['emb_info']['filename'])
        else:
            filename = results['emb_info']['filename']

        embedding = np.load(filename).astype(np.float32)
        # 如果是空,就用全0代替
        if len(embedding) == 0:
            embedding = np.zeros(200).astype(np.float32)

        results['filename'] = filename
        results['emb'] = embedding

        return results


@PIPELINES.register_module()
class LoadImageEmbeddingFromFile(object):
    """Load an image and the corresponding word embbedding from file.

    Required keys are "emb_prefix", "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        embedding = np.load(osp.join(results['emb_prefix'],
                                     results['emb_info']['filename'])).astype(np.float32)

        results["emb"] = embedding
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
