import shutil
import time
import os.path as osp

import mmcv
from mmcv import Config
from mmcls.apis import set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model

import mmcv
import numpy as np
import torch
from mmcls.models.classifiers import WordEmbeddingClassifier
from mmcls.models import build_classifier


def model_test():
    cfg = Config.fromfile('configs/img_word_emb_model_config.py')

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    print(f'Config:\n{cfg.pretty_text}')

    model = build_classifier(cfg.model)

    out = model(torch.ones(64, 3, 224, 224), torch.ones(64, 200), gt_label=torch.ones(64).long())
    print(out)


def dataset_test():
    cfg = Config.fromfile('configs/img_word_emb_dataset_config.py')
    print(f'Config:\n{cfg.pretty_text}')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    print(-1)


def run_test():
    cfg = Config.fromfile('configs/word_emb_run_config.py')

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for finetuning
    print(f'Config:\n{cfg.pretty_text}')

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 保存运行配置文件
    with open(osp.join(cfg.work_dir, "exp_conf.py"), "w") as fp:
        fp.write(cfg.pretty_text)
    print(-1)

    # Build the classifier
    model = build_classifier(cfg.model)
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Begin finetuning
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())


def swin_transformer_test():
    from mmcls.models import SwinTransformer
    import torch

    extra_config = dict(arch='tiny',
                        stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
                                                        'expansion_ratio': 3}),
                        auto_pad=True)
    self = SwinTransformer(**extra_config)
    inputs = torch.rand(1, 3, 224, 224)
    output = self.forward(inputs)
    print(output.shape)


if __name__ == '__main__':
    # run_test()
    # model_test()
    # dataset_test()
    swin_transformer_test()