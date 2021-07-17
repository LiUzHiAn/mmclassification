import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv import color_val
from mmcv.runner import BaseModule

# TODO import `auto_fp16` from mmcv and delete them from mmcls
try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import auto_fp16

import copy
import warnings

import torch

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .base import BaseClassifier

from mmcv import Config


@CLASSIFIERS.register_module()
class WordEmbeddingClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(WordEmbeddingClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def extract_feat(self, emb):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(emb)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, emb, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            emb (Tensor): of shape (N,#dim) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            emb, gt_label = self.augments(emb, gt_label)

        x = self.extract_feat(emb)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def forward_test(self, emb, **kwargs):
        """
        Args:
            emb (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxD,
                which contains all embeddings in the batch.
        """
        if isinstance(emb, torch.Tensor):
            emb = [emb]
        for var, name in [(emb, 'emb')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(emb) == 1:
            return self.simple_test(emb[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def simple_test(self, emb):
        """Test without augmentation."""
        x = self.extract_feat(emb)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        return self.head.simple_test(x)

    @auto_fp16(apply_to=('emb',))
    def forward(self, emb, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(emb, **kwargs)
        else:
            return self.forward_test(emb, **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['emb'].data))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['emb'].data))

        return outputs


@CLASSIFIERS.register_module()
class ImageWordEmbeddingClassifier(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 img_backbone,
                 emb_backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageWordEmbeddingClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.img_backbone = build_backbone(img_backbone)
        self.img_backbone.init_weights()  # 显式初始化权重
        self.emb_backbone = build_backbone(emb_backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def extract_feat(self, img, emb):
        """Directly extract features from the backbone + neck."""
        x_img = self.img_backbone(img)
        x_emb = self.emb_backbone(emb)
        if self.with_neck:
            x_img = self.neck(x_img)

        x = torch.cat([x_img, x_emb], dim=1)
        return x

    def forward_train(self, img, emb, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N,C,H,W) encoding input images.
                Typically these should be mean centered and std scaled.
            emb (Tensor): of shape (N,#dim) encoding input word embedding.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img, emb)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def forward_test(self, img, emb, **kwargs):
        """
        Args:
            emb (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxD,
                which contains all embeddings in the batch.
        """
        if isinstance(img, torch.Tensor):
            img = [img]
        if isinstance(emb, torch.Tensor):
            emb = [emb]
        for var, name in [(img, 'img'), (emb, 'emb')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(emb) == 1 and len(img) == 1:
            return self.simple_test(img[0], emb[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def simple_test(self, img, emb):
        """Test without augmentation."""
        x = self.extract_feat(img, emb)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        return self.head.simple_test(x)

    @auto_fp16(apply_to=('img', 'emb',))
    def forward(self, img, emb, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, emb, **kwargs)
        else:
            return self.forward_test(img, emb, **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['emb'].data))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['emb'].data))

        return outputs
