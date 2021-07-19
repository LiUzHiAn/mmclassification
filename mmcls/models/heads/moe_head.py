import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .cls_head import ClsHead


@HEADS.register_module()
class MoEClsHead(ClsHead):
    """Mixture of experts as in paper `Temporal Concept Localization within Video
    using a Mixture of Context-Aware and Context-Agnostic Segment Classifier`

    网络的结果大致如下:

    feats_in --> feat_hidden(Attention Module) --> MoE logits -------------> overall logits
                         |-----------------------> MoE gating weights----|

    注意,在feat_hidden之后,可以选择使用Attetion模块(例如SE)

    Args:
        num_classes (int): Number of categories.
        num_mixtures (int): Number of experts.
        in_channels (int): Number of channels in the input feature map.
        hidden_channels (int): Number of channels in the hidden feature map.
        se_reduction (int): The degree of channel reduction factor of Squeeze and Excitation Module.
            This value should be a times of 2. If -1, then the SE module is not actually applied.
        drop_p (float): Dropout probability in Linear layer, default as 0.2.
        per_class (bool): 每个专家不同类的权重是否不一样, default as Fasle.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 num_mixtures,
                 in_channels,
                 hidden_channels,
                 se_reduction=-1,
                 drop_p=0.2,
                 per_class=False,  # 每个专家关于各个类别的权重是否一样
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(MoEClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_mixtures = num_mixtures
        self.hidden_channels = hidden_channels
        self.se_reduction = se_reduction
        self.per_class = per_class

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.ReLU(),
        )

        # SE模块
        self.se_gating = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels // se_reduction),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // se_reduction),
            nn.Linear(hidden_channels // se_reduction, hidden_channels),
            nn.Sigmoid()
        ) if se_reduction > 0 else None

        # MoE分类器头
        self.expert_fc = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(drop_p),
            nn.Linear(hidden_channels, num_classes * self.num_mixtures)
        )
        if self.per_class:  # 每个专家不同类的权重也不一样
            self.gating_fc = nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Dropout(drop_p),
                nn.Linear(hidden_channels, num_classes * (self.num_mixtures + 1))
            )  # contains one gate for the dummy 'expert' (always predict none)
        else:
            self.gating_fc = nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.Dropout(drop_p),
                nn.Linear(hidden_channels, (self.num_mixtures + 1))
            )  # contains one gate for the dummy 'expert' (always predict none)

    def simple_test(self, img):
        """Test without augmentation."""

        feats_hidden = self.fc(img)
        if self.se_gating is not None:
            gating = self.se_gating(feats_hidden)
            feats_hidden = feats_hidden * gating

        expert_logits = self.expert_fc(feats_hidden).view(-1, self.n_classes, self.num_mixtures)
        if self.per_class:
            expert_distributions = F.softmax(
                self.gating_fc(feats_hidden).view(
                    -1, self.num_classes, self.num_mixtures + 1
                ), dim=-1
            )
        else:
            expert_distributions = F.softmax(
                self.gating_fc(feats_hidden), dim=-1
            ).unsqueeze(1)
        cls_score = (expert_logits * expert_distributions[..., :self.num_mixtures]).sum(dim=-1)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        feats_hidden = self.fc(x)
        if self.se_gating is not None:
            gating = self.se_gating(feats_hidden)
            feats_hidden = feats_hidden * gating

        expert_logits = self.expert_fc(feats_hidden).view(-1, self.num_classes, self.num_mixtures)
        if self.per_class:
            expert_distributions = F.softmax(
                self.gating_fc(feats_hidden).view(
                    -1, self.n_classes, self.num_mixtures + 1
                ), dim=-1
            )
        else:
            expert_distributions = F.softmax(
                self.gating_fc(feats_hidden), dim=-1
            ).unsqueeze(1)
        logits = (expert_logits * expert_distributions[..., :self.num_mixtures]).sum(dim=-1)

        losses = self.loss(logits, gt_label)

        return losses
