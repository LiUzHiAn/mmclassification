import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class ConcatNeck(nn.Module):
    """将img embedding 和 word embedding concat起来
    """

    def __init__(self, dim=2):
        super(ConcatNeck, self).__init__()

    def init_weights(self):
        pass

    def forward(self, img_feats, word_feats):
        out = torch.cat([img_feats, word_feats], dim=1)
        return out
