import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class WordEmbeddingBackbone(BaseBackbone):
    """使用word2vec的特帧作为输入,然后用几层全连接来提取特征

    The input for WordEmbeddingBackbone is the embedding vector.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, dim_embedding, out_channels=1024, num_classes=-1):
        super(WordEmbeddingBackbone, self).__init__()
        self.num_classes = num_classes
        self.dim_embedding = dim_embedding
        self.out_channels = out_channels

        self.features = nn.Sequential(
            nn.Linear(dim_embedding, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(out_channels, num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return x
