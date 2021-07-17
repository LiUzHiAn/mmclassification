# model settings
model = dict(
    type='WordEmbeddingClassifier',
    backbone=dict(
        type='WordEmbeddingBackbone',
        dim_embedding=200,
        out_channels=1024,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=137,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
