# model settings
model = dict(
    type='ImageWordEmbeddingClassifier',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    emb_backbone=dict(
        type='WordEmbeddingBackbone',
        dim_embedding=200,
        out_channels=1024,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=137,
        in_channels=2048 + 1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
