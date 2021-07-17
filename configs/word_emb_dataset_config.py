# dataset settings
dataset_type = 'WordEmbeddingDataset'

train_pipeline = [
    dict(type='LoadEmbeddingFromFile'),
    dict(type='ToTensor', keys=['emb']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='MyCollect', keys=['emb', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadEmbeddingFromFile'),
    dict(type='ToTensor', keys=['emb']),
    dict(type='MyCollect', keys=['emb'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/training_set_txt',
        ann_file='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/train.txt',
        classes='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/classes.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/training_set_txt',
        ann_file='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/val.txt',
        classes='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/classes.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/test_set',
        ann_file='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/test.txt',
        classes='/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/classes.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=(1, 5)))
