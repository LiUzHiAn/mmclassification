_base_ = [
    './img_word_emb_model_config.py',
    './img_word_emb_dataset_config.py',
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 75])
runner = dict(type='EpochBasedRunner', max_epochs=100)


# checkpoint saving
checkpoint_config = dict(interval=10, max_keep_ckpts=5)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/kdxf_cls_task_multiModal_lr0.01_decayTwice'
