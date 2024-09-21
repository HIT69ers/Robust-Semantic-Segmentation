#1. model config
checkpoint_teacher = 'pretrain/Teacher_SegFormer_B3_city.pth'
checkpoint_backbone = 'pretrain/SCT-B_Pretrain.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Distill',
    pretrained=None,
    backbone=dict(
        type='SCTNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint= checkpoint_backbone
        ),
        base_channels=64,
        spp_channels=128),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=256,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='AU_SCTHead',
            in_channels=128,
            channels=128,
            dropout_ratio=0.0,
            in_index=1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4)),
        dict(
            type='VitGuidanceHead',
            init_cfg=dict(
                type='Pretrained',
                checkpoint= checkpoint_teacher),
            in_channels=256,
            channels=256,
            base_channels=64,
            in_index=2,
            num_classes=19,
            loss_decode=dict(type='AlignmentLoss', loss_weight=[3, 15, 15, 15]))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#2. data config
_base_ = '../../_base_/datasets/bdd100k_foggy.py'

#3. optimizer config
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.0125,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=1e-06,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
find_unused_parameters = True
auto_resume = True
seed = 1209821543
