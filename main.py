import torch
import re
from blocks import MBConvBlock, FusedMBConvBlock, Steam, Head
import numpy as np
import torch.nn as nn

v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]


def decode_block_string(block_string):
    blocks = block_string.split('_')
    block_params = dict()
    for block in blocks:
        block_split = re.split(r'(\d.*)', block)
        key, value = block_split[:2]
        block_params[key] = value

    kernel_size = int(block_params['k']),
    num_repeat = int(block_params['r']),
    input_filters = int(block_params['i']),
    output_filters = int(block_params['o']),
    expand_ratio = int(block_params['e']),
    se_ratio = float(block_params['se']) if 'se' in block_params else None,
    strides = int(block_params['s']),
    conv_type = int(block_params['c']) if 'c' in block_params else 0

    block_config = {
        'kernel_size': kernel_size,
        'num_repeat': num_repeat,
        'input_filters': input_filters,
        'output_filters': output_filters,
        'expand_ratio': expand_ratio,
        'se_ratio': se_ratio,
        'strides': strides,
        'conv_type': conv_type
    }
    return block_config


def get_cfg_from_name(name):
    blocks_args = []
    cfg = dict()
    if name == 'efficientnetv2-s':
        # 83.9% @ 22M
        efficientnetv2_params = (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug')
        width, depth, train_size, eval_size, dropout, randaug, mix, aug = efficientnetv2_params[1:]
        for block_string in efficientnetv2_params[0]:
            blocks_args.append(decode_block_string(block_string))
    cfg['blocks_args'] = blocks_args
    cfg['width_coefficient'] = width
    cfg['depth_coefficient'] = depth
    cfg['dropout_rate'] = dropout
    cfg['data_format']='channels_last'
    cfg['depth_divisor'] = 8
    cfg['act_fn'] = 'silu'
    cfg['min_depth'] = 8

    return cfg


class EffnetV2Model(nn.Module):
    def __init__(self, model_name='efficientnetv2-s', include_top=True, n_channels=3):
        super().__init__()
        self.model_name = model_name
        self.include_top = include_top
        self.n_channels = n_channels
        self._build()

    def _build(self):
        conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}
        self.blocks = []
        self.cfg = get_cfg_from_name(self.model_name)
        for block_arg in self.cfg['blocks_args']:
            type = block_arg['conv_type']
            self.blocks.append(conv_block[type](block_arg))
        self.Steam = Steam(self.cfg, self.n_channels, self.cfg['block_args'][0]['input_filters'])
        self.Head = Head(self.cfg)

    def forward(self, x):
        x = self.Steam(x)
        for idx, block in enumerate(self.blocks):
            is_reduction = False # reduction flag for blocks after the stem layer
            x = block(x)
        x = self.Head(x)
        return x

