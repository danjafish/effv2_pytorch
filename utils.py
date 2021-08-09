import torch
from torch.nn import functional
import re
import functools


def activation_fn(features, act_fn):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return torch.nn.functional.silu(features)
  elif act_fn == 'silu_native':
    return features * torch.sigmoid(features)
  elif act_fn == 'hswish':
    return features * torch.nn.functional.relu6(features + 3) / 6
  elif act_fn == 'relu':
    return torch.nn.functional.relu(features)
  elif act_fn == 'relu6':
    return torch.nn.functional.relu6(features)
  elif act_fn == 'elu':
    return torch.nn.functional.elu(features)
  elif act_fn == 'leaky_relu':
    return torch.nn.functional.leaky_relu(features)
  elif act_fn == 'selu':
    return torch.nn.functional.selu(features)
  elif act_fn == 'mish':
    return features * torch.nn.functional.tanh(torch.nn.functional.softmax(features))
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))


def round_filters(filters, mconfig, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = mconfig['width_coefficient']
  divisor = mconfig['depth_divisor']
  min_depth = mconfig['min_depth']
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  return int(new_filters)


def get_act_fn(act_fn):
  if not act_fn:
    return torch.nn.SiLU
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  return act_fn


v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]

v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]

v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]


def decode_block_string(block_string):
    blocks = block_string.split('_')
    block_params = dict()
    for block in blocks:
        block_split = re.split(r'(\d.*)', block)
        key, value = block_split[:2]
        block_params[key] = value

    kernel_size = int(block_params['k'])
    num_repeat = int(block_params['r'])
    input_filters = int(block_params['i'])
    output_filters = int(block_params['o'])
    expand_ratio = int(block_params['e'])
    se_ratio = float(block_params['se']) if 'se' in block_params else None
    strides = int(block_params['s'])
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
    elif name == 'efficientnetv2-m':
        efficientnetv2_params = (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug')
    elif name == 'efficientnetv2-l':
        efficientnetv2_params = (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug')
    elif name == 'efficientnetv2-l':
        efficientnetv2_params = (v2_xl_block, 1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug')
    else:
        raise ('Model name is not supported')

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
    cfg['survival_prob'] = 0.8
    cfg['local_pooling'] = False
    cfg['conv_dropout'] = None

    return cfg


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = inputs.shape[0]
  random_tensor = survival_prob
  random_tensor += torch.rand((batch_size, 1, 1, 1), dtype=inputs.dtype)
  binary_tensor = torch.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output
