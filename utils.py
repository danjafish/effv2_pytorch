import torch
from torch.nn import functional
import functools


def activation_fn(features, act_fn):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return #torch.nn.swish?
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
  multiplier = mconfig.width_coefficient
  divisor = mconfig.depth_divisor
  min_depth = mconfig.min_depth
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