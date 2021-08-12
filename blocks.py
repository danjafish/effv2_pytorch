from torch import nn
import torch
import utils
from utils import round_filters
import copy


class SE(nn.Module):
    def __init__(self, cfg, inp_channels, se_filters, output_filters):
        super(SE, self).__init__()
        self.local_pooling = cfg['local_pooling']
        self.data_format = cfg['data_format']
        self.act = utils.get_act_fn(cfg['act_fn'])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self._se_reduce = torch.nn.Conv2d(
            in_channels=inp_channels,
            out_channels=se_filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=True
        )

        self._se_expand = torch.nn.Conv2d(
            in_channels=se_filters,
            out_channels=output_filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=True
        )

    def forward(self, x):
        h_axis, w_axis = [2, 3]
        if self.local_pooling:
            se_tensor = self.avg_pool(x).view(x.size()[0], x.size()[1])
        else:
            se_tensor = torch.mean(x, [h_axis, w_axis], keepdim=True)

        se_tensor = self._se_expand(self.act(self._se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x


# class DepthwiseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias):
#         super(DepthwiseConv, self).__init__()
#         # TODO check stride in conv layer
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=use_bias, stride=stride, padding=1)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out


class MBConvBlock(nn.Module):
    def __init__(self, block_arg, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.block_arg = copy.deepcopy(block_arg)
        self.in_channels = block_arg['input_filters']
        self._local_pooling = cfg.get('local_pooling')
        self._data_format = cfg['data_format']
        self._channel_axis = 1 if self._data_format == 'channels_first' else -1
        self._act = utils.get_act_fn(cfg['act_fn'])
        self._has_se = (
            (self.block_arg['se_ratio'] is not None) and
            (0 < self.block_arg['se_ratio'] <= 1))

        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        filters = self.block_arg['input_filters'] * self.block_arg['expand_ratio']
        kernel_size = self.block_arg['kernel_size']
        in_channels = self.in_channels
        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if self.block_arg['expand_ratio'] != 1:
            self.expand_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=1,
                stride=1,
                padding='same',
                bias=False
            )

            self.norm0 = nn.BatchNorm2d(filters)

        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=kernel_size, groups=in_channels,
                                        bias=False, stride=self.block_arg['strides'], padding=1)

        self.norm1 = nn.BatchNorm2d(filters)

        if self._has_se:
            num_reduced_filters = max(1, int(self.block_arg['input_filters'] * self.block_arg['se_ratio']))
            self._se = SE(self.cfg, filters, num_reduced_filters, filters)
        else:
            self._se = None

        out_channels = self.block_arg['output_filters']

        self.project_conv = torch.nn.Conv2d(
            in_channels=filters,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

    def residual(self, inp, x, survival_prob):
        # TODO странное условие
        if (self.block_arg['strides'] == 1) and (self.block_arg['input_filters'] == self.block_arg['output_filters']):
            if survival_prob:
                x = utils.drop_connect(x, True, survival_prob)  # TODO change not training behaviour
            x = torch.add(x, inp)

        return x

    def forward(self, x, survival_prob=None):
        inputs = x
        if self.block_arg['expand_ratio'] != 1:
            x = self._act(self.norm0(self.expand_conv(x)))
            x = self._act(self.norm1(self.depthwise_conv(x)))
        if self.cfg['conv_dropout'] and self.block_arg['expand_ratio'] > 1:
            x = torch.nn.Dropout(self.cfg['conv_dropout'])(x)
        if self._se:
            self._se(x)
        x = self.norm2(self.project_conv(x))
        x = self.residual(inputs, x, survival_prob)
        # print('after MBConv ', x.shape)
        return x


class FusedMBConvBlock(MBConvBlock):
    def _build(self):
        filters = self.block_arg['input_filters'] * self.block_arg['expand_ratio']
        kernel_size = self.block_arg['kernel_size']
        in_channels = self.in_channels
        if self.block_arg['expand_ratio'] != 1:  # TODO check
            self.expand_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=self.block_arg['strides'],
                padding=1,
                bias=False
            )

            self.norm0 = nn.BatchNorm2d(filters)

        if self._has_se:
            num_reduced_filters = max(1, int(self.block_arg['input_filters'] * self.block_arg['se_ratio']))
            self._se = SE(self.cfg, filters, num_reduced_filters, filters)
        else:
            self._se = None

        out_channels = self.block_arg['output_filters']
        self.project_conv = torch.nn.Conv2d(
            in_channels=filters,
            out_channels=out_channels,
            kernel_size=1 if self.block_arg['expand_ratio'] != 1 else kernel_size,
            stride=1 if self.block_arg['expand_ratio'] != 1 else self.block_arg['strides'],
            padding='same',
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, survival_prob=None):
        inputs = x
        if self.block_arg['expand_ratio'] != 1:
            x = self._act(self.norm0(self.expand_conv(x)))

        if self.cfg['conv_dropout'] and self.block_arg['expand_ratio'] > 1:
            x = torch.nn.Dropout(self.cfg['conv_dropout'])(x)

        if self._se:
            self._se(x)

        x = self.norm1(self.project_conv(x))

        if self.block_arg['expand_ratio'] == 1:
            x = self._act(x)

        x = self.residual(inputs, x, survival_prob)
        # print('after FusedConv ', x.shape)
        return x


# inp_channels - number of channels in input
# out_channels - number of channels in output
class Steam(nn.Module):
    def __init__(self, cfg, inp_channels, out_channels):
        super().__init__()
        self._conv_steam = nn.Conv2d(
            in_channels=inp_channels,
            out_channels=round_filters(out_channels, cfg),
            kernel_size=3,
            stride=2,
            padding=1,  # TODO check padding
            bias=False
        )

        self._norm = nn.BatchNorm2d(round_filters(out_channels, cfg))
        self._act = utils.get_act_fn(cfg['act_fn'])

    def forward(self, x):
        return self._act(self._norm(self._conv_steam(x)))


class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        out_channels = round_filters(cfg.get('feature_size') or 1280, cfg)
        self._conv_head = nn.Conv2d(
            in_channels=self.cfg['blocks_args'][-1]['output_filters'],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        self.h_axis, self.w_axis = [2, 3]
        self._norm = nn.BatchNorm2d(round_filters(cfg.get('feature_size') or 1280, cfg))
        self._act = utils.get_act_fn(cfg['act_fn'])
        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self._dropout = nn.Dropout(cfg['dropout_rate']) if cfg['dropout_rate'] > 0 else None
        self._fc = None  # TODO check it (no such parameter in original)


    def forward(self, x):
        outputs = self._act(self._norm(self._conv_head(x)))
        if self.cfg.get('local_pooling'):
            outputs = self._avg_pooling(outputs)  # TODO fix this
            if self._dropout:
                outputs = self._dropout(outputs)
            if self._fc: # TODO where are no fc in head, how it works?
                outputs = torch.squeeze(outputs, self.h_axis)
                outputs = torch.squeeze(outputs, self.w_axis)
        else:
            outputs = self._avg_pooling(outputs)
            outputs = self._dropout(outputs)
        return outputs