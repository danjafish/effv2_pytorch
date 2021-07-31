from torch import nn
import torch
import utils
from utils import round_filters


class MBConvBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


class FusedMBConvBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


# inp_channels - number of channels in input
# out_channels - number of channels in output
class Steam(nn.Module):
    def __init__(self, cfg, inp_channels, out_channels):
        super().__init__()
        self._conv_steam =nn.Conv2d(
            in_channels=inp_channels,
            out_channels=round_filters(out_channels, cfg),
            kernel_size =3,
            stride=2,
            padding='same',
            bias=False
            # TODO set initial weights
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
            in_channels=self.cfg['block_args'][-1]['output_filters'],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
            # TODO set initial weights
        )

        self._norm = nn.BatchNorm2d(round_filters(cfg.get('feature_size') or 1280, cfg))
        self._act = utils.get_act_fn(cfg['act_fn'])
        self._avg_pooling = nn.AvgPool2d(kernel_size=out_channels)
        self._dropout = nn.Dropout(cfg.dropout_rate) if cfg.dropout_rate > 0 else None
        self.h_axis, self.w_axis =[2,3]

    def forward(self, x):
        outputs = self._act(self._norm(self._conv_head(x)))
        if self.cfg.get('local_pooling'):
            outputs = self._avg_pooling(outputs) #TODO fix this
            if self._dropout:
                outputs = self._dropout(outputs)
            if self._fc: # TODO where are no fc in head, how it works?
                outputs = torch.squeeze(outputs, self.h_axis)
                outputs = torch.squeeze(outputs, self.w_axis)
        else:
            outputs = self._avg_pooling(outputs)  # TODO fix this
            outputs = self._dropout(outputs)
        return  outputs