from blocks import MBConvBlock, FusedMBConvBlock, Steam, Head
import torch.nn as nn
from utils import round_filters
from utils import get_cfg_from_name
import copy


class EffnetV2Model(nn.Module):
    def __init__(self, model_name='efficientnetv2-s', include_top=False, n_channels=3,
                 n_classes=None):
        super().__init__()
        self.model_name = model_name
        self.include_top = include_top
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._build()

    def _build(self):
        conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}
        self.blocks = []
        self.cfg = get_cfg_from_name(self.model_name)
        if self.n_classes:
            self.cfg['num_classes'] = self.n_classes
        for ind, block_arg in enumerate(self.cfg['blocks_args']):
            type = block_arg['conv_type']
            for n in range(block_arg['num_repeat']):
                block_arg_layer = copy.deepcopy(block_arg)
                if n != 0:
                    block_arg_layer['input_filters'] = block_arg['output_filters']
                    block_arg_layer['strides'] = 1
                self.blocks.append(conv_block[type](block_arg_layer, self.cfg))
        self.blocks = nn.ModuleList(self.blocks)
        self.Steam = Steam(self.cfg, self.n_channels, self.cfg['blocks_args'][0]['input_filters'])
        self.Head = Head(self.cfg)
        if self.include_top and self.cfg['num_classes']:
            self._fc = nn.Linear(round_filters(self.cfg.get('feature_size') or 1280, self.cfg), self.cfg['num_classes'])
        else:
            self._fc = None

    def forward(self, x):
        x = self.Steam(x)
        for idx, block in enumerate(self.blocks):
            survival_prob = self.cfg['survival_prob']
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(self.blocks)
            x = block(x, survival_prob=survival_prob)

        out = self.Head(x)
        if self._fc:
            out = out.flatten(start_dim=1)
            out = self._fc(out)
        return out
