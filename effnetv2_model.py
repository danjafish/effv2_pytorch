from blocks import MBConvBlock, FusedMBConvBlock, Steam, Head
import torch.nn as nn
from utils import get_cfg_from_name


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
        for ind, block_arg in enumerate(self.cfg['blocks_args']):
            type = block_arg['conv_type']
            self.blocks.append(conv_block[type](block_arg, self.cfg))
        print(len(self.blocks))
        self.Steam = Steam(self.cfg, self.n_channels, self.cfg['blocks_args'][0]['input_filters'])
        self.Head = Head(self.cfg)

    def forward(self, x):
        out = self.Steam(x)
        for idx, block in enumerate(self.blocks):
            is_reduction = False  # reduction flag for blocks after the stem layer
            if (idx == (len(self.blocks)-1)) or (self.blocks[idx+1].block_arg['strides'] > 1):
                is_reduction = True
            survival_prob = self.cfg['survival_prob']
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(self.blocks)
            print(idx)
            out = block(out, survival_prob=survival_prob)

        out = self.Head(out)
        if self._fc:
            out = self._fc(out)
        return out
