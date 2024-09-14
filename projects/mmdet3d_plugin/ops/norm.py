import torch
from mmcv.cnn import NORM_LAYERS
from mmcv.runner import force_fp32
from torch import nn as nn


@NORM_LAYERS.register_module()
class MyBN1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    @force_fp32(out_fp16=False)
    def forward(self, input):
        return super(MyBN1d, self).forward(input)

