import torch.nn as nn
import torch.nn.functional as F

from ..cluster.vgg import VGGTrunk, VGGNet

__all__ = ["SegmentationNet10a"]


# From first iteration of code, based on VGG11:
# https://github.com/xu-ji/unsup/blob/master/mutual_information/networks
# /vggseg.py

class SegmentationNet10aTrunk(VGGTrunk):
  def __init__(self, config, cfg):
    super(SegmentationNet10aTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    assert (config.input_sz % 2 == 0)

    self.conv_size = 3
    self.pad = 1
    self.cfg = cfg
    self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
      else 3

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)  # do not flatten
    return x


class SegmentationNet10aHead(nn.Module):
  def __init__(self, config, output_k, cfg):
    super(SegmentationNet10aHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.cfg = cfg
    num_features = self.cfg[-1][0]

    self.num_sub_heads = config.num_sub_heads

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Conv2d(num_features, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.Softmax2d()) for _ in xrange(self.num_sub_heads)])

    self.input_sz = config.input_sz

  def forward(self, x):
    results = []
    for i in xrange(self.num_sub_heads):
      x_i = self.heads[i](x)
      x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
      results.append(x_i)

    return results


class SegmentationNet10a(VGGNet):
  cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
         (512, 2), (512, 2)]  # 30x30 recep field

  def __init__(self, config):
    super(SegmentationNet10a, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    self.head = SegmentationNet10aHead(config, output_k=config.output_k,
                                       cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x):
    x = self.trunk(x)
    x = self.head(x)
    return x
