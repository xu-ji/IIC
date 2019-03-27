import torch.nn as nn

from vgg import VGGTrunk, VGGNet

# 4h but for cifar, 24x24

__all__ = ["ClusterNet6c"]


class ClusterNet6cTrunk(VGGTrunk):
  def __init__(self, config):
    super(ClusterNet6cTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.conv_size = 5
    self.pad = 2
    self.cfg = ClusterNet6c.cfg
    self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
      else 3

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)
    bn, nf, h, w = x.size()
    x = x.view(bn, nf * h * w)
    return x


class ClusterNet6cHead(nn.Module):
  def __init__(self, config):
    super(ClusterNet6cHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.num_sub_heads = config.num_sub_heads

    self.cfg = ClusterNet6c.cfg
    num_features = self.cfg[-1][0]

    if config.input_sz == 24:
      features_sp_size = 3
    elif config.input_sz == 64:
      features_sp_size = 8

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(num_features * features_sp_size * features_sp_size,
                config.output_k),
      nn.Softmax(dim=1)) for _ in xrange(self.num_sub_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in xrange(self.num_sub_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results


class ClusterNet6c(VGGNet):
  cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
         (256, 1), ('M', None), (512, 1)]

  def __init__(self, config):
    super(ClusterNet6c, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNet6cTrunk(config)
    self.head = ClusterNet6cHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False, trunk_features=False,
              penultimate_features=False):
    if penultimate_features:
      print("Not needed/implemented for this arch")
      exit(1)

    x = self.trunk(x)

    if trunk_features:  # for semisup
      return x

    x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
    return x
