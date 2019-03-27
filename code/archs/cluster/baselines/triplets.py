import torch.nn as nn

from code.archs.cluster.net5g import ClusterNet5gTrunk
from code.archs.cluster.net6c import ClusterNet6c, ClusterNet6cTrunk
from code.archs.cluster.residual import BasicBlock, ResNet
from code.archs.cluster.vgg import VGGNet

__all__ = ["TripletsNet5g", "TripletsNet6c"]


class TripletsNet5gHead(nn.Module):
  def __init__(self, config):
    super(TripletsNet5gHead, self).__init__()

    # no softmax, done in loss
    self.head = nn.Linear(512 * BasicBlock.expansion, config.output_k)

  def forward(self, x, kmeans_use_features=False):
    if kmeans_use_features:
      return x
    else:
      return self.head(x)


class TripletsNet5g(ResNet):
  def __init__(self, config):
    # no saving of configs
    super(TripletsNet5g, self).__init__()

    self.trunk = ClusterNet5gTrunk(config)
    self.head = TripletsNet5gHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False):
    x = self.trunk(x)
    x = self.head(x, kmeans_use_features=kmeans_use_features)
    return x


class TripletsNet6cHead(nn.Module):
  def __init__(self, config):
    super(TripletsNet6cHead, self).__init__()

    self.cfg = ClusterNet6c.cfg
    num_features = self.cfg[-1][0]

    if config.input_sz == 24:
      features_sp_size = 3
    elif config.input_sz == 64:
      features_sp_size = 8

    # no softmax, done in loss
    self.head = nn.Linear(num_features * features_sp_size * features_sp_size,
                          config.output_k)

  def forward(self, x, kmeans_use_features=False):
    if kmeans_use_features:
      return x
    else:
      return self.head(x)


class TripletsNet6c(VGGNet):
  def __init__(self, config):
    # no saving of configs
    super(TripletsNet6c, self).__init__()

    self.trunk = ClusterNet6cTrunk(config)
    self.head = TripletsNet6cHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False):
    x = self.trunk(x)
    x = self.head(x, kmeans_use_features=kmeans_use_features)
    return x
