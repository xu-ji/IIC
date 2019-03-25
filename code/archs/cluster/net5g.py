import torch.nn as nn

from residual import BasicBlock, ResNet, ResNetTrunk

# resnet34 and full channels

__all__ = ["ClusterNet5g"]


class ClusterNet5gTrunk(ResNetTrunk):
  def __init__(self, config):
    super(ClusterNet5gTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    block = BasicBlock
    layers = [3, 4, 6, 3]

    in_channels = config.in_channels
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    if config.input_sz == 96:
      avg_pool_sz = 7
    elif config.input_sz == 64:
      avg_pool_sz = 5
    elif config.input_sz == 32:
      avg_pool_sz = 3
    print("avg_pool_sz %d" % avg_pool_sz)

    self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

  def forward(self, x, penultimate_features=False):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    if not penultimate_features:
      # default
      x = self.layer4(x)
      x = self.avgpool(x)

    x = x.view(x.size(0), -1)

    return x


class ClusterNet5gHead(nn.Module):
  def __init__(self, config):
    super(ClusterNet5gHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.num_sub_heads = config.num_sub_heads

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(512 * BasicBlock.expansion, config.output_k),
      nn.Softmax(dim=1)) for _ in xrange(self.num_sub_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in xrange(self.num_sub_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results


class ClusterNet5g(ResNet):
  def __init__(self, config):
    # no saving of configs
    super(ClusterNet5g, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNet5gTrunk(config)
    self.head = ClusterNet5gHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False, trunk_features=False,
              penultimate_features=False):
    x = self.trunk(x, penultimate_features=penultimate_features)

    if trunk_features:  # for semisup
      return x

    x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
    return x
