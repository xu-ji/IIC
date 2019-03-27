import torch.nn as nn

from net6c import ClusterNet6c, ClusterNet6cTrunk
from vgg import VGGNet

__all__ = ["ClusterNet6cTwoHead"]


class ClusterNet6cTwoHeadHead(nn.Module):
  def __init__(self, config, output_k, semisup=False):
    super(ClusterNet6cTwoHeadHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.cfg = ClusterNet6c.cfg
    num_features = self.cfg[-1][0]

    self.semisup = semisup

    if config.input_sz == 24:
      features_sp_size = 3
    elif config.input_sz == 64:
      features_sp_size = 8

    if not semisup:
      self.num_sub_heads = config.num_sub_heads

      # is default (used for iid loss)
      # use multi heads
      # include softmax
      self.heads = nn.ModuleList([nn.Sequential(
        nn.Linear(num_features * features_sp_size * features_sp_size, output_k),
        nn.Softmax(dim=1)) for _ in xrange(self.num_sub_heads)])
    else:
      self.head = nn.Linear(num_features * features_sp_size * features_sp_size,
                            output_k)

  def forward(self, x, kmeans_use_features=False):

    if not self.semisup:
      results = []
      for i in xrange(self.num_sub_heads):
        if kmeans_use_features:
          results.append(x)  # duplicates
        else:
          results.append(self.heads[i](x))
      return results

    else:
      return self.head(x)


class ClusterNet6cTwoHead(VGGNet):
  cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
         (256, 1), ('M', None), (512, 1)]

  def __init__(self, config):
    super(ClusterNet6cTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNet6cTrunk(config)

    self.head_A = ClusterNet6cTwoHeadHead(config, output_k=config.output_k_A)

    semisup = (hasattr(config, "semisup") and
               config.semisup)
    print("semisup: %s" % semisup)

    self.head_B = ClusterNet6cTwoHeadHead(config, output_k=config.output_k_B,
                                          semisup=semisup)

    self._initialize_weights()

  def forward(self, x, head="B", kmeans_use_features=False,
              trunk_features=False,
              penultimate_features=False):
    if penultimate_features:
      print("Not needed/implemented for this arch")
      exit(1)

    # default is "B" for use by eval code
    # training script switches between A and B

    x = self.trunk(x)

    if trunk_features:  # for semisup
      return x

    # returns list or single
    if head == "A":
      x = self.head_A(x, kmeans_use_features=kmeans_use_features)
    elif head == "B":
      x = self.head_B(x, kmeans_use_features=kmeans_use_features)
    else:
      assert (False)

    return x
