import torch.nn as nn

# suphead2 but with batchnorm instead of relu

__all__ = ["SupHead5"]


class SupHead5(nn.Module):
  def __init__(self, net_features, dlen=None, gt_k=None):
    super(SupHead5, self).__init__()

    self.trunk = net_features

    net_head = nn.Sequential(
      nn.Linear(dlen, 2048),
      nn.BatchNorm1d(2048),
      nn.ReLU(),
      nn.Linear(2048, gt_k)
    )

    net_head.cuda()
    self.head = net_head

    for m in self.head.modules():
      print(m)
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

  def forward(self, x, penultimate_features=False):
    x = self.trunk(x, trunk_features=True,
                   penultimate_features=penultimate_features)
    x = self.head(x)  # no softmax
    return x
