import torch.nn as nn


class VGGTrunk(nn.Module):
  def __init__(self):
    super(VGGTrunk, self).__init__()

  def _make_layers(self, batch_norm=True):
    layers = []
    in_channels = self.in_channels
    for tup in self.cfg:
      assert (len(tup) == 2)

      out, dilation = tup
      sz = self.conv_size
      stride = 1
      pad = self.pad  # to avoid shrinking

      if out == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      elif out == 'A':
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
      else:
        conv2d = nn.Conv2d(in_channels, out, kernel_size=sz,
                           stride=stride, padding=pad,
                           dilation=dilation, bias=False)
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(out,
                                            track_running_stats=self.batchnorm_track),
                     nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out

    return nn.Sequential(*layers)


class VGGNet(nn.Module):
  def __init__(self):
    super(VGGNet, self).__init__()

  def _initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
