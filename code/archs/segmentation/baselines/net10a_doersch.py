import torch
import torch.nn as nn
import torch.nn.functional as F

from code.archs.cluster.vgg import VGGNet
from code.archs.segmentation.net10a import SegmentationNet10aTrunk, \
  SegmentationNet10a
from code.utils.segmentation.baselines.general import get_patches

__all__ = ["SegmentationNet10aDoersch"]


class DoerschHead(nn.Module):
  def __init__(self, config):
    super(DoerschHead, self).__init__()
    self.patch_side = config.doersch_patch_side

    self.siamese_branch = nn.Sequential(
      nn.Conv2d(in_channels=SegmentationNet10a.cfg[-1][0], out_channels=1024,
                kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True)
    )

    self.joint = nn.Sequential(
      nn.Linear(2 * 1024 * self.patch_side * self.patch_side, 1024),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(1024, 9)  # 9 gt positions, N, NE... NW.
    )

  def forward(self, patches1, patches2):
    patches1 = self.siamese_branch(patches1)
    patches2 = self.siamese_branch(patches2)

    ni, k, h, w = patches1.size()
    ni2, k2, h2, w2 = patches1.size()

    if not ((ni == ni2) and (k == k2) and (h == h2) and (w == w2) and \
              (h == self.patch_side) and (w == self.patch_side)):
      print (ni, k, h, w)
      print (ni2, k2, h2, w2)
      assert (False)

    # flatten all but first dim
    patches1 = patches1.contiguous()  # otherwise view may behave funny
    patches2 = patches2.contiguous()

    patches1 = patches1.view(patches1.size(0), -1)
    patches2 = patches2.view(patches2.size(0), -1)
    concatenated = torch.cat((patches1, patches2), dim=1)

    ni3, nf = concatenated.size()
    if not ((ni3 == ni) and (nf == (2 * 1024 * self.patch_side *
                                      self.patch_side))):
      print (ni, k, h, w)
      print (ni2, k2, h2, w2)
      print patches1.size()
      print patches2.size()
      print (ni3, nf)
      assert (False)

    return self.joint(concatenated)


class SegmentationNet10aDoersch(VGGNet):
  def __init__(self, config):
    super(SegmentationNet10aDoersch, self).__init__()

    self.patch_side = config.doersch_patch_side
    self.input_sz = config.input_sz
    self.features_sz = SegmentationNet10a.cfg[-1][0]

    print("SegmentationNet10aDoersch: %d %d %d" % (self.patch_side,
                                                   self.input_sz,
                                                   self.features_sz))

    self.features = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    self.doersch_head = DoerschHead(config)

    self._initialize_weights()

  def forward(self, x, centre=None, other=None, penultimate=False):
    x = self.features(x)
    x = F.interpolate(x, size=self.input_sz, mode="bilinear")

    if not penultimate:
      assert ((centre is not None) and (other is not None))
      patches1, patches2 = \
        get_patches(x, centre, other, self.patch_side)

      # predicted position distribution, no softmax - using
      # torch.CrossEntropyLoss
      # shape: bn, 9
      x = self.doersch_head(patches1, patches2)

    return x
