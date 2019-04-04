from .net10a import SegmentationNet10aHead, SegmentationNet10aTrunk, \
  SegmentationNet10a
from ..cluster.vgg import VGGNet

__all__ = ["SegmentationNet10aTwoHead"]


class SegmentationNet10aTwoHead(VGGNet):
  def __init__(self, config):
    super(SegmentationNet10aTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    self.head_A = SegmentationNet10aHead(config, output_k=config.output_k_A,
                                         cfg=SegmentationNet10a.cfg)
    self.head_B = SegmentationNet10aHead(config, output_k=config.output_k_B,
                                         cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x, head="B"):
    x = self.trunk(x)
    if head == "A":
      x = self.head_A(x)
    elif head == "B":
      x = self.head_B(x)
    else:
      assert (False)

    return x
