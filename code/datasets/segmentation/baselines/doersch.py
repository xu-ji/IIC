import os
import pickle
from sys import stdout as sysout

import numpy as np
import torch
from torch.utils import data

"""
Do the random colour dropping
our doerschdataset class is passed a real base class in the main script,
calls its getitem, if include_rgb, sub out rgb channels for noise

https://www.cv-foundation.org/openaccess/content_iccv_2015/papers
/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf

"randomly drop 2 of the 3 color channels from
each patch (color dropping), replacing the dropped colors
with Gaussian noise (standard deviation 1/100 the standard
deviation of the remaining channel)."

"""


class DoerschDataset(data.Dataset):
  def __init__(self, config, base_dataset):
    # base_dataset already constructed with config
    super(DoerschDataset, self).__init__()
    self.base_dataset = base_dataset

    if config.include_rgb and self.base_dataset.purpose == "train":
      self.input_sz = config.input_sz
      self.stddev_fname = os.path.join(config.doersch_stats,
                                       "%s_stats.pickle" % config.dataset)
      if not os.path.exists(self.stddev_fname):
        self.make_stats_file()

      with open(self.stddev_fname, "rb") as f:
        stats = pickle.load(f)
        self.stddev = stats["stddev"]
        self.mean = stats["mean"]
      print("created Doersch dataset wrapping %s, stddev %s" % (config.dataset,
                                                                self.stddev))
      sysout.flush()

  def __getitem__(self, index):
    tup = self.base_dataset.__getitem__(index)

    if self.base_dataset.purpose == "test":
      return tup
    else:
      assert (self.base_dataset.purpose == "train")
      assert (self.base_dataset.single_mode)
      img, _ = tup

      channel_pair = np.random.choice(3, size=2, replace=False)
      remaining_channel = 3 - channel_pair.sum()

      mean = float(self.mean[remaining_channel])
      stddev = float(self.stddev[remaining_channel] / 100.)
      noise = torch.zeros((2, self.input_sz, self.input_sz),
                          dtype=torch.float32).cuda()
      noise = noise.normal_(mean, stddev)

      # noise = np.random.normal(loc=mean,
      #                         scale=stddev,
      #                         size=(2, self.input_sz, self.input_sz))
      # noise = torch.from_numpy(noise)

      for c in xrange(2):
        img[channel_pair[c], :, :] = noise[c, :, :]

      return (img,) + tup[1:]

  def __len__(self):
    return self.base_dataset.__len__()

  def make_stats_file(self):
    print("making stats")
    sysout.flush()

    # get mean and stddev of all rgb values in set
    num_imgs = len(self.base_dataset)
    pixels = np.zeros((num_imgs * self.input_sz * self.input_sz, 3),
                      dtype=np.float32)
    count = 0
    for i in xrange(num_imgs):
      if i % (num_imgs / 10) == 0:
        print("img %d out of %d" % (i, num_imgs))
        sysout.flush()

      tup = self.base_dataset[i]
      if self.base_dataset.purpose == "train":
        assert (self.base_dataset.single_mode)
        img, mask = tup
      else:
        assert (False)

      img = img.permute(1, 2, 0)  # features last

      img = img[:, :, :3]  # rgb - then grey/sobel if using, then ir if using

      num = mask.sum()
      img = img.masked_select(mask.unsqueeze(2)).view(num, 3)  # n, c

      assert (len(img.shape) == 2)
      img = img.cpu().numpy()

      pixels[count:count + num, :] = img
      count += num

    pixels = pixels[:count]
    stddev = np.std(pixels, axis=0)
    mean = np.mean(pixels, axis=0)

    print("got rgb mean %s and std %s" % (stddev, mean))
    with open(self.stddev_fname, "wb") as outfile:
      pickle.dump({"stddev": stddev, "mean": mean}, outfile)
