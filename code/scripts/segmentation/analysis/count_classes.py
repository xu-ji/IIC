import argparse
import os
import pickle

import numpy as np

from code.utils.segmentation.data import make_Coco_dataloaders, \
  make_Potsdam_dataloaders

parser = argparse.ArgumentParser()
# {555} - with 1.5 on head_b & \cmt{512} & \cmt{545} - with 1.5 on head_b &
# \cmt{544} \\
parser.add_argument("--model_inds", type=int, nargs="+", required=True)
given_config = parser.parse_args()

for model_ind in given_config.model_inds:
  print(model_ind)
  reloaded_config_path = os.path.join("/scratch/shared/slow/xuji/iid_private",
                                      str(model_ind),
                                      "config.pickle")
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)

  if not hasattr(config, "using_IR"):
    config.using_IR = False

  if not hasattr(config, "no_sobel"):
    config.no_sobel = False

  if not hasattr(config, "use_uncollapsed_loss"):
    config.use_uncollapsed_loss = False

  if not hasattr(config, "save_multiple"):
    config.save_multiple = False

  if not hasattr(config, "use_doersch_datasets"):
    config.use_doersch_datasets = False

  assert (config.mode == "IID")  # same assign/test

  if "Coco" in config.dataset:
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_Coco_dataloaders(config)
  elif config.dataset == "Potsdam":
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_Potsdam_dataloaders(config)

  counts = np.zeros(config.gt_k)

  # count classes in mapping_assign dataloader
  for b_i, batch in enumerate(mapping_assignment_dataloader):
    imgs, flat_targets, mask = batch
    for c in xrange(config.gt_k):
      counts[c] += ((flat_targets == c) * mask).sum().item()

  print("counts")
  print(counts)
  print("proportions")
  print(counts / counts.sum())
