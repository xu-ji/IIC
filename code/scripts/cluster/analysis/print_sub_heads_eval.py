import argparse
import os
import pickle
import torch

import code.archs as archs
from code.utils.cluster.cluster_eval import get_subhead_using_loss
from code.utils.cluster.data import cluster_twohead_create_dataloaders

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_inds", type=int, nargs="+",
                      default=[570, 569, 640, 579, 685])
  parser.add_argument("--out_root", type=str,
                      default="/scratch/shared/slow/xuji/iid_private")

  given_config = parser.parse_args()

  for model_ind in given_config.model_inds:
    print("\n%d -------------------------------------------------" % model_ind)

    given_config.out_dir = os.path.join(given_config.out_root, str(model_ind))
    reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
      config = pickle.load(config_f)
    assert (config.model_ind == model_ind)

    if not hasattr(config, "twohead"):
      config.twohead = ("TwoHead" in config.arch)

    config.double_eval = False  # no double eval, not training (or saving config)

    net = archs.__dict__[config.arch](config)
    model_path = os.path.join(config.out_dir, "best_net.pytorch")
    net.load_state_dict(
      torch.load(model_path, map_location=lambda storage, loc: storage))
    net.cuda()
    net = torch.nn.DataParallel(net)

    dataloaders_head_A, dataloaders_head_B, \
    mapping_assignment_dataloader, mapping_test_dataloader = \
      cluster_twohead_create_dataloaders(config)

    if "MNIST" in config.dataset:
      sobel = False
      lamb = config.lamb_B
    else:
      sobel = True
      lamb = config.lamb

    get_subhead_using_loss(config, dataloaders_head_B, net, sobel, lamb,
                           compare=True)

main()