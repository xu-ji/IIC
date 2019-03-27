import argparse
import os
import pickle

import torch
from torch import nn as nn

import code.archs as archs

# Print the modules of each network including if batchnorm is used or not

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")

given_config = parser.parse_args()

given_config.out_dir = os.path.join(given_config.out_root,
                                    str(given_config.model_ind))

reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
print("Loading restarting config from: %s" % reloaded_config_path)
with open(reloaded_config_path, "rb") as config_f:
  config = pickle.load(config_f)
assert (config.model_ind == given_config.model_ind)

if not hasattr(config, "num_sub_heads"):
  config.num_sub_heads = config.num_heads

if not hasattr(config, "twohead"):
  config.twohead = ("TwoHead" in config.arch)

net = archs.__dict__[config.arch](config)
model_path = os.path.join(config.out_dir, "best_net.pytorch")
net.load_state_dict(
  torch.load(model_path, map_location=lambda storage, loc: storage))
net.cuda()
net = torch.nn.DataParallel(net)

# for name, param in net.named_parameters():
#  print("%s, %s, %s" % (name, param.requires_grad, param.data.shape))


print(net)

print("--------------")

for m in net.modules():
  if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    if not (m.track_running_stats):
      print(m)
      print("not tracking stats for this batchnorm")
      # print("... found a batchnorm, tracking: %s" % m.track_running_stats)
