from __future__ import print_function

import argparse
import pickle
import sys
from datetime import datetime

import matplotlib
import torch

matplotlib.use('Agg')
import os

import code.archs as archs
from code.utils.cluster.general import nice
from code.utils.segmentation.segmentation_eval import \
  segmentation_eval
from code.utils.segmentation.data import segmentation_create_dataloaders

# Clone any old model (from config and best_net) and re-evaluate, including
# finding 1-1 mapping from output channels to ground truth clusters.

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--old_model_ind", type=int, required=True)
parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")

config = parser.parse_args()

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
old_out_dir = os.path.join(config.out_root, str(config.old_model_ind))

if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

reloaded_config_path = os.path.join(old_out_dir, "config.pickle")
print("Loading restarting config from: %s" % reloaded_config_path)
with open(reloaded_config_path, "rb") as config_f:
  old_config = pickle.load(config_f)
assert (old_config.model_ind == config.old_model_ind)

if not hasattr(old_config, "batchnorm_track"):
  old_config.batchnorm_track = True

if not hasattr(old_config, "num_sub_heads"):
  old_config.num_sub_heads = old_config.num_heads

if not hasattr(old_config, "use_doersch_datasets"):
  old_config.use_doersch_datasets = False

with open(os.path.join(old_config.out_dir, "config.pickle"), 'wb') as outfile:
  pickle.dump(old_config, outfile)

with open(os.path.join(old_config.out_dir, "config.txt"), "w") as text_file:
  text_file.write("%s" % old_config)

# Model ------------------------------------------------------

dataloaders_head_A, mapping_assignment_dataloader, mapping_test_dataloader = \
  segmentation_create_dataloaders(old_config)
dataloaders_head_B = dataloaders_head_A  # unlike for clustering datasets

net = archs.__dict__[old_config.arch](old_config)

net_state = torch.load(os.path.join(old_config.out_dir, "best_net.pytorch"),
                       map_location=lambda storage, loc: storage)
net.load_state_dict(net_state)
net.cuda()
net = torch.nn.DataParallel(net)

stats_dict = segmentation_eval(old_config, net,
                               mapping_assignment_dataloader=mapping_assignment_dataloader,
                               mapping_test_dataloader=mapping_test_dataloader,
                               sobel=(not old_config.no_sobel),
                               using_IR=old_config.using_IR,
                               return_only=True)

acc = stats_dict["best"]

config.epoch_stats = [stats_dict]
config.epoch_acc = [acc]
config.epoch_avg_subhead_acc = stats_dict["avg"]

print("Time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
sys.stdout.flush()

with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
  pickle.dump(config, outfile)

with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
  text_file.write("%s" % config)
