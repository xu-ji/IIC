from __future__ import print_function

import argparse
import os
import pickle
from sys import stdout as sysout

import torch

import code.archs as archs
from code.utils.cluster.cluster_eval import cluster_subheads_eval
from code.utils.cluster.data import make_STL_data, make_CIFAR_data
from code.utils.cluster.transforms import \
  sobel_make_transforms

# Reassess IID+ models by doing the mapping_assign with smaller numbers of
# labelled images

# to reassess model as originally done, set new_assign_set_szs_pc to [1.0]

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--new_assign_set_szs_pc", type=float, nargs="+",
                    default=[1.0])  # 0.01, 0.02, 0.05, 0.1, 0.5
parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")

parser.add_argument("--use_eval", default=False, action="store_true")
parser.add_argument("--dont_save", default=False, action="store_true")
parser.add_argument("--rewrite", default=False, action="store_true")

config = parser.parse_args()
if config.rewrite:
  assert (not config.dont_save)

new_assign_set_szs_pc = config.new_assign_set_szs_pc
print("given new_assign_set_szs_pc: %s" % new_assign_set_szs_pc)
sysout.flush()

given_config = config
reloaded_config_path = os.path.join(given_config.out_root,
                                    str(given_config.model_ind),
                                    "config.pickle")
print("Loading restarting config from: %s" % reloaded_config_path)
with open(reloaded_config_path, "rb") as config_f:
  config = pickle.load(config_f)
assert (config.model_ind == given_config.model_ind)

assert (config.mode == "IID+")

assert (config.output_k >= config.gt_k)
assert (config.eval_mode == "orig")
if "CIFAR" in config.dataset:
  assert (config.train_partitions == [True])
  assert (config.mapping_assignment_partitions == [True])
  assert (config.mapping_test_partitions == [False])
elif config.dataset == "STL10":
  assert (config.train_partitions == ["train+unlabeled"])
  assert (config.mapping_assignment_partitions == ["train"])
  assert (config.mapping_test_partitions == ["test"])

# append to old results
if not hasattr(config, "assign_set_szs_pc_acc") or given_config.rewrite:
  print("resetting config.assign_set_szs_pc_acc to empty")
  config.assign_set_szs_pc_acc = {}

for pc in new_assign_set_szs_pc:
  print("doing %f" % pc)
  sysout.flush()

  # datasets produce either 2 or 5 channel images based on config.include_rgb
  tf1, tf2, tf3 = sobel_make_transforms(config,
                                        cutout=config.cutout,
                                        cutout_p=config.cutout_p,
                                        cutout_max_box=config.cutout_max_box)

  if config.dataset == "STL10":
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_STL_data(config, tf1, tf2, tf3, truncate_assign=True, truncate_pc=pc)
  elif (config.dataset == "CIFAR10") or (config.dataset == "CIFAR100") or \
    (config.dataset == "CIFAR20"):
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_CIFAR_data(config, tf1, tf2, tf3, truncate_assign=True,
                      truncate_pc=pc)
  else:
    print(config.dataset)
    assert (False)

  num_train_batches = len(dataloaders[0])
  print("num assign batches: %d" % len(mapping_assignment_dataloader))
  num_imgs = len(mapping_assignment_dataloader.dataset)
  print("num imgs in assign dataset: %d" % num_imgs)

  # networks and optimisers
  # ------------------------------------------------------

  net = archs.__dict__[config.arch](config)
  model_path = os.path.join(config.out_dir, "best_net.pytorch")
  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))
  net.cuda()

  if given_config.use_eval:
    print("doing eval mode")
    net.eval()

  net = torch.nn.DataParallel(net)
  acc, nmi, ari, _ = cluster_subheads_eval(config, net,
                                           mapping_assignment_dataloader=mapping_assignment_dataloader,
                                           mapping_test_dataloader=mapping_test_dataloader,
                                           sobel=True)

  config.assign_set_szs_pc_acc[str(pc)] = (num_imgs, acc)

  print("for model %d assign set sz pc %f, got %f, compared to best stored "
        "acc %f" % (config.model_ind, pc, acc, max(config.epoch_acc)))
  print(config.assign_set_szs_pc_acc)
  sysout.flush()

if not given_config.dont_save:
  print("writing to new config")
  # store to config
  with open(os.path.join(config.out_dir, "config.pickle"),
            "wb") as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)
