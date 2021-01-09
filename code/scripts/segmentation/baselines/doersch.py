from __future__ import print_function

import argparse
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import code.archs as archs
from code.utils.cluster.general import config_to_str, get_opt, update_lr
from code.utils.cluster.transforms import sobel_process
from code.utils.segmentation.data import make_Coco_dataloaders, \
  make_Potsdam_dataloaders
from code.utils.segmentation.baselines.kmeans_segmentation_eval import \
  kmeans_segmentation_eval
from code.utils.segmentation.baselines.doersch_utils import \
  doersch_set_patches, doersch_loss

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

# Doersch options
parser.add_argument("--doersch_patch_side", type=int, default=11)
parser.add_argument("--max_num_kmeans_samples", type=int, default=-1)
parser.add_argument("--doersch_stats", type=str,
                    default="/users/xuji/iid/iid_private/code/datasets"
                            "/segmentation/baselines")

# for COCO
parser.add_argument("--fine_to_coarse_dict", type=str,
                    default="/users/xuji/iid/iid_private/code/datasets"
                            "/segmentation/util/out/fine_to_coarse_dict.pickle")

# COCO and Potsdam
parser.add_argument("--use_coarse_labels", default=False,
                    action="store_true")  # new

# for COCO only
parser.add_argument("--include_things_labels", default=False,
                    action="store_true")  # new
parser.add_argument("--incl_animal_things", default=False,
                    action="store_true")  # new
parser.add_argument("--coco_164k_curated_version", type=int, default=-1)

parser.add_argument("--gt_k", type=int, required=True)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=3200)
parser.add_argument("--batch_sz", type=int, required=True)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", default=False, action="store_true")
parser.add_argument("--no_pre_eval", default=False, action="store_true")

parser.add_argument("--save_multiple", default=False, action="store_true")
parser.add_argument("--verbose", default=False, action="store_true")

# data options common to both img1 and img2
parser.add_argument("--no_sobel", default=False, action="store_true")
parser.add_argument("--include_rgb", default=False, action="store_true")

parser.add_argument("--pre_scale_all", default=False,
                    action="store_true")  # new
parser.add_argument("--pre_scale_factor", type=float, default=0.5)  #

parser.add_argument("--input_sz", type=int, default=161)

# data options for img2 (i.e. transforms we learn invariance/equivariance for)
# jitter invariance
parser.add_argument("--jitter_brightness", type=float, default=0.4)
parser.add_argument("--jitter_contrast", type=float, default=0.4)
parser.add_argument("--jitter_saturation", type=float, default=0.4)
parser.add_argument("--jitter_hue", type=float, default=0.125)

# flip equivariance
parser.add_argument("--flip_p", type=float, default=0.5)

config = parser.parse_args()

assert (not (config.no_sobel and (not config.include_rgb)))
assert ("Doersch" in config.arch)
config.use_doersch_datasets = True

if "Coco" in config.dataset:
  if not config.include_rgb:
    config.in_channels = 2  # just sobel
  else:
    config.in_channels = 3  # rgb
    if not config.no_sobel:
      config.in_channels += 2  # rgb + sobel
  config.using_IR = False
elif config.dataset == "Potsdam":
  if not config.include_rgb:
    config.in_channels = 1 + 2  # ir + sobel
  else:
    config.in_channels = 4  # rgbir
    if not config.no_sobel:
      config.in_channels += 2  # rgbir + sobel

  config.using_IR = True
else:
  assert (False)

# list of dataloaders has one dataloader, which returns single pair (img,
# mask) for training
config.num_dataloaders = 1
config.single_mode = True  # used by dataset

config.use_random_scale = False
config.use_random_affine = False

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
# assert(config.batch_sz % config.num_dataloaders == 0)
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)

# copy of IID, or fully unsupervised eval, setting
# PERM, one-to-one, IID:
#   mapping can be found and tested on *same set*

if "Coco10k" in config.dataset:
  config.train_partitions = ["all"]
  config.mapping_assignment_partitions = ["all"]
  config.mapping_test_partitions = ["all"]
elif "Coco164k" in config.dataset:
  config.train_partitions = ["train2017", "val2017"]
  config.mapping_assignment_partitions = ["train2017", "val2017"]
  config.mapping_test_partitions = ["train2017", "val2017"]
elif config.dataset == "Potsdam":
  config.train_partitions = ["unlabelled_train", "labelled_train",
                             "labelled_test"]
  config.mapping_assignment_partitions = ["labelled_train", "labelled_test"]
  config.mapping_test_partitions = ["labelled_train", "labelled_test"]
else:
  assert (False)

print("Given config: %s" % config_to_str(config))

if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

if config.restart:
  given_config = config
  reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == given_config.model_ind)
  config.restart = True

  # copy over new num_epochs and lr schedule
  config.num_epochs = given_config.num_epochs
  config.lr_schedule = given_config.lr_schedule

# Data -------------------------------------------------------------------------

# datasets produce either 2 or 5 channel images based on config.include_rgb

# because fully unsupervised
assert (config.mapping_assignment_partitions == config.mapping_test_partitions)

if "Coco" in config.dataset:
  dataloaders, mapping_assignment_test_dataloader, _ = \
    make_Coco_dataloaders(config)
elif config.dataset == "Potsdam":
  dataloaders, mapping_assignment_test_dataloader, _ = \
    make_Potsdam_dataloaders(config)
else:
  raise NotImplementedError

num_train_batches = len(dataloaders[0])
print("length of train dataloader %d" % num_train_batches)
print("length of mapping assign and test dataloader %d" % len(
  mapping_assignment_test_dataloader))

assert (len(dataloaders) == 1)
dataloader = dataloaders[0]

# networks and optimisers ------------------------------------------------------

net = archs.__dict__[config.arch](config)
if config.restart:
  model_path = os.path.join(config.out_dir, "latest_net.pytorch")
  choose_best = False
  if not os.path.exists(model_path):
    print("latest doesn't exist, using best")
    model_path = os.path.join(config.out_dir, "best_net.pytorch")
    choose_best = True
  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))
net.cuda()
net = torch.nn.DataParallel(net)

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
  if not choose_best:
    optimiser.load_state_dict(
      torch.load(os.path.join(config.out_dir, "latest_optimiser.pytorch")))
  else:
    print("latest doesn't exist, using best")
    optimiser.load_state_dict(torch.load(os.path.join(config.out_dir,
                                                      "best_optimiser.pytorch")))

if config.restart:
  if not choose_best:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
  else:
    next_epoch = np.argmax(np.array(config.epoch_acc)) + 1

  print("starting from epoch %d" % next_epoch)

  # in case we overshot without saving
  config.epoch_acc = config.epoch_acc[:next_epoch]
  config.epoch_nmi = config.epoch_nmi[:next_epoch]
  config.epoch_ari = config.epoch_ari[:next_epoch]

  config.epoch_loss = config.epoch_loss[:(next_epoch - 1)]
else:
  config.epoch_acc = []
  config.epoch_nmi = []
  config.epoch_ari = []

  config.epoch_loss = []

  if (not config.no_pre_eval):
    torch.cuda.empty_cache()
    net.module.eval()
    acc, nmi, ari, masses = kmeans_segmentation_eval(config, net,
                                                     mapping_assignment_test_dataloader)
    config.epoch_acc.append(acc)
    config.epoch_nmi.append(nmi)
    config.epoch_ari.append(ari)
    config.epoch_masses = masses.reshape((1, config.gt_k))

    print("Pre: acc %f nmi %f ari %f time %s" % (acc, nmi, ari, datetime.now()))
    sys.stdout.flush()

  next_epoch = 1

fig, axarr = plt.subplots(3, sharex=False, figsize=(20, 20))

crossent = torch.nn.CrossEntropyLoss(reduction="none").cuda()

for e_i in range(next_epoch, config.num_epochs):
  torch.cuda.empty_cache()

  net.module.train()
  is_best = False

  if e_i in config.lr_schedule:
    optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

  avg_loss = 0.  # over epoch

  for b_i, tup in enumerate(dataloader):
    net.module.zero_grad()

    img, mask = tup  # cuda

    # no need for requires_grad or Variable (torch 0.4.1)
    if (not config.no_sobel):
      img = sobel_process(img, config.include_rgb, using_IR=config.using_IR)

    centre, other, position_gt = doersch_set_patches(input_sz=config.input_sz,
                                                     patch_side=config.doersch_patch_side)
    position_pred = net(img, centre=centre, other=other)

    loss = doersch_loss(position_pred, centre, other, position_gt, mask,
                        crossent=crossent,
                        verbose=config.verbose)

    if ((b_i % 100) == 0) or (e_i == next_epoch):
      print("Model ind %d epoch %d batch: %d loss %f "
            "time %s" % \
            (config.model_ind, e_i, b_i, float(loss.item()), datetime.now()))
      sys.stdout.flush()

    if not np.isfinite(loss.item()):
      print("Loss is not finite... %s:" % str(loss.item()))
      exit(1)

    avg_loss += loss.item()

    loss.backward()
    optimiser.step()

    b_i += 1

  avg_loss /= num_train_batches
  avg_loss = float(avg_loss)

  torch.cuda.empty_cache()
  net.module.eval()

  acc, nmi, ari, masses = kmeans_segmentation_eval(config, net,
                                                   mapping_assignment_test_dataloader)
  print("... metrics acc %f nmi %f ari %f time %s" %
        (acc, nmi, ari, datetime.now()))
  sys.stdout.flush()

  if (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc)):
    is_best = True

  config.epoch_acc.append(acc)
  config.epoch_nmi.append(nmi)
  config.epoch_ari.append(ari)

  config.epoch_loss.append(avg_loss)  # config stores 1

  masses = masses.reshape((1, config.gt_k))
  config.epoch_masses = np.concatenate((config.epoch_masses, masses), axis=0)

  axarr[0].clear()
  axarr[0].plot(config.epoch_loss)
  axarr[0].set_title("Loss")

  axarr[1].clear()
  axarr[1].plot(config.epoch_acc)
  axarr[1].set_title("ACC")

  axarr[2].clear()
  for c in range(config.gt_k):
    axarr[2].plot(config.epoch_masses[:, c])
  axarr[2].set_title("Masses (reordered)")

  fig.canvas.draw_idle()
  fig.savefig(os.path.join(config.out_dir, "plots.png"))

  if is_best or (e_i % 10 == 0) or config.save_multiple:
    # save cpu version
    net.module.cpu()

    if is_best:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "best_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "best_optimiser.pytorch"))

    # save model sparingly for this script
    if e_i % 10 == 0:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "latest_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "latest_optimiser.pytorch"))
      config.last_epoch = e_i  # for last saved version

    if config.save_multiple and (e_i % 3 == 0):
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "e_%d_net.pytorch" % e_i))

    net.module.cuda()

  with open(os.path.join(config.out_dir, "config.pickle"),
            "wb") as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)
