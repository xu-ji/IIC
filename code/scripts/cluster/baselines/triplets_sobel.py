from __future__ import print_function

import argparse
import itertools
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
from code.utils.cluster.baselines.triplets import make_triplets_data, \
  triplets_eval, triplets_loss

"""
  Triplets.
  Makes output distribution same as that of attractor, and different to that 
  of repeller.
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument("--gt_k", type=int, required=True)
parser.add_argument("--output_k", type=int, required=True)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, required=True)  # num pairs

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", dest="restart", default=False,
                    action="store_true")
parser.add_argument("--test_code", dest="test_code", default=False,
                    action="store_true")

parser.add_argument("--stl_leave_out_unlabelled", default=False,
                    action="store_true")

parser.add_argument("--save_freq", type=int, default=10)

parser.add_argument("--kmeans_on_features", default=False,
                    action="store_true")

# transforms
# used for "positive" sample
parser.add_argument("--mix_train", dest="mix_train", default=False,
                    action="store_true")
parser.add_argument("--include_rgb", dest="include_rgb", default=False,
                    action="store_true")

parser.add_argument("--demean", dest="demean", default=False,
                    action="store_true")
parser.add_argument("--per_img_demean", dest="per_img_demean", default=False,
                    action="store_true")
parser.add_argument("--data_mean", type=float, nargs="+", default=[])
parser.add_argument("--data_std", type=float, nargs="+", default=[])

parser.add_argument("--crop_orig", dest="crop_orig", default=False,
                    action="store_true")
parser.add_argument("--rand_crop_sz", type=int, default=84)
parser.add_argument("--input_sz", type=int, default=96)

parser.add_argument("--fluid_warp", dest="fluid_warp", default=False,
                    action="store_true")
parser.add_argument("--rand_crop_szs_tf", type=int, nargs="+",
                    default=[])  # only used if fluid warp true
parser.add_argument("--rot_val", type=float,
                    default=0.)  # only used if fluid warp true

parser.add_argument("--cutout", default=False, action="store_true")
parser.add_argument("--cutout_p", type=float, default=0.5)
parser.add_argument("--cutout_max_box", type=float, default=0.5)

config = parser.parse_args()

# Fixed settings and checks ----------------------------------------------------

if not config.include_rgb:
  config.in_channels = 2
else:
  config.in_channels = 5

if config.output_k != config.gt_k:
  assert (config.output_k > config.gt_k)
  assert (config.kmeans_on_features)

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = config.batch_sz
config.num_dataloaders = 1

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

  if not hasattr(config, "kmeans_on_features"):
    config.kmeans_on_features = False

else:
  print("Config: %s" % config_to_str(config))

# Data, nets, optimisers -------------------------------------------------------

dataloader_original, dataloader_positive, dataloader_negative, \
dataloader_test = make_triplets_data(config)

train_dataloaders = [dataloader_original, dataloader_positive,
                     dataloader_negative]

net = archs.__dict__[config.arch](config)
if config.restart:
  model_path = os.path.join(config.out_dir, "latest_net.pytorch")
  taking_best = not os.path.exists(model_path)
  if taking_best:
    print("using best instead of latest")
    model_path = os.path.join(config.out_dir, "best_net.pytorch")

  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))
net.cuda()
net = torch.nn.DataParallel(net)
net.train()

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
  opt_path = os.path.join(config.out_dir, "latest_optimiser.pytorch")
  if taking_best:
    opt_path = os.path.join(config.out_dir, "best_optimiser.pytorch")
    optimiser.load_state_dict(torch.load(opt_path))

# Results storage --------------------------------------------------------------

if config.restart:
  if not taking_best:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
  else:
    next_epoch = np.argmax(np.array(config.epoch_acc)) + 1
  print("starting from epoch %d" % next_epoch)

  config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
  config.epoch_loss = config.epoch_loss[:next_epoch]
  config.masses = config.masses[:next_epoch, :]
  config.per_class_acc = config.per_class_acc[:next_epoch, :]
else:
  config.epoch_acc = []
  config.epoch_loss = []

  config.masses = None
  config.per_class_acc = None

  _ = triplets_eval(config, net,
                    dataloader_test=dataloader_test,
                    sobel=True)

  print("Pre: time %s: \n %s" % (datetime.now(), config.epoch_acc[-1]))
  sys.stdout.flush()
  next_epoch = 1

fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

# Train ------------------------------------------------------------------------

for e_i in xrange(next_epoch, config.num_epochs):
  print("Starting e_i: %d" % (e_i))

  if e_i in config.lr_schedule:
    optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

  avg_loss = 0.  # over heads and head_epochs (and sub_heads)
  avg_loss_count = 0

  sys.stdout.flush()

  iterators = (d for d in train_dataloaders)

  b_i = 0
  for tup in itertools.izip(*iterators):
    net.module.zero_grad()

    # no sobel yet
    imgs_orig = sobel_process(tup[0][0].cuda(), config.include_rgb)
    imgs_pos = sobel_process(tup[1][0].cuda(), config.include_rgb)
    imgs_neg = sobel_process(tup[2][0].cuda(), config.include_rgb)

    outs_orig = net(imgs_orig)
    outs_pos = net(imgs_pos)
    outs_neg = net(imgs_neg)

    curr_loss = triplets_loss(outs_orig, outs_pos, outs_neg)

    if ((b_i % 100) == 0) or (e_i == next_epoch and b_i < 10):
      print("Model ind %d epoch %d batch %d "
            "loss %f time %s" % \
            (config.model_ind, e_i, b_i, curr_loss.item(), datetime.now()))
      sys.stdout.flush()

    if not np.isfinite(float(curr_loss.item())):
      print("Loss is not finite... %s:" % str(curr_loss.item()))
      exit(1)

    avg_loss += curr_loss.item()
    avg_loss_count += 1

    curr_loss.backward()
    optimiser.step()

    b_i += 1
    if b_i == 2 and config.test_code:
      break

  avg_loss = float(avg_loss / avg_loss_count)

  config.epoch_loss.append(avg_loss)

  # Eval and storage -----------------------------------------------------------

  # when epoch over both heads is finished
  is_best = triplets_eval(config, net,
                          dataloader_test=dataloader_test,
                          sobel=True)

  print("Time %s, acc %s" % (datetime.now(), config.epoch_acc[-1]))
  sys.stdout.flush()

  axarr[0].clear()
  axarr[0].plot(config.epoch_acc)
  axarr[0].set_title("acc, top: %f" % max(config.epoch_acc))

  axarr[1].clear()
  axarr[1].plot(config.epoch_loss)
  axarr[1].set_title("Loss")

  axarr[2].clear()
  for c in xrange(config.gt_k):
    axarr[2].plot(config.masses[:, c])
  axarr[2].set_title("masses")

  axarr[3].clear()
  for c in xrange(config.gt_k):
    axarr[3].plot(config.per_class_acc[:, c])
  axarr[3].set_title("per_class_acc")

  fig.tight_layout()
  fig.canvas.draw_idle()
  fig.savefig(os.path.join(config.out_dir, "plots.png"))

  if is_best or (e_i % config.save_freq == 0):
    net.module.cpu()

    if is_best:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "best_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "best_optimiser.pytorch"))

    if e_i % config.save_freq == 0:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "latest_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "latest_optimiser.pytorch"))
      config.last_epoch = e_i  # for last saved version

    net.module.cuda()

  with open(os.path.join(config.out_dir, "config.pickle"),
            'wb') as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)

  if config.test_code:
    exit(0)
