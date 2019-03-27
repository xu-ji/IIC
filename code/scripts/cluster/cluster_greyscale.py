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
from code.utils.cluster.general import config_to_str, get_opt, \
  update_lr, nice
from code.utils.cluster.cluster_eval import cluster_eval
from code.utils.cluster.data import cluster_create_dataloaders
from code.utils.cluster.IID_losses import IID_loss

"""
  Semisupervised overclustering ("IIC+" = "IID+")
  Note network is trained entirely unsupervised, as labels are found for 
  evaluation only and do not affect the network.
  Train and test script (greyscale datasets).
  Network has one output head only.
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, default="ClusterNet4h")
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID+")

parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--dataset_root", type=str,
                    default="/scratch/local/ssd/xuji/MNIST")

parser.add_argument("--gt_k", type=int, default=10)
parser.add_argument("--output_k", type=int, default=10)
parser.add_argument("--lamb", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, default=240)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", dest="restart", default=False,
                    action="store_true")
parser.add_argument("--restart_from_best", dest="restart_from_best",
                    default=False, action="store_true")
parser.add_argument("--test_code", dest="test_code", default=False,
                    action="store_true")

parser.add_argument("--save_freq", type=int, default=10)

parser.add_argument("--batchnorm_track", default=False, action="store_true")

# transforms
parser.add_argument("--mix_train", dest="mix_train", default=False,
                    action="store_true")
parser.add_argument("--demean", dest="demean", default=False,
                    action="store_true")
parser.add_argument("--per_img_demean", dest="per_img_demean", default=False,
                    action="store_true")
parser.add_argument("--data_mean", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])
parser.add_argument("--data_std", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])

parser.add_argument("--crop_orig", dest="crop_orig", default=False,
                    action="store_true")
parser.add_argument("--crop_other", dest="crop_other", default=False,
                    action="store_true")
parser.add_argument("--tf1_crop", type=str, default="random")  # type name
parser.add_argument("--tf2_crop", type=str, default="random")
parser.add_argument("--tf1_crop_sz", type=int, default=84)
parser.add_argument("--tf2_crop_szs", type=int, nargs="+",
                    default=[84])  # allow diff crop for imgs_tf
parser.add_argument("--tf3_crop_diff", dest="tf3_crop_diff", default=False,
                    action="store_true")
parser.add_argument("--tf3_crop_sz", type=int, default=0)
parser.add_argument("--input_sz", type=int, default=96)

parser.add_argument("--rot_val", type=float, default=0.)
parser.add_argument("--always_rot", dest="always_rot", default=False,
                    action="store_true")
parser.add_argument("--no_jitter", dest="no_jitter", default=False,
                    action="store_true")
parser.add_argument("--no_flip", dest="no_flip", default=False,
                    action="store_true")

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

config.twohead = False

config.in_channels = 1
config.out_dir = os.path.join(config.out_root, str(config.model_ind))
assert (config.batch_sz % config.num_dataloaders == 0)
config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders

assert (config.mode == "IID+")
assert (config.output_k >= config.gt_k)
config.eval_mode = "orig"
config.double_eval = False

if "MNIST" == config.dataset:
  config.train_partitions = [True]
  config.mapping_assignment_partitions = [True]
  config.mapping_test_partitions = [False]

if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

if config.restart:
  config_name = "config.pickle"
  net_name = "latest_net.pytorch"
  opt_name = "latest_optimiser.pytorch"

  if config.restart_from_best:
    config_name = "best_config.pickle"
    net_name = "best_net.pytorch"
    opt_name = "best_optimiser.pytorch"

  given_config = config
  reloaded_config_path = os.path.join(given_config.out_dir, config_name)
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == given_config.model_ind)
  config.restart = True
  config.restart_from_best = given_config.restart_from_best

  config.num_epochs = given_config.num_epochs
  config.lr_schedule = given_config.lr_schedule

else:
  print("Config: %s" % config_to_str(config))

# Model ------------------------------------------------------------------------

dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
  cluster_create_dataloaders(config)

net = archs.__dict__[config.arch](config)
if config.restart:
  model_path = os.path.join(config.out_dir, net_name)
  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))

net.cuda()
net = torch.nn.DataParallel(net)
net.train()

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
  print("loading latest opt")
  optimiser.load_state_dict(
    torch.load(os.path.join(config.out_dir, opt_name)))

if config.restart:
  if not config.restart_from_best:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
  else:
    # sanity check
    next_epoch = np.argmax(np.array(config.epoch_acc)) + 1
    assert (next_epoch == config.last_epoch + 1)
  print("starting from epoch %d" % next_epoch)

  config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
  config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
  config.epoch_stats = config.epoch_stats[:next_epoch]

  config.epoch_loss = config.epoch_loss[:(next_epoch - 1)]
  config.epoch_loss_no_lamb = config.epoch_loss_no_lamb[:(next_epoch - 1)]
else:
  config.epoch_acc = []
  config.epoch_avg_subhead_acc = []
  config.epoch_stats = []

  config.epoch_loss = []
  config.epoch_loss_no_lamb = []

  _ = cluster_eval(config, net,
                   mapping_assignment_dataloader=mapping_assignment_dataloader,
                   mapping_test_dataloader=mapping_test_dataloader,
                   sobel=False)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  sys.stdout.flush()
  next_epoch = 1

fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

# Train ------------------------------------------------------------------------

for e_i in xrange(next_epoch, config.num_epochs):
  print("Starting e_i: %d" % e_i)
  sys.stdout.flush()

  iterators = (d for d in dataloaders)

  b_i = 0

  if e_i in config.lr_schedule:
    optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

  avg_loss = 0.  # over epoch
  avg_loss_no_lamb = 0.
  avg_loss_count = 0

  for tup in itertools.izip(*iterators):
    net.module.zero_grad()

    all_imgs = torch.zeros(config.batch_sz, config.in_channels,
                           config.input_sz,
                           config.input_sz).cuda()
    all_imgs_tf = torch.zeros(config.batch_sz, config.in_channels,
                              config.input_sz,
                              config.input_sz).cuda()

    imgs_curr = tup[0][0]  # always the first
    curr_batch_sz = imgs_curr.size(0)
    for d_i in xrange(config.num_dataloaders):
      imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
      assert (curr_batch_sz == imgs_tf_curr.size(0))

      actual_batch_start = d_i * curr_batch_sz
      actual_batch_end = actual_batch_start + curr_batch_sz
      all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
        imgs_curr.cuda()
      all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
        imgs_tf_curr.cuda()

    if not (curr_batch_sz == config.dataloader_batch_sz):
      print("last batch sz %d" % curr_batch_sz)

    curr_total_batch_sz = curr_batch_sz * config.num_dataloaders  # times 2
    all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
    all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

    x_outs = net(all_imgs)
    x_tf_outs = net(all_imgs_tf)

    avg_loss_batch = None  # avg over the heads
    avg_loss_no_lamb_batch = None
    for i in xrange(config.num_sub_heads):
      loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i], lamb=config.lamb)
      if avg_loss_batch is None:
        avg_loss_batch = loss
        avg_loss_no_lamb_batch = loss_no_lamb
      else:
        avg_loss_batch += loss
        avg_loss_no_lamb_batch += loss_no_lamb

    avg_loss_batch /= config.num_sub_heads
    avg_loss_no_lamb_batch /= config.num_sub_heads

    if ((b_i % 100) == 0) or (e_i == next_epoch):
      print("Model ind %d epoch %d batch: %d avg loss %f avg loss no lamb %f "
            "time %s" % \
            (config.model_ind, e_i, b_i, avg_loss_batch.item(),
             avg_loss_no_lamb_batch.item(), datetime.now()))
      sys.stdout.flush()

    if not np.isfinite(avg_loss_batch.item()):
      print("Loss is not finite... %s:" % str(avg_loss_batch))
      exit(1)

    avg_loss += avg_loss_batch.item()
    avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
    avg_loss_count += 1

    avg_loss_batch.backward()

    optimiser.step()

    b_i += 1
    if b_i == 2 and config.test_code:
      break

  # Eval -----------------------------------------------------------------------

  avg_loss = float(avg_loss / avg_loss_count)
  avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

  config.epoch_loss.append(avg_loss)
  config.epoch_loss_no_lamb.append(avg_loss_no_lamb)

  is_best = cluster_eval(config, net,
                         mapping_assignment_dataloader=mapping_assignment_dataloader,
                         mapping_test_dataloader=mapping_test_dataloader,
                         sobel=False)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  sys.stdout.flush()

  axarr[0].clear()
  axarr[0].plot(config.epoch_acc)
  axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

  axarr[1].clear()
  axarr[1].plot(config.epoch_avg_subhead_acc)
  axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

  axarr[2].clear()
  axarr[2].plot(config.epoch_loss)
  axarr[2].set_title("Loss")

  axarr[3].clear()
  axarr[3].plot(config.epoch_loss_no_lamb)
  axarr[3].set_title("Loss no lamb")

  fig.canvas.draw_idle()
  fig.savefig(os.path.join(config.out_dir, "plots.png"))

  if is_best or (e_i % config.save_freq == 0):
    net.module.cpu()

    if e_i % config.save_freq == 0:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "latest_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "latest_optimiser.pytorch"))
      config.last_epoch = e_i  # for last saved version

    if is_best:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "best_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "best_optimiser.pytorch"))

      with open(os.path.join(config.out_dir, "best_config.pickle"),
                'wb') as outfile:
        pickle.dump(config, outfile)

      with open(os.path.join(config.out_dir, "best_config.txt"),
                "w") as text_file:
        text_file.write("%s" % config)

    net.module.cuda()

  with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
    text_file.write("%s" % config)

  if config.test_code:
    exit(0)
