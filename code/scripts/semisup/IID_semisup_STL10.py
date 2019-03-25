from __future__ import print_function

import argparse
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import torch
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

import code.archs as archs
from code.archs.semisup.sup_head5 import SupHead5
from code.utils.cluster.general import update_lr
from code.utils.cluster.transforms import sobel_process, \
  sobel_make_transforms
from code.utils.semisup.general import get_dlen, assess_acc_block
from code.utils.semisup.dataset import TenCropAndFinish


# Finetune a network that has been trained in an unsupervised fashion but with a
# train/test split (e.g. network that has been trained with IIC+)

# Options ----------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ind", type=int, required=True)

  parser.add_argument("--arch", type=str, required=True)

  parser.add_argument("--head_lr", type=float, required=True)
  parser.add_argument("--trunk_lr", type=float, required=True)

  parser.add_argument("--num_epochs", type=int, default=3200)

  parser.add_argument("--new_batch_sz", type=int, default=-1)

  parser.add_argument("--old_model_ind", type=int, required=True)

  parser.add_argument("--penultimate_features", default=False,
                      action="store_true")

  parser.add_argument("--random_affine", default=False, action="store_true")
  parser.add_argument("--affine_p", type=float, default=0.5)

  parser.add_argument("--cutout", default=False, action="store_true")
  parser.add_argument("--cutout_p", type=float, default=0.5)
  parser.add_argument("--cutout_max_box", type=float, default=0.5)

  parser.add_argument("--restart", default=False, action="store_true")
  parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
  parser.add_argument("--lr_mult", type=float, default=0.5)

  parser.add_argument("--restart_new_model_ind", default=False,
                      action="store_true")
  parser.add_argument("--new_model_ind", type=int, default=0)

  parser.add_argument("--out_root", type=str,
                      default="/scratch/shared/slow/xuji/iid_private")
  config = parser.parse_args()  # new config

  # Setup ----------------------------------------------------------------------

  config.contiguous_sz = 10  # Tencrop
  config.out_dir = os.path.join(config.out_root, str(config.model_ind))

  if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)

  if config.restart:
    given_config = config
    reloaded_config_path = os.path.join(given_config.out_dir,
                                        "config.pickle")
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
      config = pickle.load(config_f)
    assert (config.model_ind == given_config.model_ind)

    config.restart = True
    config.num_epochs = given_config.num_epochs  # train for longer

    config.restart_new_model_ind = given_config.restart_new_model_ind
    config.new_model_ind = given_config.new_model_ind

    start_epoch = config.last_epoch + 1

    print("...restarting from epoch %d" % start_epoch)

    # in case we overshot without saving
    config.epoch_acc = config.epoch_acc[:start_epoch]
    config.epoch_loss = config.epoch_loss[:start_epoch]


  else:
    config.epoch_acc = []
    config.epoch_loss = []
    start_epoch = 0

  # old config only used retrospectively for setting up model at start
  reloaded_config_path = os.path.join(os.path.join(config.out_root,
                                                   str(config.old_model_ind)),
                                      "config.pickle")
  print("Loading old features config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    old_config = pickle.load(config_f)
    assert (old_config.model_ind == config.old_model_ind)

  if config.new_batch_sz == -1:
    config.new_batch_sz = old_config.batch_sz

  fig, axarr = plt.subplots(2, sharex=False, figsize=(20, 20))

  # Data -----------------------------------------------------------------------

  assert (old_config.dataset == "STL10")

  # make supervised data: train on train, test on test, unlabelled is unused
  tf1, tf2, tf3 = sobel_make_transforms(old_config,
                                        random_affine=config.random_affine,
                                        cutout=config.cutout,
                                        cutout_p=config.cutout_p,
                                        cutout_max_box=config.cutout_max_box,
                                        affine_p=config.affine_p)

  dataset_class = torchvision.datasets.STL10
  train_data = dataset_class(
    root=old_config.dataset_root,
    transform=tf2,  # also could use tf1
    split="train")

  train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=config.new_batch_sz,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=False)

  test_data = dataset_class(
    root=old_config.dataset_root,
    transform=None,
    split="test")
  test_data = TenCropAndFinish(test_data, input_sz=old_config.input_sz,
                               include_rgb=old_config.include_rgb)

  test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=config.new_batch_sz,
                                            # full batch
                                            shuffle=False,
                                            num_workers=0,
                                            drop_last=False)

  # Model ----------------------------------------------------------------------

  net_features = archs.__dict__[old_config.arch](old_config)

  if not config.restart:
    model_path = os.path.join(old_config.out_dir, "best_net.pytorch")
    net_features.load_state_dict(
      torch.load(model_path, map_location=lambda storage, loc: storage))

  dlen = get_dlen(net_features, train_loader,
                  include_rgb=old_config.include_rgb,
                  penultimate_features=config.penultimate_features)
  print("dlen: %d" % dlen)

  assert (config.arch == "SupHead5")
  net = SupHead5(net_features, dlen=dlen, gt_k=old_config.gt_k)

  if config.restart:
    print("restarting from latest net")
    model_path = os.path.join(config.out_dir, "latest_net.pytorch")
    net.load_state_dict(
      torch.load(model_path, map_location=lambda storage, loc: storage))

  net.cuda()
  net = torch.nn.DataParallel(net)

  opt_trunk = torch.optim.Adam(
    net.module.trunk.parameters(),
    lr=config.trunk_lr
  )
  opt_head = torch.optim.Adam(
    net.module.head.parameters(),
    lr=(config.head_lr)
  )

  if config.restart:
    print("restarting from latest optimiser")
    optimiser_states = torch.load(
      os.path.join(config.out_dir, "latest_optimiser.pytorch"))
    opt_trunk.load_state_dict(optimiser_states["opt_trunk"])
    opt_head.load_state_dict(optimiser_states["opt_head"])
  else:
    print("using new optimiser state")

  criterion = nn.CrossEntropyLoss().cuda()

  if not config.restart:
    net.eval()
    acc = assess_acc_block(net, test_loader, gt_k=old_config.gt_k,
                           include_rgb=old_config.include_rgb,
                           penultimate_features=config.penultimate_features,
                           contiguous_sz=config.contiguous_sz)

    print("pre: model %d old model %d, acc %f time %s" % (
      config.model_ind, config.old_model_ind, acc, datetime.now()))
    sys.stdout.flush()

    config.epoch_acc.append(acc)

  if config.restart_new_model_ind:
    assert (config.restart)
    config.model_ind = config.new_model_ind  # old_model_ind stays same
    config.out_dir = os.path.join(config.out_root, str(config.model_ind))
    print("restarting as model %d" % config.model_ind)

    if not os.path.exists(config.out_dir):
      os.makedirs(config.out_dir)

  # Train ----------------------------------------------------------------------

  for e_i in xrange(start_epoch, config.num_epochs):
    net.train()

    if e_i in config.lr_schedule:
      print("e_i %d, multiplying lr for opt trunk and head by %f" %
            (e_i, config.lr_mult))
      opt_trunk = update_lr(opt_trunk, lr_mult=config.lr_mult)
      opt_head = update_lr(opt_head, lr_mult=config.lr_mult)
      if not hasattr(config, "lr_changes"):
        config.lr_changes = []
      config.lr_changes.append((e_i, config.lr_mult))

    avg_loss = 0.
    num_batches = len(train_loader)
    for i, (imgs, targets) in enumerate(train_loader):
      imgs = sobel_process(imgs.cuda(), old_config.include_rgb)
      targets = targets.cuda()

      x_out = net(imgs, penultimate_features=config.penultimate_features)
      loss = criterion(x_out, targets)

      avg_loss += float(loss.data)

      opt_trunk.zero_grad()
      opt_head.zero_grad()

      loss.backward()

      opt_trunk.step()
      opt_head.step()

      if (i % 100 == 0) or (e_i == start_epoch):
        print("batch %d of %d, loss %f, time %s" % (i, num_batches,
                                                    float(loss.data),
                                                    datetime.now()))
        sys.stdout.flush()

    avg_loss /= num_batches

    net.eval()
    acc = assess_acc_block(net, test_loader, gt_k=old_config.gt_k,
                           include_rgb=old_config.include_rgb,
                           penultimate_features=config.penultimate_features,
                           contiguous_sz=config.contiguous_sz)

    print("model %d old model %d epoch %d acc %f time %s" % (
      config.model_ind, config.old_model_ind, e_i, acc, datetime.now()))
    sys.stdout.flush()

    is_best = False
    if acc > max(config.epoch_acc):
      is_best = True

    config.epoch_acc.append(acc)
    config.epoch_loss.append(avg_loss)

    axarr[0].clear()
    axarr[0].plot(config.epoch_acc)
    axarr[0].set_title("Acc")

    axarr[1].clear()
    axarr[1].plot(config.epoch_loss)
    axarr[1].set_title("Loss")

    fig.canvas.draw_idle()
    fig.savefig(os.path.join(config.out_dir, "plots.png"))

    if is_best or (e_i % 10 == 0):
      net.module.cpu()

      if is_best:
        torch.save(net.module.state_dict(),
                   os.path.join(config.out_dir, "best_net.pytorch"))
        torch.save({"opt_head": opt_head.state_dict(),
                    "opt_trunk": opt_trunk.state_dict()},
                   os.path.join(config.out_dir,
                                "best_optimiser.pytorch"))

      # save model sparingly for this script
      if e_i % 10 == 0:
        torch.save(net.module.state_dict(),
                   os.path.join(config.out_dir, "latest_net.pytorch"))
        torch.save({"opt_head": opt_head.state_dict(),
                    "opt_trunk": opt_trunk.state_dict()},
                   os.path.join(config.out_dir,
                                "latest_optimiser.pytorch"))

      net.module.cuda()

      config.last_epoch = e_i  # for last saved version

    with open(os.path.join(config.out_dir, "config.pickle"),
              'w') as outfile:
      pickle.dump(config, outfile)

    with open(os.path.join(config.out_dir, "config.txt"),
              "w") as text_file:
      text_file.write("%s" % config)


if __name__ == "__main__":
  main()
