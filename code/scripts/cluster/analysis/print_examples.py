import argparse
import itertools
import os
import pickle

import numpy as np
import torch
from PIL import Image

import code.archs as archs
from code.utils.cluster.data import cluster_twohead_create_dataloaders
from code.utils.cluster.transforms import sobel_process

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--num_imgs", type=int, default=200)
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

net = archs.__dict__[config.arch](config)
model_path = os.path.join(config.out_dir, "best_net.pytorch")
net.load_state_dict(
  torch.load(model_path, map_location=lambda storage, loc: storage))

net.cuda()
net.eval()

net = torch.nn.DataParallel(net)

# model dataloader
_, _, _, dataloader = cluster_twohead_create_dataloaders(config)

# render dataloader
old_value = config.include_rgb
config.include_rgb = True
_, _, _, render_dataloader = cluster_twohead_create_dataloaders(config)
config.include_rgb = old_value

if "MNIST" in config.dataset:
  sobel = False
else:
  sobel = True

using_IR = False  # not segmentation

# from first batch
img_inds = np.random.choice(config.batch_sz, size=given_config.num_imgs,
                            replace=False)

# already know the best head (and one-to-one mapping, but not used)
best_i = np.argmax(np.array(config.epoch_acc))
stats_dict = config.epoch_stats[best_i]

print(stats_dict)
if "best_train_sub_head" in stats_dict:
  best_head = stats_dict["best_train_sub_head"]
  print("best_train_sub_head: %d" % best_head)
  best_match = stats_dict["best_train_sub_head_match"]  # pred -> target

if "best_head" in stats_dict:
  best_head = stats_dict["best_head"]
  print("best_head: %d" % best_head)
  best_match = stats_dict["best_head_match"]

assert (not ("best_train_sub_head" in stats_dict and "best_head" in stats_dict))

best_match_dict = {}
for pred_i, target_i in best_match:
  best_match_dict[pred_i] = target_i

render_out_dir = os.path.join(config.out_dir, "print_examples")
if not os.path.exists(render_out_dir):
  os.makedirs(render_out_dir)

results_f = os.path.join(render_out_dir, "results.txt")

iterators = (d for d in [dataloader, render_dataloader])

for tup in itertools.izip(*iterators):
  train_batch = tup[0]
  render_batch = tup[1]

  imgs = train_batch[0].cuda()
  orig_imgs = render_batch[0]

  if sobel:
    imgs = sobel_process(imgs, config.include_rgb, using_IR=using_IR)

  flat_targets = train_batch[1]

  with torch.no_grad():
    x_outs = net(imgs)

  assert (x_outs[0].shape[1] == config.output_k)
  assert (len(x_outs[0].shape) == 2)

  x_outs_curr = x_outs[best_head]
  flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k

  with open(results_f, "w") as f:
    for i, img_i in enumerate(img_inds):
      img = orig_imgs[img_i].numpy()
      img = img[:3]
      img = img.transpose((1, 2, 0))  # channels last
      img *= 255.

      print(img.shape)
      print(img.max())
      print(img.min())

      img = Image.fromarray(img.astype(np.uint8))
      img.save(os.path.join(render_out_dir, "%d.png" % i))

      f.write("(%d) %d %d %d\n" % (i,
                                   best_match_dict[
                                     flat_preds_curr[img_i].item()],
                                   flat_targets[img_i].item(),
                                   flat_preds_curr[img_i].item()))

  break

print("finished rendering to: %s" % render_out_dir)
