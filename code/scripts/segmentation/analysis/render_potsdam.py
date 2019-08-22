import glob
import os
import pickle
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from libtiff import TIFF
import argparse

import code.archs as archs

# Script to render Potsdam model

# export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.segmentation.analysis.render_potsdam --model_ind 738 --net_name best.pytorch > out/gnodec4_render_738.out &


# Original image rendered as *_img.png
# Our predictions rendered as *_preds.png
# Ground truth rendered as *_gt.png

# Settings ----
args = argparse.ArgumentParser()
args.add_argument("--test_code", default=False, action="store_true")
args.add_argument("--model_ind", type=int, required=True)
args.add_argument("--net_name", type=str, default="best")

args.add_argument("--best_match", type=int, nargs="+", default=[],
                  help="Not compulsory if match has been stored in config")
args = args.parse_args()

#best_matches = {545: [(0, 0), (1, 1), (2, 2)]} # can find with clone_and_eval.py
#best_match = best_matches[model_ind]

SOURCE_IMGS_DIR = "/scratch/local/ssd/xuji/POTSDAM/raw/4_Ortho_RGBIR"
SOURCE_IMGS_SUFFIX = "_RGBIR.tif"
SOURCE_GT_DIR = "/scratch/local/ssd/xuji/POTSDAM/raw/5_Labels_for_participants"
SOURCE_GT_SUFFIX = "_label.tif"
NUM_SOURCE_IMGS = 38
NUM_SOURCE_GT = 24

if args.test_code:
  NUM_SOURCE_IMGS = 2
  NUM_SOURCE_GT = 2

IN_OUT_ROOT = "/scratch/shared/slow/xuji/iid_private/"
SUB_DIR = "full"
OUT_PER_SOURCE = 15 * 15  # 200x200 or 100x100 images, shrunk from 400x400
NUM_TRAIN = NUM_SOURCE_IMGS * OUT_PER_SOURCE

_fine_to_coarse_dict = {0: 0, 4: 0,  # roads and cars
                        1: 1, 5: 1,  # buildings and clutter
                        2: 2, 3: 2  # vegetation and trees
                        }


def main():
  # Load the model config ----
  out_dir = os.path.join(IN_OUT_ROOT, str(args.model_ind))
  reloaded_config_path = os.path.join(out_dir, "config.pickle")
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == args.model_ind)

  out_sub_dir = os.path.join(out_dir, SUB_DIR)
  if not os.path.exists(out_sub_dir):
    os.makedirs(out_sub_dir)

  print("Model output size: %d" % config.input_sz)

  if config.input_sz == 200:
    rerescale = 1.
  else:
    assert (config.input_sz == 100)
    rerescale = 0.5

  next_index = 0  # blocks
  num_img = 0  # input imgs
  num_gt = 0

  # make colour dict for predictions (gt_k - we render reordered)
  if (config.gt_k == 3):
    colour_map = [np.array([175, 28, 12], dtype=np.uint8),
                  np.array([111, 138, 155], dtype=np.uint8),
                  np.array([81, 188, 0], dtype=np.uint8),
                  ]
  else:
    colour_map = [(np.random.rand(3) * 255.).astype(np.uint8)
                  for _ in xrange(config.gt_k)]

  # so it's a random order in forward pass
  save_names = np.random.permutation(NUM_TRAIN)  # order in input_blocks
  save_names_to_orig_pos = {}  # int to (img int, row int, col int)
  input_blocks = np.zeros((NUM_TRAIN, 4, config.input_sz, config.input_sz),
                          dtype=np.uint8)

  for img_path in sorted(glob.glob(SOURCE_IMGS_DIR + "/*.tif")):
    print("on img: %d %s" % (num_img, datetime.now()))
    sys.stdout.flush()
    # each block's image and gt (if exists) share same filenames

    handle = os.path.basename(img_path)[:-len(SOURCE_IMGS_SUFFIX)]
    tif = TIFF.open(img_path, mode="r")
    img = tif.read_image()
    assert (img.shape == (6000, 6000, 4))  # uint8 np array, RGBIR

    # Print input image rgb
    shrunk_img = cv2.resize(img, dsize=None, fx=0.5 * rerescale,
                            fy=0.5 * rerescale,
                            interpolation=cv2.INTER_LINEAR)
    Image.fromarray(shrunk_img[:, :, :3]).save(os.path.join(out_sub_dir,
                                                            "%d_img.png" %
                                                            num_img))

    # Store blocks in randomly shuffled array
    split_imgs(num_img, img, next_index, names=save_names, cut=400,
               rescale=0.5, rerescale=rerescale, storage=input_blocks, \
               save_names_to_orig_pos=save_names_to_orig_pos)

    # Get gt image
    gt_path = os.path.join(SOURCE_GT_DIR, handle + SOURCE_GT_SUFFIX)
    if os.path.isfile(gt_path):
      num_gt += 1

      gt_tif = TIFF.open(gt_path, mode="r")
      gt = gt_tif.read_image()
      assert (gt.shape == (6000, 6000, 3))  # uint8 np array, RGB

      filter_gt_and_store(config, num_img, gt, rescale=0.5, rerescale=rerescale,
                          colour_map=colour_map, out_dir=out_sub_dir)

    next_index += OUT_PER_SOURCE
    num_img += 1

    if args.test_code and num_img == NUM_SOURCE_IMGS:
      break

  assert (next_index == NUM_TRAIN)
  assert (num_img == NUM_SOURCE_IMGS)
  assert (num_gt == NUM_SOURCE_GT)

  predict_and_reassemble(config, input_blocks, NUM_SOURCE_IMGS,
                         save_names_to_orig_pos,
                         colour_map, out_sub_dir)


def split_imgs(img_ind, img, next_index, names, cut,
               rescale, rerescale, storage, save_names_to_orig_pos):
  img = cv2.resize(img, dsize=None, fx=rescale, fy=rescale,
                   interpolation=cv2.INTER_LINEAR)

  img = cv2.resize(img, dsize=None, fx=rerescale, fy=rerescale,
                   interpolation=cv2.INTER_LINEAR)  # 1500, 1500 or 3000, 3000

  cut = int(cut * rescale * rerescale)

  assert (cut == 100 or cut == 200)  # sanity

  h, w, c = img.shape
  num_blocks = int(h / cut)
  assert (num_blocks == 15)  # sanity

  assert (h == w)
  assert h % cut == 0

  offset = 0
  for i_h in xrange(num_blocks):
    for i_w in xrange(num_blocks):
      start_h = i_h * cut
      start_w = i_w * cut
      img_curr = img[start_h:(start_h + cut), start_w:(start_w + cut), :]
      img_curr = img_curr.transpose((2, 0, 1))  # channels first
      name = names[next_index + offset]

      storage[name, :, :, :] = img_curr

      save_names_to_orig_pos[name] = (img_ind, i_h, i_w)

      offset += 1


def filter_gt_and_store(config, img_ind, gt, rescale, rerescale, colour_map,
                        out_dir):
  # turn rgb into flat indices, then flat into coarse if necessary
  colour_dict = {"[255, 255, 255]": 0,  # roads
                 "[0, 0, 255]": 1,  # buildings
                 "[0, 255, 255]": 2,  # vegetation
                 "[0, 255, 0]": 3,  # tree
                 "[255, 255, 0]": 4,  # car
                 "[255, 0, 0]": 5  # clutter
                 }

  gt = gt.astype(np.uint8)
  h, w, c = gt.shape
  assert (c == 3)

  recoloured = np.zeros((h, w, 3), dtype=np.uint8)
  for y in xrange(h):
    for x in xrange(w):
      colour = str(list(gt[y, x]))
      gt_c = colour_dict[colour]

      if config.use_coarse_labels:
        gt_c = _fine_to_coarse_dict[gt_c]

      recoloured[y, x, :] = colour_map[gt_c]

  # rescale
  recoloured = cv2.resize(recoloured, dsize=None, fx=rescale, fy=rescale,
                          interpolation=cv2.INTER_NEAREST)
  recoloured = cv2.resize(recoloured, dsize=None, fx=rerescale, fy=rerescale,
                          interpolation=cv2.INTER_NEAREST)

  Image.fromarray(recoloured).save(os.path.join(out_dir, "%d_gt.png" % img_ind))


def predict_and_reassemble(config, input_blocks, num_big_imgs,
                           save_names_to_orig_pos, colour_map, out_dir):
  # Run randomly shuffled blocks through network
  # Reassemble predictions into full image

  # Load model
  net = archs.__dict__[config.arch](config)
  model_path = os.path.join(config.out_dir, args.net_name)
  stored = torch.load(model_path, map_location=lambda storage, loc: storage)
  if "net" in stored.keys():
    net.load_state_dict(stored["net"])
  else:
    net.load_state_dict(stored)

  net.cuda()
  net = torch.nn.DataParallel(net)
  net.module.eval()  # <- put in eval state

  print("loaded model")
  sys.stdout.flush()

  assert (config.no_sobel)
  assert (config.in_channels == 4)
  assert (config.num_sub_heads == 1)
  num_imgs, in_channels, h, w = input_blocks.shape
  assert (in_channels == 4)
  assert (h == w)
  assert (h == config.input_sz)

  # Make predictions
  num_batches, rem = divmod(num_imgs, config.batch_sz)
  if rem != 0:
    num_batches += 1

  flat_preds = np.zeros((num_imgs, h, w), dtype=np.int32)
  for b_i in xrange(num_batches):
    start = b_i * config.batch_sz
    end_excl = min(num_imgs, start + config.batch_sz)
    imgs = torch.from_numpy(input_blocks[start:end_excl, :, :, :].astype(
      np.float32) / 255.)

    with torch.no_grad():
      x_outs_all = net(imgs)

    x_outs = x_outs_all[0]  # 1 head: bn, output_k, h, w
    flat_preds[start:end_excl, :, :] = torch.argmax(x_outs, dim=1)

  print("run through model")
  sys.stdout.flush()

  # Apply match
  if not (args.best_match == []):
    best_match = []
    for pred_i in xrange(config.output_k):
      best_match.append(pred_i, args.best_match[pred_i])
  else:
    best_epoch = np.argmax(np.array(config.epoch_acc))
    stats = config.epoch_stats[best_epoch]
    best_match = stats["best_train_sub_head_match"]
    assert(stats["best_train_sub_head"] == 0) # one sub head
    assert(stats["test_accs"][0] == config.epoch_acc[best_epoch])

  assert(len(best_match) == config.output_k)

  reordered_preds = np.zeros((num_imgs, h, w), dtype=np.int32)
  for pred_i, target_i in best_match:
    reordered_preds[flat_preds == pred_i] = target_i

  # Colour predictions
  coloured_preds = np.zeros((num_imgs, h, w, 3), dtype=np.uint8)
  for gt_c, colour in enumerate(colour_map):
    coloured_preds[(reordered_preds == gt_c), :] = colour

  # Put blocks in order
  block_side = np.sqrt(OUT_PER_SOURCE)
  trimmed_h_w = h - 2  # remove 2 cols/rows from each block
  output_h_w = int(block_side * trimmed_h_w)
  assert (output_h_w == (1500 - block_side * 2) or output_h_w == (
    3000 - block_side * 2))

  output_imgs = np.zeros((num_big_imgs, output_h_w, output_h_w, 3),
                         dtype=np.uint8)

  for name in xrange(num_imgs):
    output_img_i, h_i, w_i = save_names_to_orig_pos[name]
    output_h_i, output_w_i = h_i * trimmed_h_w, w_i * trimmed_h_w
    output_imgs[output_img_i, output_h_i:(output_h_i + trimmed_h_w),
    output_w_i:(output_w_i + trimmed_h_w), :] = \
      coloured_preds[name, 1:(h - 1), 1:(w - 1), :]

  # Save
  for output_img_i in xrange(num_big_imgs):
    Image.fromarray(output_imgs[output_img_i, :, :, :]).save(
      os.path.join(out_dir, "%d_preds.png" % output_img_i))


if __name__ == "__main__":
  main()
