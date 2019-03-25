import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import scipy.io as sio
from libtiff import TIFF

# split each 6000x6000 image into 15x15 half scaled 200x200 images
# make train and test lists

SOURCE_IMGS_DIR = "/scratch/local/ssd/xuji/POTSDAM/raw/4_Ortho_RGBIR"
SOURCE_IMGS_SUFFIX = "_RGBIR.tif"
SOURCE_GT_DIR = "/scratch/local/ssd/xuji/POTSDAM/raw/5_Labels_for_participants"
SOURCE_GT_SUFFIX = "_label.tif"
NUM_SOURCE_IMGS = 38
NUM_SOURCE_GT = 24

OUT_DIR = "/scratch/local/ssd/xuji/POTSDAM/"
OUT_PER_SOURCE = 15 * 15  # 200x200 images, shrunk in half from 400x400
NUM_TRAIN = NUM_SOURCE_IMGS * OUT_PER_SOURCE
NUM_TEST = int(NUM_TRAIN * 0.1)
assert (NUM_TEST == 855)


# NUM_TEST = 20

def main():
  out_dir_imgs = os.path.join(OUT_DIR, "imgs")
  out_dir_gt = os.path.join(OUT_DIR, "gt")

  if not os.path.exists(out_dir_imgs):
    os.makedirs(out_dir_imgs)
  if not os.path.exists(out_dir_gt):
    os.makedirs(out_dir_gt)

  indices_with_gt = []
  next_index = 0

  # so it's a random order
  save_names = np.random.permutation(NUM_TRAIN)

  num_img = 0
  num_gt = 0
  for img_path in sorted(glob.glob(SOURCE_IMGS_DIR + "/*.tif")):
    print("on img: %d %s" % (num_img, datetime.now()))
    sys.stdout.flush()
    num_img += 1
    # each block's image and gt (if exists) share same name (from save_names)

    handle = os.path.basename(img_path)[:-len(SOURCE_IMGS_SUFFIX)]
    tif = TIFF.open(img_path, mode="r")
    img = tif.read_image()
    assert (img.shape == (6000, 6000, 4))  # uint8 np array, RGBIR

    split_and_save_imgs(img, next_index, names=save_names, cut=400,
                        rescale=0.5, dir=out_dir_imgs)

    gt_path = os.path.join(SOURCE_GT_DIR, handle + SOURCE_GT_SUFFIX)
    if os.path.isfile(gt_path):
      num_gt += 1
      current_indices = range(next_index, next_index + OUT_PER_SOURCE)
      indices_with_gt += current_indices

      gt_tif = TIFF.open(gt_path, mode="r")
      gt = gt_tif.read_image()
      assert (gt.shape == (6000, 6000, 3))  # uint8 np array, RGB

      split_and_save_gts(gt, next_index, names=save_names, cut=400,
                         rescale=0.5, dir=out_dir_gt)

    next_index += OUT_PER_SOURCE

  # IID:
  # unlabelled_train+labelled_train+labelled_test,
  # labelled_train+labelled_test, labelled_train+labelled_test
  # IID+:
  # unlabelled_train+labelled_train, labelled_train, labelled_test

  # make train (does not need GT) and test splits
  # choose num_test randomly out of indices_with_gt for test, rest is train
  test_inds = np.random.choice(indices_with_gt, size=NUM_TEST, replace=False)

  unlabelled_train = open(os.path.join(OUT_DIR, "unlabelled_train.txt"), "w+")
  labelled_train = open(os.path.join(OUT_DIR, "labelled_train.txt"), "w+")
  labelled_test = open(os.path.join(OUT_DIR, "labelled_test.txt"), "w+")

  for i in xrange(NUM_TRAIN):
    if i in test_inds:
      file = labelled_test
    elif i in indices_with_gt:
      file = labelled_train
    else:
      file = unlabelled_train

    file.write("%d\n" % save_names[i])

  unlabelled_train.close()
  labelled_train.close()
  labelled_test.close()

  assert (next_index == NUM_TRAIN)
  assert (num_img == NUM_SOURCE_IMGS)
  assert (num_gt == NUM_SOURCE_GT)


def split_and_save_imgs(img, next_index, names, cut, rescale, dir):
  # takes uint8 np array
  # saves as cut*rescale, cut*rescale sized png, under names[index]

  img = cv2.resize(img, dsize=None, fx=rescale, fy=rescale,
                   interpolation=cv2.INTER_LINEAR)
  cut = int(cut * rescale)

  assert (cut == 200)  # sanity

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
      name = names[next_index + offset]
      sio.savemat(os.path.join(dir, "%s.mat" % str(name)), {"img": img_curr})
      offset += 1


def split_and_save_gts(gt, next_index, names, cut, rescale, dir):
  # takes uint8 np array
  # saves as cut*rescale, cut*rescale sized mat files, under names[index]

  # turn rgb into flat indices
  colour_dict = {"[255, 255, 255]": 0,  # roads
                 "[0, 0, 255]": 1,  # buildings
                 "[0, 255, 255]": 2,  # vegetation
                 "[0, 255, 0]": 3,  # tree
                 "[255, 255, 0]": 4,  # car
                 "[255, 0, 0]": 5  # clutter
                 }

  gt = gt.astype(np.uint8)
  h, w, c = gt.shape
  flat_gt = np.zeros((h, w), dtype=np.int32)
  assert (c == 3)
  for y in xrange(h):
    for x in xrange(w):
      colour = str(list(gt[y, x]))
      gt_c = colour_dict[colour]
      flat_gt[y, x] = gt_c

  # rescale and split up
  flat_gt = cv2.resize(flat_gt, dsize=None, fx=rescale, fy=rescale,
                       interpolation=cv2.INTER_NEAREST)
  cut = int(cut * rescale)

  assert (cut == 200)  # sanity
  h, w = flat_gt.shape
  assert (h == gt.shape[0] * rescale and w == gt.shape[1] * rescale)
  num_blocks = int(h / cut)

  assert (num_blocks == 15)  # sanity

  assert (h == w)
  assert h % cut == 0

  offset = 0
  for i_h in xrange(num_blocks):
    for i_w in xrange(num_blocks):
      start_h = i_h * cut
      start_w = i_w * cut
      flat_gt_curr = flat_gt[start_h:(start_h + cut), start_w:(start_w + cut)]
      name = names[next_index + offset]
      sio.savemat(os.path.join(dir, "%s.mat" % str(name)), {"gt": flat_gt_curr})
      offset += 1


if __name__ == "__main__":
  main()
