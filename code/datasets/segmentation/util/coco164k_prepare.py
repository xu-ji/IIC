import os.path as osp
import pickle
from datetime import datetime
from glob import glob
from os import makedirs
from sys import stdout as sysstd

import cv2
import numpy as np

# for coarse only
"""

_datasets = ["Coco164kFew_Stuff",
            "Coco164kFew_Stuff_People", "Coco164kFew_Stuff_People_Animals",
            "Coco164kFew_Stuff_Animals",
             "Coco164kFull_Stuff_Coarse"]
"""
_datasets = ["Coco164kFull_Stuff_Coarse"]

_dataset_root = "/scratch/local/ssd/xuji/COCO/CocoStuff164k"

_class_thresh = 0.75
_size_thresh = 360
_version = 7

_datasets_to_coarse = {"Coco164kFull_Stuff_Coarse": range(12, 27),
                       "Coco164kFew_Stuff": [23, 22, 21],
                       "Coco164kFew_Stuff_People": [23, 22, 21, 9],
                       "Coco164kFew_Stuff_People_Animals": [23, 22, 21, 9, 7],
                       "Coco164kFew_Stuff_Animals": [23, 22, 21, 7]}

with open("/users/xuji/iid/iid_private/code/datasets"
          "/segmentation/util/out/fine_to_coarse_dict.pickle", "rb") as dict_f:
  d = pickle.load(dict_f)
  _fine_to_coarse_dict = d["fine_index_to_coarse_index"]


def main():
  # curate version of train/val lists where >= thresh% of pixels belong to
  # classes of interest

  if not osp.exists(osp.join(_dataset_root, "curated")):
    makedirs(osp.join(_dataset_root, "curated"))

  if not osp.exists(osp.join(_dataset_root, "curated", "train2017")):
    makedirs(osp.join(_dataset_root, "curated", "train2017"))
    makedirs(osp.join(_dataset_root, "curated", "val2017"))

  for dataset in _datasets:
    for split in ["train2017", "val2017"]:
      output_path = osp.join(_dataset_root, "curated", split,
                             "%s_%d.txt" % (dataset, _version))
      print("doing %s" % output_path)
      sysstd.flush()

      # unvetted 164k - no orig list, just image dump, per split
      file_list_gt = sorted(
        glob(osp.join(_dataset_root, "annotations", split, "*.png")))

      fine_list = fine_from_coarse(_datasets_to_coarse[dataset])
      print("using fine inds list: %s" % fine_list)

      curated_handles = []
      for img_i, gt_path in enumerate(file_list_gt):
        handle = osp.basename(gt_path).replace(".png", "")
        if img_i % 2000 == 0:
          print("on %d of %d, handle %s, accepted so far %d, %s" %
                (img_i, len(file_list_gt), handle, len(curated_handles),
                 datetime.now()))
          sysstd.flush()

        if meets_conditions(gt_path, fine_list, _class_thresh, _size_thresh):
          curated_handles.append(handle)

      print("...num_imgs: %d" % len(curated_handles))
      sysstd.flush()

      with open(output_path, "w+") as outf:
        for handle in curated_handles:
          outf.write("%s\n" % handle)


def meets_conditions(gt_path, fine_class_ids, class_thresh, size_thresh):
  # we are already in the right directory
  gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
  h, w = gt.shape

  if not ((h >= size_thresh) and (w >= size_thresh)):
    return False

  pixel_count = 0
  pixel_thresh = h * w * class_thresh
  for c in fine_class_ids:
    pixel_count += (gt == c).sum()
    if pixel_count >= pixel_thresh:
      return True

  return False


def fine_from_coarse(coarse_list):
  fine_list = []

  # 182 class inds, [0, 181] canonical fine indexing
  # (255 means unlabelled in 164k)
  for fine in xrange(182):
    coarse = _fine_to_coarse_dict[fine]
    if coarse in coarse_list:
      fine_list.append(fine)

  return fine_list


if __name__ == "__main__":
  main()
