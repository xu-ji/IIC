from __future__ import print_function

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import vlfeat  # calls constructor
from sklearn.cluster import MiniBatchKMeans

from code.utils.cluster.eval_metrics import _hungarian_match, _original_match, \
  _acc
from code.utils.segmentation.data import make_Coco_dataloaders, \
  make_Potsdam_dataloaders

SIFT_DLEN = 128
SIFT_STEP = 10


def _get_vectorised_sift_samples(archetype_config, dataloader):
  # returns num unmasked pixels x SIFT_DLEN, in uint8 format
  # operates on greyscale 128 bit images

  num_batches, batch_sz = len(dataloader), archetype_config.dataloader_batch_sz
  num_imgs_max = num_batches * batch_sz  # estimate
  img_sz = archetype_config.input_sz

  # cluster individual (box central) pixels
  desc_side = int(img_sz / SIFT_STEP)
  print("img sz %d, desc_side %d" % (img_sz, desc_side))
  sys.stdout.flush()

  descs_all = np.zeros((num_imgs_max, desc_side * desc_side,
                        SIFT_DLEN), dtype=np.uint8)
  masks_all = np.zeros((num_imgs_max, desc_side * desc_side), dtype=np.bool)
  labels_all = None
  actual_num_imgs = 0

  # when descriptor matrix flattened, goes along rows first (rows change slow)
  central_inds_h = (np.arange(desc_side) * SIFT_STEP +
                    (SIFT_STEP / 2)).reshape((desc_side, 1)).repeat(desc_side,
                                                                    axis=1)
  central_inds_w = (np.arange(desc_side) * SIFT_STEP +
                    (SIFT_STEP / 2)).reshape((1, desc_side)).repeat(desc_side,
                                                                    axis=0)
  central_inds_h, central_inds_w = central_inds_h.reshape(-1), \
                                   central_inds_w.reshape(-1)

  for b_i, batch in enumerate(dataloader):
    if len(batch) == 3:  # test dataloader
      store_labels = True

      if (labels_all is None):
        labels_all = np.zeros((num_imgs_max, desc_side * desc_side),
                              dtype=np.int32)
      imgs, labels, masks = batch
      labels = labels.cpu().numpy().astype(np.int32)
    else:  # training dataloader
      store_labels = False
      imgs, _, _, masks = batch

    # imgs currently channel first, [0-1] range, floats
    imgs = (imgs * 255.).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    masks = masks.cpu().numpy().astype(np.bool)

    curr_batch_sz, h, w, c = imgs.shape
    assert (h == archetype_config.input_sz and w == archetype_config.input_sz
            and c == archetype_config.in_channels)
    if b_i < num_batches - 1:
      assert (batch_sz == curr_batch_sz)

    start = b_i * batch_sz
    for i in range(curr_batch_sz):
      grey_img = cv2.cvtColor(imgs[i, :, :, :], cv2.COLOR_RGB2GRAY)
      locs, descs = vlfeat.vl_dsift(grey_img, step=SIFT_STEP)
      descs = descs.transpose((1, 0))  # 40*40, 128
      descs = descs.reshape(-1, SIFT_DLEN)  # rows change slowest

      # get the corresponding box central mask/label
      mask = masks[i][central_inds_h, central_inds_w]

      offset = start + i
      descs_all[offset, :, :] = descs
      masks_all[offset, :] = mask
      if store_labels:
        label = labels[i][central_inds_h, central_inds_w]
        labels_all[offset, :] = label

    actual_num_imgs += curr_batch_sz

  descs_all = descs_all[:actual_num_imgs, :, :]
  masks_all = masks_all[:actual_num_imgs, :]
  num_unmasked = masks_all.sum()
  if store_labels:
    labels_all = labels_all[:actual_num_imgs, :]
    samples_labels = labels_all[masks_all].reshape(-1)
    assert (samples_labels.shape[0] == num_unmasked)

  samples = descs_all[masks_all, :].reshape(-1, SIFT_DLEN)
  assert (samples.shape[0] == num_unmasked)

  if not store_labels:
    return samples
  else:
    return samples, samples_labels


def _get_vectorised_colour_samples(archetype_config, dataloader):
  num_batches, batch_sz = len(dataloader), archetype_config.dataloader_batch_sz
  num_imgs_max = num_batches * batch_sz  # estimate
  img_sz = archetype_config.input_sz

  # cluster individual pixels
  imgs_all = np.zeros(
    (num_imgs_max, img_sz, img_sz, archetype_config.in_channels),
    dtype=np.uint8)
  masks_all = np.zeros((num_imgs_max, img_sz, img_sz), dtype=np.bool)
  labels_all = None
  actual_num_imgs = 0
  for b_i, batch in enumerate(dataloader):
    if len(batch) == 3:
      store_labels = True

      if (labels_all is None):
        labels_all = np.zeros((num_imgs_max, img_sz, img_sz), dtype=np.int32)
      imgs, labels, masks = batch
      labels = labels.cpu().numpy().astype(np.int32)
    else:
      store_labels = False
      imgs, _, _, masks = batch

    # channels last
    imgs = (imgs * 255.).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    masks = masks.cpu().numpy().astype(np.bool)

    curr_batch_sz, h, w, c = imgs.shape
    assert (h == archetype_config.input_sz and w == archetype_config.input_sz
            and c == archetype_config.in_channels)
    if b_i < num_batches - 1:
      assert (batch_sz == curr_batch_sz)

    start = b_i * batch_sz
    imgs_all[start:(start + curr_batch_sz), :, :, :] = imgs
    masks_all[start:(start + curr_batch_sz), :, :] = masks
    if store_labels:
      labels_all[start:(start + curr_batch_sz), :, :] = labels

    actual_num_imgs += curr_batch_sz

  imgs_all = imgs_all[:actual_num_imgs, :, :, :]
  masks_all = masks_all[:actual_num_imgs, :, :]
  num_unmasked = masks_all.sum()
  if store_labels:
    labels_all = labels_all[:actual_num_imgs, :, :]
    samples_labels = labels_all[masks_all].reshape(-1)
    assert (samples_labels.shape[0] == num_unmasked)

  samples = imgs_all[masks_all, :].reshape(-1, archetype_config.in_channels)
  assert (samples.shape[0] == num_unmasked)

  if not store_labels:
    return samples
  else:
    return samples, samples_labels


def main():
  # based on segmentation_multioutput_twohead - we pass in the config of the
  # IID run we are comparing against, so the settings can be copied

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ind", type=int, required=True)
  parser.add_argument("--out_root", type=str,
                      default="/scratch/shared/slow/xuji/iid_private")
  parser.add_argument("--IID_model_ind", type=int, required=True)
  parser.add_argument("--max_num_train", type=int, required=True)
  parser.add_argument("--test_code", default=False, action="store_true")
  parser.add_argument("--do_sift", default=False, action="store_true")

  config = parser.parse_args()
  config.out_dir = os.path.join(config.out_root, str(config.model_ind))
  if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)

  archetype_config_path = os.path.join(config.out_root,
                                       str(config.IID_model_ind),
                                       "config.pickle")
  print("Loading archetype config from: %s" % archetype_config_path)
  with open(archetype_config_path, "rb") as config_f:
    archetype_config = pickle.load(config_f)
  assert (config.IID_model_ind == archetype_config.model_ind)
  assert (archetype_config.mode == "IID")  # compare against fully unsup

  sample_fn = _get_vectorised_colour_samples
  if config.do_sift:
    sample_fn = _get_vectorised_sift_samples

  # set it to be only rgb (and ir if nec) but no sobel - we're clustering
  # single pixel colours
  archetype_config.include_rgb = True
  archetype_config.no_sobel = True
  if "Coco" in archetype_config.dataset:
    assert (not archetype_config.using_IR)
    archetype_config.in_channels = 3
  elif archetype_config.dataset == "Potsdam":  # IR
    assert (archetype_config.using_IR)
    archetype_config.in_channels = 4

  # Data
  # -------------------------------------------------------------------------
  if "Coco" in archetype_config.dataset:
    dataloaders_head_A, mapping_assignment_dataloader, \
    mapping_test_dataloader = \
      make_Coco_dataloaders(archetype_config)

  elif archetype_config.dataset == "Potsdam":
    dataloaders_head_A, mapping_assignment_dataloader, \
    mapping_test_dataloader = \
      make_Potsdam_dataloaders(archetype_config)
  else:
    raise NotImplementedError

  # unlike in clustering script for STL - isn't any data from unknown classes
  dataloaders_head_B = dataloaders_head_A

  # networks and optimisers
  # ------------------------------------------------------
  assert (archetype_config.num_dataloaders == 1)
  dataloader = dataloaders_head_B[0]

  samples = sample_fn(archetype_config, dataloader)
  print("got training samples")
  sys.stdout.flush()

  if config.test_code:
    print("testing code, taking 10000 samples only")
    samples = samples[:10000, :]
  else:
    num_samples_train = min(samples.shape[0], config.max_num_train)
    print("taking %d samples" % num_samples_train)
    chosen_inds = np.random.choice(samples.shape[0], size=num_samples_train,
                                   replace=False)
    samples = samples[chosen_inds, :]
    print(samples.shape)
  sys.stdout.flush()

  kmeans = MiniBatchKMeans(n_clusters=archetype_config.gt_k, verbose=1).fit(
    samples)
  print("trained kmeans")
  sys.stdout.flush()

  # use mapping assign to assign output_k=gt_k to gt_k
  # and also assess on its predictions, since it's identical to
  # mapping_test_dataloader
  assign_samples, assign_labels = sample_fn(archetype_config,
                                            mapping_assignment_dataloader)
  num_samples = assign_samples.shape[0]
  assign_preds = kmeans.predict(assign_samples)
  print("finished prediction for mapping assign/test data")
  sys.stdout.flush()

  assign_preds = torch.from_numpy(assign_preds).cuda()
  assign_labels = torch.from_numpy(assign_labels).cuda()

  if archetype_config.eval_mode == "hung":
    match = _hungarian_match(assign_preds, assign_labels,
                             preds_k=archetype_config.gt_k,
                             targets_k=archetype_config.gt_k)
  elif archetype_config.eval_mode == "orig":  # flat!
    match = _original_match(assign_preds, assign_labels,
                            preds_k=archetype_config.gt_k,
                            targets_k=archetype_config.gt_k)
  elif archetype_config.eval_mode == "orig_soft":
    assert (False)  # not used

  # reorder predictions to be same cluster assignments as gt_k
  found = torch.zeros(archetype_config.gt_k)
  reordered_preds = torch.zeros(num_samples).to(torch.int32).cuda()
  for pred_i, target_i in match:
    reordered_preds[assign_preds == pred_i] = target_i
    found[pred_i] = 1
  assert (found.sum() == archetype_config.gt_k)  # each output_k must get mapped

  acc = _acc(reordered_preds, assign_labels, archetype_config.gt_k)

  print("got acc %f" % acc)
  config.epoch_acc = [acc]
  config.centroids = kmeans.cluster_centers_
  config.match = match

  # write results and centroids to model_ind output file
  with open(os.path.join(config.out_dir, "config.pickle"), "w") as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
    text_file.write("%s" % config)


if __name__ == "__main__":
  main()
