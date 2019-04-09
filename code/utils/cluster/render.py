import os
import sys
from colorsys import hsv_to_rgb

import numpy as np
import torch
from PIL import Image

from code.utils.cluster.cluster_eval import _get_assignment_data_matches, \
  _clustering_get_data

# for all heads/models, keep the colouring consistent
GT_TO_ORDER = [2, 5, 3, 8, 6, 7, 0, 9, 1, 4]


def save_progress(config, net, mapping_assignment_dataloader,
                  mapping_test_dataloader, index, sobel, render_count):
  """
  Draws all predictions using convex combination.
  """

  # Using this for MNIST
  if sobel:
    raise NotImplementedError

  prog_out_dir = os.path.join(config.out_dir, "progression")
  if not os.path.exists(prog_out_dir):
    os.makedirs(prog_out_dir)

  # find the best head
  using_IR = False  # whole images
  all_matches, train_accs = _get_assignment_data_matches(net,
                                                         mapping_assignment_dataloader,
                                                         config,
                                                         sobel=sobel,
                                                         using_IR=using_IR,
                                                         get_data_fn=_clustering_get_data)

  best_sub_head = np.argmax(train_accs)
  match = all_matches[best_sub_head]

  # get clustering results
  flat_predss_all, flat_targets_all, soft_predss_all = \
    _clustering_get_data(config, net, mapping_test_dataloader, sobel=sobel,
                         using_IR=using_IR, get_soft=True)
  soft_preds = soft_predss_all[best_sub_head]

  num_samples, C = soft_preds.shape
  assert (C == config.gt_k)
  reordered_soft_preds = torch.zeros((num_samples, config.gt_k),
                                     dtype=soft_preds.dtype).cuda()
  for pred_i, target_i in match:
    reordered_soft_preds[:, GT_TO_ORDER[target_i]] += \
      soft_preds[:, pred_i]  # 1-1 for IIC
  reordered_soft_preds = reordered_soft_preds.cpu().numpy()

  # render point cloud in GT order ---------------------------------------------
  hues = torch.linspace(0.0, 1.0, config.gt_k + 1)[0:-1]  # ignore last one
  best_colours = [list((np.array(hsv_to_rgb(hue, 0.8, 0.8)) * 255.).astype(
    np.uint8)) for hue in hues]

  all_colours = [best_colours]

  for colour_i, colours in enumerate(all_colours):
    scale = 50  # [-1, 1] -> [-scale, scale]
    border = 24  # averages are in the borders
    point_half_side = 1  # size 2 * pixel_half_side + 1

    half_border = int(border * 0.5)

    average_half_side = int(half_border * np.cos(np.radians(45)))
    average_side = average_half_side * 2

    image = np.ones((2 * (scale + border), 2 * (scale + border), 3),
                    dtype=np.uint8) * 255

    # image = np.zeros((2 * (scale + border), 2 * (scale + border), 3),
    #                dtype=np.int32)

    for i in range(num_samples):
      # in range [-1, 1] -> [0, 2 * scale] -> [border, 2 * scale + border]
      coord = get_coord(reordered_soft_preds[i, :], num_classes=config.gt_k)
      coord = (coord * scale + scale).astype(np.int32)
      coord += border
      pt_start = coord - point_half_side
      pt_end = coord + point_half_side

      render_c = GT_TO_ORDER[flat_targets_all[i]]
      colour = (np.array(colours[render_c])).astype(np.uint8)
      image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = np.reshape(
        colour, (1, 1, 3))

    # add on the average image per cluster in the border
    # -------------------------
    # dataloaders not shuffled, or jittered here
    averaged_imgs = [np.zeros((config.input_sz, config.input_sz, 1)) for _ in
                     range(config.gt_k)]
    averaged_imgs_norm = [0. for _ in range(config.gt_k)]
    counter = 0
    for b_i, batch in enumerate(mapping_test_dataloader):
      imgs = batch[0].numpy()  # n, c, h, w
      n, c, h, w = imgs.shape
      assert (c == 1)

      for offset in range(n):
        img_i = counter + offset
        img = imgs[offset]
        img = img.transpose((1, 2, 0))
        img = img * 255

        # already in right order
        predicted_cluster_render = np.argmax(reordered_soft_preds[img_i, :])
        predicted_cluster_gt_weight = reordered_soft_preds[
          img_i, predicted_cluster_render]

        averaged_imgs[predicted_cluster_render] += \
          predicted_cluster_gt_weight * img
        averaged_imgs_norm[predicted_cluster_render] += \
          predicted_cluster_gt_weight

      counter += n

    for c in range(config.gt_k):
      if averaged_imgs_norm[c] > (sys.float_info.epsilon):
        averaged_imgs[c] /= averaged_imgs_norm[c]

      averaged_img = averaged_imgs[c].astype(np.uint8)
      averaged_img = averaged_img.repeat(3, axis=2)
      averaged_img = Image.fromarray(averaged_img)
      averaged_img = averaged_img.resize((average_side, average_side),
                                         Image.BILINEAR)
      averaged_img = np.array(averaged_img)

      coord = np.zeros(config.gt_k)
      coord[c] = 1.
      coord = get_coord(coord, num_classes=config.gt_k)

      # recall coord is for center of image
      # [-1, 1] -> [0, 2 * (scale + half_border)]
      coord = (coord * (scale + half_border) + (scale + half_border)).astype(
        np.int32)
      # -> [half_border, 2 * (scale + half_border) + half_border]
      coord += half_border

      pt_start = coord - average_half_side
      pt_end = coord + average_half_side  # exclusive

      image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = averaged_img

    # save to out_dir ---------------------------
    img = Image.fromarray(image)
    img.save(os.path.join(prog_out_dir,
                          "%d_run_%d_colour_%d_pointcloud_%s.png" %
                          (config.model_ind, render_count, colour_i, index)))


def get_coord(probs, num_classes):
  # computes coordinate for 1 sample based on probability distribution over c
  coords_total = np.zeros(2, dtype=np.float32)
  probs_sum = probs.sum()

  fst_angle = 0.

  for c in range(num_classes):
    # compute x, y coordinates
    coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle
    coords[0] = np.sin(coords[0])
    coords[1] = np.cos(coords[1])
    coords_total += (probs[c] / probs_sum) * coords
  return coords_total
