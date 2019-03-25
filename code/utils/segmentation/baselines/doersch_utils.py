import numpy as np
import torch

from .general import pol2cart


def doersch_set_patches(input_sz, patch_side):
  # pick the same locations for all images, for computational ease
  # not necessarily in relevancy mask of each image - ignored later if it isn't
  # unlike isola_set_patches, there are random gaps between patches

  # there are 9 2D positions (N, NE.. NW)

  h, w = input_sz, input_sz
  img_sz = np.array([h, w])

  d_border = int(np.floor(patch_side / 2.0))
  d_border = np.array([d_border, d_border])
  found = False
  while not found:
    position_gt = np.random.choice(9)

    # first patch. Start at least a patch_side and a half away from border
    patch_sz = np.array([patch_side, patch_side])
    start_range = 1.5 * patch_sz
    end_range = img_sz - (1.5 * patch_sz)
    centre = np.floor((np.random.rand(2) * (end_range - start_range)) +
                      start_range).astype("int")

    r_start = 1.5 * patch_side  # min distance from centre
    r_end = 2.0 * patch_side  # max distance from centre

    # use polar coordinates to be sure we're picking at least r_start away
    r = (np.random.rand() * (r_end - r_start)) + r_start
    phi = (position_gt * np.pi / 4.)
    dh, dw = pol2cart(r, phi)
    d = np.array([dh, dw])

    other = (centre + d).astype(np.int32)

    # need to check it's actually not in the border too
    found = (other >= d_border).all() and (other < (img_sz - d_border)).all()

  return centre, other, position_gt


def doersch_loss(position_pred, centre, other, position_gt, mask,
                 crossent, verbose):
  bn, num_positions = position_pred.shape
  assert (num_positions == 9)

  # allow if at least one of the pairs is in - otherwise too sparse
  # this will allow repulsion to be learned between relevant patches and
  # irrelevant patches, as well as between relevant patches
  mask_centre = mask[:, centre[0], centre[1]]
  mask_other = mask[:, other[0], other[1]]
  mask_per_pred = ((mask_centre + mask_other) > 0).to(torch.float32)
  assert (mask_per_pred.shape == (bn,))

  norm_factor = mask_per_pred.sum().item()

  # use CrossEntropyLoss
  gt = torch.ones(bn, dtype=torch.int64).cuda() * position_gt
  per_elem_loss = crossent(position_pred, gt)
  assert (per_elem_loss.shape == (bn,))
  loss = (mask_per_pred * per_elem_loss).sum() / norm_factor

  return loss
