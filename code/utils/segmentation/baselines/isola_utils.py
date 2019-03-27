from sys import float_info
from sys import stdout as sysout

import numpy as np
import numpy.random as random
import torch

from .general import pol2cart


def isola_loss(adjacent_pred, centre, other, adjacent_gt, mask, verbose=False,
               EPS=float_info.epsilon):
  bn = adjacent_pred.shape[0]
  assert (isinstance(centre, np.ndarray) and isinstance(other, np.ndarray)
          and isinstance(adjacent_gt, bool) and (mask.dtype == torch.uint8))
  assert (centre.shape == (2,) and other.shape == (2,))
  assert (adjacent_pred.is_cuda)
  assert (mask.is_cuda)

  # figure out mask for image pair samples that are located within
  # relevancy_mask - ignore others
  # yes this is a for loop - but it's very negligible

  # allow if at least one of the pairs is in - otherwise too sparse
  # this will allow repulsion to be learned between relevant patches and
  # irrelevant patches, as well as between relevant patches
  mask_centre = mask[:, centre[0], centre[1]]
  mask_other = mask[:, other[0], other[1]]
  mask_per_pred = ((mask_centre + mask_other) > 0).to(torch.float32)

  norm_factor = mask_per_pred.sum().item()

  if verbose:
    print("kept %d out of %d" % (norm_factor, bn))
    print(
      "if stricter, would have kept %d" % (
      mask_centre * mask_other).sum().item())
    sysout.flush()

  # already passed through sigmoid
  adjacent_pred = adjacent_pred.squeeze()
  if adjacent_gt:
    # adjacent_pred = adjacent_pred.clone() # preserves grads

    # avoid NaNs, and avoid counting those
    less_than_eps = (adjacent_pred < EPS).to(torch.float32)
    less_than_eps.requires_grad = False
    not_less_than_eps = 1.0 - less_than_eps

    proximity_to_replace = (adjacent_pred < EPS)
    proximity_to_replace.requires_grad = False
    adjacent_pred[proximity_to_replace] = EPS

    loss = (- (1.0 / norm_factor) * \
            mask_per_pred * not_less_than_eps * \
            torch.log(adjacent_pred)).sum()
  else:
    neg_adjacent_pred = 1.0 - adjacent_pred

    # neg_prox_cl = neg_prox.clone() # preserves grads

    less_than_eps = (neg_adjacent_pred < EPS).to(torch.float32)
    less_than_eps.requires_grad = False
    not_less_than_eps = 1.0 - less_than_eps

    neg_adjacent_pred_to_replace = (neg_adjacent_pred < EPS)
    neg_adjacent_pred_to_replace.requires_grad = False
    neg_adjacent_pred[neg_adjacent_pred_to_replace] = EPS

    loss = - (1.0 / norm_factor) * \
           (mask_per_pred * not_less_than_eps * \
            torch.log(neg_adjacent_pred)).sum()

  if not np.isfinite(loss.cpu().item()):
    print "Isola"
    print loss
    assert (False)

  return loss


def isola_set_patches(input_sz, patch_side):
  # ** this has been tested, from old repo **

  # pick the same pair, the same C for all images
  # NOT necessarily in relevancy mask - ignored later if it isn't

  h, w = input_sz, input_sz
  img_sz = np.array([h, w])

  adjacent = random.rand() < 0.5  # adjacent or not

  d_border = int(np.floor(patch_side / 2.0))
  d_border = np.array([d_border, d_border])

  found = False
  while not found:
    # first patch. Start at least a patch_side and a half away from border
    patch_sz = np.array([patch_side, patch_side])
    start_range = 1.5 * patch_sz
    end_range = img_sz - (1.5 * patch_sz)
    centre = np.floor((random.rand(2) * (end_range - start_range)) +
                      start_range).astype(np.int32)

    if adjacent:
      dh_block = random.choice([-1, 1])
      dw_block = random.choice([-1, 1])
      d = np.array([dh_block * patch_side, dw_block * patch_side])
      other = np.floor(centre + d).astype(np.int32)

      # want to check it's actually not in the border too!
      found = (other >= d_border).all() and (other < (img_sz - d_border)).all()
    else:
      r_start = 2.0 * patch_side
      r_end = max(h, w)

      # use polar coordinates to be sure we're picking at least r_start away
      r = (random.rand() * (r_end - r_start)) + r_start
      phi = (random.rand() * 2.0 * np.pi)
      dh, dw = pol2cart(r, phi)
      d = np.array([dh, dw])

      other = (centre + d).astype(np.int32)

      # need to check it's actually not in the border too!
      found = (other >= d_border).all() and (other < (img_sz - d_border)).all()

  return centre, other, adjacent
