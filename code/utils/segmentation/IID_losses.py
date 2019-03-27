from sys import float_info

import torch
import torch.nn.functional as F

from .render import render
from .transforms import perform_affine_tf, random_translation_multiple

EPS = float_info.epsilon

RENDER = False


def IID_segmentation_loss(x1_outs, x2_outs, all_affine2_to_1=None,
                          all_mask_img1=None, lamb=1.0,
                          half_T_side_dense=None,
                          half_T_side_sparse_min=None,
                          half_T_side_sparse_max=None):
  assert (x1_outs.requires_grad)
  assert (x2_outs.requires_grad)
  assert (not all_affine2_to_1.requires_grad)
  assert (not all_mask_img1.requires_grad)

  assert (x1_outs.shape == x2_outs.shape)

  # bring x2 back into x1's spatial frame
  x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

  if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
    x2_outs_inv = random_translation_multiple(x2_outs_inv,
                                              half_side_min=half_T_side_sparse_min,
                                              half_side_max=half_T_side_sparse_max)

  if RENDER:
    # indices added to each name by render()
    render(x1_outs, mode="image_as_feat", name="invert_img1_")
    render(x2_outs, mode="image_as_feat", name="invert_img2_pre_")
    render(x2_outs_inv, mode="image_as_feat", name="invert_img2_post_")
    render(all_mask_img1, mode="mask", name="invert_mask_")

  # zero out all irrelevant patches
  bn, k, h, w = x1_outs.shape
  all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
  x1_outs = x1_outs * all_mask_img1  # broadcasts
  x2_outs_inv = x2_outs_inv * all_mask_img1

  # sum over everything except classes, by convolving x1_outs with x2_outs_inv
  # which is symmetric, so doesn't matter which one is the filter
  x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
  x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

  # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
  p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(half_T_side_dense,
                                                         half_T_side_dense))
  p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

  # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
  # also, some pixels did not have a completely unmasked box neighbourhood,
  # but it's fine - just less samples from that pixel
  current_norm = float(p_i_j.sum())
  p_i_j = p_i_j / current_norm

  # symmetrise
  p_i_j = (p_i_j + p_i_j.t()) / 2.

  # compute marginals
  p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
  p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

  # for log stability; tiny values cancelled out by mult with p_i_j anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_i_mat[(p_i_mat < EPS).data] = EPS
  p_j_mat[(p_j_mat < EPS).data] = EPS

  # maximise information
  loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                    lamb * torch.log(p_j_mat))).sum()

  # for analysis only
  loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                            torch.log(p_j_mat))).sum()

  return loss, loss_no_lamb


def IID_segmentation_loss_uncollapsed(x1_outs, x2_outs, all_affine2_to_1=None,
                                      all_mask_img1=None, lamb=1.0,
                                      half_T_side_dense=None,
                                      half_T_side_sparse_min=None,
                                      half_T_side_sparse_max=None):
  assert (x1_outs.requires_grad)
  assert (x2_outs.requires_grad)
  assert (not all_affine2_to_1.requires_grad)
  assert (not all_mask_img1.requires_grad)

  assert (x1_outs.shape == x2_outs.shape)

  # bring x2 back into x1's spatial frame
  x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

  if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
    x2_outs_inv = random_translation_multiple(x2_outs_inv,
                                              half_side_min=half_T_side_sparse_min,
                                              half_side_max=half_T_side_sparse_max)

  if RENDER:
    # indices added to each name by render()
    render(x1_outs, mode="image_as_feat", name="invert_img1_")
    render(x2_outs, mode="image_as_feat", name="invert_img2_pre_")
    render(x2_outs_inv, mode="image_as_feat", name="invert_img2_post_")
    render(all_mask_img1, mode="mask", name="invert_mask_")

  # zero out all irrelevant patches
  bn, k, h, w = x1_outs.shape
  all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
  x1_outs = x1_outs * all_mask_img1  # broadcasts
  x2_outs_inv = x2_outs_inv * all_mask_img1

  # sum over everything except classes, by convolving x1_outs with x2_outs_inv
  # which is symmetric, so doesn't matter which one is the filter
  x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
  x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

  # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
  p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(half_T_side_dense,
                                                         half_T_side_dense))

  # do expectation over each shift location in the T_side_dense *
  # T_side_dense box
  T_side_dense = half_T_side_dense * 2 + 1

  # T x T x k x k
  p_i_j = p_i_j.permute(2, 3, 0, 1)
  p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2,
                                                     keepdim=True)  # norm

  # symmetrise, transpose the k x k part
  p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

  # T x T x k x k
  p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
  p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

  # for log stability; tiny values cancelled out by mult with p_i_j anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_i_mat[(p_i_mat < EPS).data] = EPS
  p_j_mat[(p_j_mat < EPS).data] = EPS

  # maximise information
  loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                    lamb * torch.log(p_j_mat))).sum() / (
           T_side_dense * T_side_dense)

  # for analysis only
  loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                            torch.log(p_j_mat))).sum() / (
                   T_side_dense * T_side_dense)

  return loss, loss_no_lamb
