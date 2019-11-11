from __future__ import print_function

import numpy as np
import torch
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def _original_match(flat_preds, flat_targets, preds_k, targets_k):
  # map each output channel to the best matching ground truth (many to one)

  assert (isinstance(flat_preds, torch.Tensor) and
          isinstance(flat_targets, torch.Tensor) and
          flat_preds.is_cuda and flat_targets.is_cuda)

  out_to_gts = {}
  out_to_gts_scores = {}
  for out_c in xrange(preds_k):
    for gt_c in xrange(targets_k):
      # the amount of out_c at all the gt_c samples
      tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())
      if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
        out_to_gts[out_c] = gt_c
        out_to_gts_scores[out_c] = tp_score

  return list(out_to_gts.iteritems())


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
  assert (isinstance(flat_preds, torch.Tensor) and
          isinstance(flat_targets, torch.Tensor) and
          flat_preds.is_cuda and flat_targets.is_cuda)

  num_samples = flat_targets.shape[0]

  assert (preds_k == targets_k)  # one to one
  num_k = preds_k
  num_correct = np.zeros((num_k, num_k))

  for c1 in xrange(num_k):
    for c2 in xrange(num_k):
      # elementwise, so each sample contributes once
      votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
      num_correct[c1, c2] = votes

  # num_correct is small
  match = linear_assignment(num_samples - num_correct)

  # return as list of tuples, out_c to gt_c
  res = []
  for out_c, gt_c in match:
    res.append((out_c, gt_c))

  return res


def _acc(preds, targets, num_k, verbose=0):
  assert (isinstance(preds, torch.Tensor) and
          isinstance(targets, torch.Tensor) and
          preds.is_cuda and targets.is_cuda)

  if verbose >= 2:
    print("calling acc...")

  assert (preds.shape == targets.shape)
  print(preds.shape, torch.max(preds), targets.shape, torch.max(targets), num_k)
  assert (preds.max() <= num_k and targets.max() <= num_k)

  acc = int((preds == targets).sum()) / float(preds.shape[0])

  return acc


def _nmi(preds, targets):
  return metrics.normalized_mutual_info_score(targets, preds)


def _ari(preds, targets):
  return metrics.adjusted_rand_score(targets, preds)
