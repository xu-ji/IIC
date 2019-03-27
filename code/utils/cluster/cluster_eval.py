from __future__ import print_function

import sys
from datetime import datetime

import numpy as np
import torch

from .eval_metrics import _hungarian_match, _original_match, _acc
from .transforms import sobel_process


def _clustering_get_data(config, net, dataloader, sobel=False,
                         using_IR=False, get_soft=False, verbose=0):
  """
  Returns cuda tensors for flat preds and targets.
  """

  assert (not using_IR)  # sanity; IR used by segmentation only

  num_batches = len(dataloader)
  flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                 dtype=torch.int32).cuda()
  flat_predss_all = [torch.zeros((num_batches * config.batch_sz),
                                 dtype=torch.int32).cuda() for _ in
                     xrange(config.num_sub_heads)]

  if get_soft:
    soft_predss_all = [torch.zeros((num_batches * config.batch_sz,
                                    config.output_k),
                                   dtype=torch.float32).cuda() for _ in xrange(
      config.num_sub_heads)]

  num_test = 0
  for b_i, batch in enumerate(dataloader):
    imgs = batch[0].cuda()

    if sobel:
      imgs = sobel_process(imgs, config.include_rgb, using_IR=using_IR)

    flat_targets = batch[1]

    with torch.no_grad():
      x_outs = net(imgs)

    assert (x_outs[0].shape[1] == config.output_k)
    assert (len(x_outs[0].shape) == 2)

    num_test_curr = flat_targets.shape[0]
    num_test += num_test_curr

    start_i = b_i * config.batch_sz
    for i in xrange(config.num_sub_heads):
      x_outs_curr = x_outs[i]
      flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
      flat_predss_all[i][start_i:(start_i + num_test_curr)] = flat_preds_curr

      if get_soft:
        soft_predss_all[i][start_i:(start_i + num_test_curr), :] = x_outs_curr

    flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

  flat_predss_all = [flat_predss_all[i][:num_test] for i in
                     xrange(config.num_sub_heads)]
  flat_targets_all = flat_targets_all[:num_test]

  if not get_soft:
    return flat_predss_all, flat_targets_all
  else:
    soft_predss_all = [soft_predss_all[i][:num_test] for i in
                       xrange(config.num_sub_heads)]

    return flat_predss_all, flat_targets_all, soft_predss_all


def cluster_subheads_eval(config, net,
                          mapping_assignment_dataloader,
                          mapping_test_dataloader,
                          sobel,
                          using_IR=False,
                          get_data_fn=_clustering_get_data,
                          verbose=0):
  """
  Used by both clustering and segmentation.
  Returns metrics for test set.
  Get result from average accuracy of all sub_heads (mean and std).
  All matches are made from training data.
  Best head metric, which is order selective unlike mean/std, is taken from 
  best head determined by training data (but metric computed on test data).
  
  ^ detail only matters for IID+/semisup where there's a train/test split.
  """

  all_matches, train_accs = _get_assignment_data_matches(net,
                                                         mapping_assignment_dataloader,
                                                         config,
                                                         sobel=sobel,
                                                         using_IR=using_IR,
                                                         get_data_fn=get_data_fn,
                                                         verbose=verbose)

  best_sub_head = np.argmax(train_accs)

  if config.mode == "IID":
    assert (
      config.mapping_assignment_partitions == config.mapping_test_partitions)
    test_accs = train_accs
  elif config.mode == "IID+":
    flat_predss_all, flat_targets_all, = \
      get_data_fn(config, net, mapping_test_dataloader, sobel=sobel,
                  using_IR=using_IR,
                  verbose=verbose)

    num_samples = flat_targets_all.shape[0]
    test_accs = np.zeros(config.num_sub_heads, dtype=np.float32)
    for i in xrange(config.num_sub_heads):
      reordered_preds = torch.zeros(num_samples,
                                    dtype=flat_predss_all[0].dtype).cuda()
      for pred_i, target_i in all_matches[i]:
        reordered_preds[flat_predss_all[i] == pred_i] = target_i
      test_acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose=0)

      test_accs[i] = test_acc
  else:
    assert (False)

  if test_accs.max() != test_accs[best_sub_head]:
    assert (config.mode != "IID")
    print("Training data best head is not same as test data best head - only "
          "possible for IID+ (using training)")

  return {"test_accs": list(test_accs),
          "avg": np.mean(test_accs),
          "std": np.std(test_accs),
          "best": test_accs[best_sub_head],  # head selected from training data
          "worst": test_accs.min(),
          "best_train_sub_head": best_sub_head,  # from training data
          "best_train_sub_head_match": all_matches[best_sub_head],
          "train_accs": list(train_accs)}


def _get_assignment_data_matches(net, mapping_assignment_dataloader, config,
                                 sobel=False,
                                 using_IR=False,
                                 get_data_fn=None,
                                 just_matches=False,
                                 verbose=0):
  """
  Get all best matches per head based on train set i.e. mapping_assign,
  and mapping_assign accs.
  """

  if verbose:
    print("calling cluster eval direct (helper) %s" % datetime.now())
    sys.stdout.flush()

  flat_predss_all, flat_targets_all = \
    get_data_fn(config, net, mapping_assignment_dataloader, sobel=sobel,
                using_IR=using_IR,
                verbose=verbose)

  if verbose:
    print("getting data fn has completed %s" % datetime.now())
    print("flat_targets_all %s, flat_predss_all[0] %s" %
          (list(flat_targets_all.shape), list(flat_predss_all[0].shape)))
    sys.stdout.flush()

  num_test = flat_targets_all.shape[0]
  if verbose == 2:
    print("num_test: %d" % num_test)
    for c in xrange(config.gt_k):
      print("gt_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

  assert (flat_predss_all[0].shape == flat_targets_all.shape)
  num_samples = flat_targets_all.shape[0]

  all_matches = []
  if not just_matches:
    all_accs = np.zeros(config.num_sub_heads, dtype=np.float32)

  for i in xrange(config.num_sub_heads):
    if verbose:
      print("starting head %d with eval mode %s, %s" % (i, config.eval_mode,
                                                        datetime.now()))
      sys.stdout.flush()

    if config.eval_mode == "hung":
      match = _hungarian_match(flat_predss_all[i], flat_targets_all,
                               preds_k=config.output_k,
                               targets_k=config.gt_k)
    elif config.eval_mode == "orig":
      match = _original_match(flat_predss_all[i], flat_targets_all,
                              preds_k=config.output_k,
                              targets_k=config.gt_k)
    else:
      assert (False)

    if verbose:
      print("got match %s" % (datetime.now()))
      sys.stdout.flush()

    all_matches.append(match)

    if not just_matches:
      # reorder predictions to be same cluster assignments as gt_k
      found = torch.zeros(config.output_k)
      reordered_preds = torch.zeros(num_samples,
                                    dtype=flat_predss_all[0].dtype).cuda()

      for pred_i, target_i in match:
        reordered_preds[flat_predss_all[i] == pred_i] = target_i
        found[pred_i] = 1
        if verbose == 2:
          print((pred_i, target_i))
      assert (found.sum() == config.output_k)  # each output_k must get mapped

      if verbose:
        print("reordered %s" % (datetime.now()))
        sys.stdout.flush()

      acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose)
      all_accs[i] = acc

  if just_matches:
    return all_matches
  else:
    return all_matches, all_accs


def cluster_eval(config, net, mapping_assignment_dataloader,
                 mapping_test_dataloader, sobel, print_stats=False):
  if config.double_eval:
    # Pytorch's behaviour varies depending on whether .eval() is called or not
    # The effect is batchnorm updates if .eval() is not called
    # So double eval can be used (optionally) for IID, where train = test set.
    # https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm2d

    stats_dict2 = cluster_subheads_eval(config, net,
                                        mapping_assignment_dataloader=mapping_assignment_dataloader,
                                        mapping_test_dataloader=mapping_test_dataloader,
                                        sobel=sobel)

    if print_stats:
      print("double eval stats:")
      print(stats_dict2)
    else:
      config.double_eval_stats.append(stats_dict2)
      config.double_eval_acc.append(stats_dict2["best"])
      config.double_eval_avg_subhead_acc.append(stats_dict2["avg"])

  net.eval()
  stats_dict = cluster_subheads_eval(config, net,
                                     mapping_assignment_dataloader=mapping_assignment_dataloader,
                                     mapping_test_dataloader=mapping_test_dataloader,
                                     sobel=sobel)
  net.train()

  if print_stats:
    print("eval stats:")
    print(stats_dict)
  else:
    acc = stats_dict["best"]
    is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

    config.epoch_stats.append(stats_dict)
    config.epoch_acc.append(acc)
    config.epoch_avg_subhead_acc.append(stats_dict["avg"])

    return is_best
