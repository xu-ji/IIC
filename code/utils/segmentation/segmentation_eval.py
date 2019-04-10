from __future__ import print_function

import sys
from datetime import datetime

import torch

from code.utils.cluster.cluster_eval import cluster_subheads_eval
from code.utils.cluster.transforms import sobel_process


def segmentation_eval(config, net,
                      mapping_assignment_dataloader,
                      mapping_test_dataloader,
                      sobel, using_IR=False, verbose=0, return_only=False):
  torch.cuda.empty_cache()
  net.eval()

  stats_dict = cluster_subheads_eval(config, net,
                                     mapping_assignment_dataloader=mapping_assignment_dataloader,
                                     mapping_test_dataloader=mapping_test_dataloader,
                                     sobel=sobel,
                                     using_IR=using_IR,
                                     get_data_fn=_segmentation_get_data,
                                     verbose=verbose)

  net.train()

  acc = stats_dict["best"]
  is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

  torch.cuda.empty_cache()

  if not return_only:
    config.epoch_stats.append(stats_dict)
    config.epoch_acc.append(acc)
    config.epoch_avg_subhead_acc.append(stats_dict["avg"])

    return is_best
  else:
    return stats_dict


def _segmentation_get_data(config, net, dataloader, sobel=False,
                           using_IR=False, verbose=0):
  # returns (vectorised) cuda tensors for flat preds and targets
  # sister of _clustering_get_data

  assert (config.output_k <= 255)

  num_batches = len(dataloader)
  num_samples = 0

  # upper bound, will be less for last batch
  samples_per_batch = config.batch_sz * config.input_sz * config.input_sz

  if verbose > 0:
    print("started _segmentation_get_data %s" % datetime.now())
    sys.stdout.flush()

  # vectorised
  flat_predss_all = [torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda() for _ in range(
    config.num_sub_heads)]
  flat_targets_all = torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda()
  mask_all = torch.zeros((num_batches * samples_per_batch),
                         dtype=torch.uint8).cuda()

  if verbose > 0:
    batch_start = datetime.now()
    all_start = batch_start
    print("starting batches %s" % batch_start)

  for b_i, batch in enumerate(dataloader):

    imgs, flat_targets, mask = batch
    imgs = imgs.cuda()

    if sobel:
      imgs = sobel_process(imgs, config.include_rgb, using_IR=using_IR)

    with torch.no_grad():
      x_outs = net(imgs)

    assert (x_outs[0].shape[1] == config.output_k)
    assert (x_outs[0].shape[2] == config.input_sz and x_outs[0].shape[
      3] == config.input_sz)

    # actual batch size
    actual_samples_curr = (
      flat_targets.shape[0] * config.input_sz * config.input_sz)
    num_samples += actual_samples_curr

    # vectorise: collapse from 2D to 1D
    start_i = b_i * samples_per_batch
    for i in range(config.num_sub_heads):
      x_outs_curr = x_outs[i]
      assert (not x_outs_curr.requires_grad)
      flat_preds_curr = torch.argmax(x_outs_curr, dim=1)
      flat_predss_all[i][
      start_i:(start_i + actual_samples_curr)] = flat_preds_curr.view(-1)

    flat_targets_all[
    start_i:(start_i + actual_samples_curr)] = flat_targets.view(-1)
    mask_all[start_i:(start_i + actual_samples_curr)] = mask.view(-1)

    if verbose > 0 and b_i < 3:
      batch_finish = datetime.now()
      print("finished batch %d, %s, took %s, of %d" %
            (b_i, batch_finish, batch_finish - batch_start, num_batches))
      batch_start = batch_finish
      sys.stdout.flush()

  if verbose > 0:
    all_finish = datetime.now()
    print(
      "finished all batches %s, took %s" % (all_finish, all_finish - all_start))
    sys.stdout.flush()

  flat_predss_all = [flat_predss_all[i][:num_samples] for i in
                     range(config.num_sub_heads)]
  flat_targets_all = flat_targets_all[:num_samples]
  mask_all = mask_all[:num_samples]

  flat_predss_all = [flat_predss_all[i].masked_select(mask=mask_all) for i in
                     range(config.num_sub_heads)]
  flat_targets_all = flat_targets_all.masked_select(mask=mask_all)

  if verbose > 0:
    print("ended _segmentation_get_data %s" % datetime.now())
    sys.stdout.flush()

  selected_samples = mask_all.sum()
  assert (len(flat_predss_all[0].shape) == 1 and
          len(flat_targets_all.shape) == 1)
  assert (flat_predss_all[0].shape[0] == selected_samples)
  assert (flat_targets_all.shape[0] == selected_samples)

  return flat_predss_all, flat_targets_all
