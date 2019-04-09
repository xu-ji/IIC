import numpy as np
from sklearn.cluster import KMeans

from .cluster_eval import _acc, _original_match, _hungarian_match


def multioutput_k_means_assess(config, x_outs_all, targets, verbose=0):
  assert (False)  # outdated function
  num_sub_heads = len(x_outs_all)
  print("assessing multioutput using k-means, heads: %d" % num_sub_heads)

  accs = []
  nmis = []
  aris = []
  best_i = None
  for i in range(num_sub_heads):
    x_outs = x_outs_all[i]  # not flat
    n, dlen = x_outs.shape
    # random_state=0
    kmeans = KMeans(n_clusters=config.gt_k).fit(x_outs)

    n2, = targets.shape
    assert (n == n2)
    assert (max(targets) == (config.gt_k - 1))

    flat_predictions = kmeans.labels_

    # get into same indices

    if config.kmeans_map == "many_to_one":
      assert (config.eval_mode == "orig")
      match = _original_match(flat_predictions, targets,
                              preds_k=config.output_k,
                              targets_k=config.gt_k)
    elif config.kmeans_map == "one_to_one":
      # hungarian
      match = _hungarian_match(flat_predictions, targets,
                               preds_k=config.output_k,
                               targets_k=config.gt_k)
    else:
      assert (False)

    # reorder predictions to be same cluster assignments as gt_k
    reordered_preds = np.zeros(n)
    for pred_i, target_i in match:
      reordered_preds[flat_predictions == pred_i] = target_i
      if verbose > 1:
        print((pred_i, target_i))

    acc = _acc(reordered_preds, targets, config.gt_k, verbose)

    # this works because for max acc, will get set and never un-set
    if (best_i is None) or (acc > max(accs)):
      best_i = i

    if verbose > 0:
      print("head %d acc %f" % (i, acc))

    accs.append(acc)

  return accs[best_i], nmis[best_i], aris[best_i]
