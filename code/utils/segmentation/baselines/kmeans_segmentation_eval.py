from datetime import datetime
from sys import stdout as sysout

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from code.utils.cluster.eval_metrics import _hungarian_match, _acc, _nmi, _ari
from code.utils.cluster.transforms import sobel_process

GET_NMI_ARI = False


# expects single head architectures (ie not IID/+)
# Train all-in-one using samples. Eval batch using full dataset. Datasets are
#  both the full labelled segments. Could have included unlabelled for former.
#  (only impacts on Potsdam)
def kmeans_segmentation_eval(config, net, test_dataloader):
  net.eval()

  kmeans = train_kmeans(config, net, test_dataloader)

  torch.cuda.empty_cache()

  return apply_trained_kmeans(config, net, test_dataloader, kmeans)


def train_kmeans(config, net, test_dataloader):
  num_imgs = len(test_dataloader.dataset)
  max_num_pixels_per_img = int(config.max_num_kmeans_samples / num_imgs)

  features_all = np.zeros(
    (config.max_num_kmeans_samples, net.module.features_sz),
    dtype=np.float32)

  actual_num_features = 0

  # discard the label information in the dataloader
  for i, tup in enumerate(test_dataloader):
    if (config.verbose and i < 10) or (i % int(len(test_dataloader) / 10) == 0):
      print("(kmeans_segmentation_eval) batch %d time %s" % (i, datetime.now()))
      sysout.flush()

    imgs, _, mask = tup  # test dataloader, cpu tensors
    imgs = imgs.cuda()
    mask = mask.numpy().astype(np.bool)
    # mask = mask.numpy().astype(np.bool)
    num_unmasked = mask.sum()

    if not config.no_sobel:
      imgs = sobel_process(imgs, config.include_rgb, using_IR=config.using_IR)
      # now rgb(ir) and/or sobel

    with torch.no_grad():
      # penultimate = features
      x_out = net(imgs, penultimate=True).cpu().numpy()

    if config.verbose and i < 2:
      print("(kmeans_segmentation_eval) through model %d time %s" % (i,
                                                                     datetime.now()))
      sysout.flush()

    num_imgs_batch = x_out.shape[0]
    x_out = x_out.transpose((0, 2, 3, 1))  # features last

    x_out = x_out[mask, :]

    if config.verbose and i < 2:
      print("(kmeans_segmentation_eval) applied mask %d time %s" % (i,
                                                                    datetime.now()))
      sysout.flush()

    if i == 0:
      assert (x_out.shape[1] == net.module.features_sz)
      assert (x_out.shape[0] == num_unmasked)

    # select pixels randomly, and record how many selected
    num_selected = min(num_unmasked, num_imgs_batch * max_num_pixels_per_img)
    selected = np.random.choice(num_selected, replace=False)

    x_out = x_out[selected, :]

    if config.verbose and i < 2:
      print("(kmeans_segmentation_eval) applied select %d time %s" % (i,
                                                                      datetime.now()))
      sysout.flush()

    features_all[actual_num_features:actual_num_features + num_selected, :] = \
      x_out

    actual_num_features += num_selected

    if config.verbose and i < 2:
      print("(kmeans_segmentation_eval) stored %d time %s" % (i,
                                                              datetime.now()))
      sysout.flush()

  assert (actual_num_features <= config.max_num_kmeans_samples)
  features_all = features_all[:actual_num_features, :]

  if config.verbose:
    print("running kmeans")
    sysout.flush()
  kmeans = MiniBatchKMeans(n_clusters=config.gt_k, verbose=config.verbose).fit(
    features_all)

  return kmeans


def apply_trained_kmeans(config, net, test_dataloader, kmeans):
  if config.verbose:
    print("starting inference")
    sysout.flush()

  # on the entire test dataset
  num_imgs = len(test_dataloader.dataset)
  max_num_samples = num_imgs * config.input_sz * config.input_sz
  preds_all = torch.zeros(max_num_samples, dtype=torch.int32).cuda()
  targets_all = torch.zeros(max_num_samples, dtype=torch.int32).cuda()

  actual_num_unmasked = 0

  # discard the label information in the dataloader
  for i, tup in enumerate(test_dataloader):
    if (config.verbose and i < 10) or (i % int(len(test_dataloader) / 10) == 0):
      print("(apply_trained_kmeans) batch %d time %s" % (i, datetime.now()))
      sysout.flush()

    imgs, targets, mask = tup  # test dataloader, cpu tensors
    imgs, mask_cuda, targets, mask_np = imgs.cuda(), mask.cuda(), \
                                        targets.cuda(), mask.numpy().astype(
      np.bool)
    num_unmasked = mask_cuda.sum().item()

    if not config.no_sobel:
      imgs = sobel_process(imgs, config.include_rgb, using_IR=config.using_IR)
      # now rgb(ir) and/or sobel

    with torch.no_grad():
      # penultimate = features
      x_out = net(imgs, penultimate=True).cpu().numpy()

    x_out = x_out.transpose((0, 2, 3, 1))  # features last
    x_out = x_out[mask_np, :]
    targets = targets.masked_select(mask_cuda)  # can do because flat

    assert (x_out.shape == (num_unmasked, net.module.features_sz))
    preds = torch.from_numpy(kmeans.predict(x_out)).cuda()

    preds_all[actual_num_unmasked: actual_num_unmasked + num_unmasked] = preds
    targets_all[
    actual_num_unmasked: actual_num_unmasked + num_unmasked] = targets

    actual_num_unmasked += num_unmasked

  preds_all = preds_all[:actual_num_unmasked]
  targets_all = targets_all[:actual_num_unmasked]

  torch.cuda.empty_cache()

  # permutation, not many-to-one
  match = _hungarian_match(preds_all, targets_all, preds_k=config.gt_k,
                           targets_k=config.gt_k)
  torch.cuda.empty_cache()

  # do in cpu because of RAM
  reordered_preds = torch.zeros(actual_num_unmasked, dtype=preds_all.dtype)
  for pred_i, target_i in match:
    selected = (preds_all == pred_i).cpu()
    reordered_preds[selected] = target_i

  reordered_preds = reordered_preds.cuda()

  # this checks values
  acc = _acc(reordered_preds, targets_all, config.gt_k, verbose=config.verbose)

  if GET_NMI_ARI:
    nmi, ari = _nmi(reordered_preds, targets_all), \
               _ari(reordered_preds, targets_all)
  else:
    nmi, ari = -1., -1.

  reordered_masses = np.zeros(config.gt_k)
  for c in xrange(config.gt_k):
    reordered_masses[c] = float(
      (reordered_preds == c).sum()) / actual_num_unmasked

  return acc, nmi, ari, reordered_masses
