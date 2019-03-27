import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.cluster import KMeans

from code.utils.cluster.data import _cifar100_to_cifar20, \
  _create_dataloaders, _create_mapping_loader
from code.utils.cluster.eval_metrics import _hungarian_match, _acc
from code.utils.cluster.transforms import sobel_make_transforms, \
  greyscale_make_transforms
from code.utils.cluster.transforms import sobel_process


def make_triplets_data(config):
  target_transform = None

  if "CIFAR" in config.dataset:
    config.train_partitions_head_A = [True, False]
    config.train_partitions_head_B = config.train_partitions_head_A

    config.mapping_assignment_partitions = [True, False]
    config.mapping_test_partitions = [True, False]

    if config.dataset == "CIFAR10":
      dataset_class = torchvision.datasets.CIFAR10
    elif config.dataset == "CIFAR100":
      dataset_class = torchvision.datasets.CIFAR100
    elif config.dataset == "CIFAR20":
      dataset_class = torchvision.datasets.CIFAR100
      target_transform = _cifar100_to_cifar20
    else:
      assert (False)

    # datasets produce either 2 or 5 channel images based on config.include_rgb
    tf1, tf2, tf3 = sobel_make_transforms(config)

  elif config.dataset == "STL10":
    assert (config.mix_train)
    if not config.stl_leave_out_unlabelled:
      print("adding unlabelled data for STL10")
      config.train_partitions_head_A = ["train+unlabeled", "test"]
    else:
      print("not using unlabelled data for STL10")
      config.train_partitions_head_A = ["train", "test"]

    config.train_partitions_head_B = ["train", "test"]

    config.mapping_assignment_partitions = ["train", "test"]
    config.mapping_test_partitions = ["train", "test"]

    dataset_class = torchvision.datasets.STL10

    # datasets produce either 2 or 5 channel images based on config.include_rgb
    tf1, tf2, tf3 = sobel_make_transforms(config)

  elif config.dataset == "MNIST":
    config.train_partitions_head_A = [True, False]
    config.train_partitions_head_B = config.train_partitions_head_A

    config.mapping_assignment_partitions = [True, False]
    config.mapping_test_partitions = [True, False]

    dataset_class = torchvision.datasets.MNIST

    tf1, tf2, tf3 = greyscale_make_transforms(config)

  else:
    assert (False)

  dataloaders = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions=config.train_partitions_head_A,
                        target_transform=target_transform)

  dataloader_original = dataloaders[0]
  dataloader_positive = dataloaders[1]

  shuffled_dataloaders = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions=config.train_partitions_head_A,
                        target_transform=target_transform,
                        shuffle=True)

  dataloader_negative = shuffled_dataloaders[0]

  # since this is fully unsupervised, assign dataloader = test dataloader
  dataloader_test = \
    _create_mapping_loader(config, dataset_class, tf3,
                           partitions=config.mapping_test_partitions,
                           target_transform=target_transform)

  return dataloader_original, dataloader_positive, dataloader_negative, \
         dataloader_test


def triplets_get_data(config, net, dataloader, sobel):
  num_batches = len(dataloader)
  flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                 dtype=torch.int32).cuda()
  flat_preds_all = torch.zeros((num_batches * config.batch_sz),
                               dtype=torch.int32).cuda()

  num_test = 0
  for b_i, batch in enumerate(dataloader):
    imgs = batch[0].cuda()

    if sobel:
      imgs = sobel_process(imgs, config.include_rgb)

    flat_targets = batch[1]

    with torch.no_grad():
      x_outs = net(imgs)

    assert (x_outs.shape[1] == config.output_k)
    assert (len(x_outs.shape) == 2)

    num_test_curr = flat_targets.shape[0]
    num_test += num_test_curr

    start_i = b_i * config.batch_sz
    flat_preds_curr = torch.argmax(x_outs, dim=1)  # along output_k
    flat_preds_all[start_i:(start_i + num_test_curr)] = flat_preds_curr

    flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

  flat_preds_all = flat_preds_all[:num_test]
  flat_targets_all = flat_targets_all[:num_test]

  return flat_preds_all, flat_targets_all


def triplets_get_data_kmeans_on_features(config, net, dataloader, sobel):
  # ouput of network is features (not softmaxed)
  num_batches = len(dataloader)
  flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                 dtype=torch.int32).cuda()
  features_all = np.zeros((num_batches * config.batch_sz, config.output_k),
                          dtype=np.float32)

  num_test = 0
  for b_i, batch in enumerate(dataloader):
    imgs = batch[0].cuda()

    if sobel:
      imgs = sobel_process(imgs, config.include_rgb)

    flat_targets = batch[1]

    with torch.no_grad():
      x_outs = net(imgs)

    assert (x_outs.shape[1] == config.output_k)
    assert (len(x_outs.shape) == 2)

    num_test_curr = flat_targets.shape[0]
    num_test += num_test_curr

    start_i = b_i * config.batch_sz
    features_all[start_i:(start_i + num_test_curr), :] = x_outs.cpu().numpy()
    flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

  features_all = features_all[:num_test, :]
  flat_targets_all = flat_targets_all[:num_test]

  kmeans = KMeans(n_clusters=config.gt_k).fit(features_all)
  flat_preds_all = torch.from_numpy(kmeans.labels_).cuda()

  assert (flat_targets_all.shape == flat_preds_all.shape)
  assert (max(flat_preds_all) < config.gt_k)

  return flat_preds_all, flat_targets_all


def triplets_eval(config, net, dataloader_test, sobel):
  net.eval()

  if not config.kmeans_on_features:
    flat_preds_all, flat_targets_all = triplets_get_data(config, net,
                                                         dataloader_test, sobel)
    assert (config.output_k == config.gt_k)
  else:
    flat_preds_all, flat_targets_all = triplets_get_data_kmeans_on_features(
      config, net, dataloader_test, sobel)

  num_samples = flat_preds_all.shape[0]
  assert (num_samples == flat_targets_all.shape[0])

  net.train()

  match = _hungarian_match(flat_preds_all, flat_targets_all,
                           preds_k=config.gt_k,
                           targets_k=config.gt_k)

  found = torch.zeros(config.gt_k)  # sanity
  reordered_preds = torch.zeros(num_samples,
                                dtype=flat_preds_all.dtype).cuda()

  for pred_i, target_i in match:
    reordered_preds[flat_preds_all == pred_i] = target_i
    found[pred_i] = 1

  assert (found.sum() == config.gt_k)  # each class must get mapped

  mass = np.zeros((1, config.gt_k))
  per_class_acc = np.zeros((1, config.gt_k))
  for c in xrange(config.gt_k):
    flags = (reordered_preds == c)
    actual = (flat_targets_all == c)
    mass[0, c] = flags.sum().item()
    per_class_acc[0, c] = (flags * actual).sum().item()

  acc = _acc(reordered_preds, flat_targets_all, config.gt_k)

  is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))
  config.epoch_acc.append(acc)

  if config.masses is None:
    assert (config.per_class_acc is None)
    config.masses = mass
    config.per_class_acc = per_class_acc
  else:
    config.masses = np.concatenate((config.masses, mass), axis=0)
    config.per_class_acc = np.concatenate(
      (config.per_class_acc, per_class_acc), axis=0)

  return is_best


def triplets_loss(outs_orig, outs_pos, outs_neg):
  orig = F.log_softmax(outs_orig, dim=1)
  pos = F.softmax(outs_pos, dim=1)
  neg = F.softmax(outs_neg, dim=1)

  # loss is minimised
  return F.kl_div(orig, pos, reduction="elementwise_mean") - \
         F.kl_div(orig, neg, reduction="elementwise_mean")
