import torch
import torchvision
from torch.optim import Adam

_opt_dict = {"Adam": Adam}


def get_opt(name):
  return _opt_dict[name]


def config_to_str(config):
  attrs = vars(config)
  string_val = "Config: -----\n"
  string_val += "\n".join("%s: %s" % item for item in attrs.items())
  string_val += "\n----------"
  return string_val


def update_lr(optimiser, lr_mult=0.1):
  for param_group in optimiser.param_groups:
    param_group['lr'] *= lr_mult
  return optimiser


def reorder_train_deterministic(dataset):
  assert (isinstance(dataset, torchvision.datasets.STL10))
  assert (dataset.split == "train+unlabeled")

  # move first 5k into rest of 100k
  # one every 20 images
  assert (dataset.data.shape == (105000, 3, 96, 96))

  # 0, 5000...5019, 1, 5020...5039, 2, ... 4999, 104980 ... 104999
  ids = []
  for i in range(5000):
    ids.append(i)
    ids += range(5000 + i * 20, 5000 + (i + 1) * 20)

  dataset.data = dataset.data[ids]
  assert (dataset.data.shape == (105000, 3, 96, 96))
  dataset.labels = dataset.labels[ids]
  assert (dataset.labels.shape == (105000,))

  return dataset


def print_weights_and_grad(net):
  print("---------------")
  for n, p in net.named_parameters():
    print("%s abs: min %f max %f max grad %f" %
          (n, torch.abs(p.data).min(), torch.abs(p.data).max(), \
           torch.abs(p.grad).max()))
  print("---------------")


def nice(dict):
  res = ""
  for k, v in dict.items():
    res += ("\t%s: %s\n" % (k, v))
  return res
