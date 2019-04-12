import sys
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

from code.datasets.segmentation import DoerschDataset
from code.datasets.segmentation import cocostuff
from code.datasets.segmentation import potsdam


def segmentation_create_dataloaders(config):
  if config.mode == "IID+":
    if "Coco10k" in config.dataset:
      config.train_partitions = ["train"]
      config.mapping_assignment_partitions = ["train"]
      config.mapping_test_partitions = ["test"]
    elif "Coco164k" in config.dataset:
      config.train_partitions = ["train2017"]
      config.mapping_assignment_partitions = ["train2017"]
      config.mapping_test_partitions = ["val2017"]
    elif config.dataset == "Potsdam":
      config.train_partitions = ["unlabelled_train", "labelled_train"]
      config.mapping_assignment_partitions = ["labelled_train"]
      config.mapping_test_partitions = ["labelled_test"]
    else:
      raise NotImplementedError

  elif config.mode == "IID":
    if "Coco10k" in config.dataset:
      config.train_partitions = ["all"]
      config.mapping_assignment_partitions = ["all"]
      config.mapping_test_partitions = ["all"]
    elif "Coco164k" in config.dataset:
      config.train_partitions = ["train2017", "val2017"]
      config.mapping_assignment_partitions = ["train2017", "val2017"]
      config.mapping_test_partitions = ["train2017", "val2017"]
    elif config.dataset == "Potsdam":
      config.train_partitions = ["unlabelled_train", "labelled_train",
                                 "labelled_test"]
      config.mapping_assignment_partitions = ["labelled_train", "labelled_test"]
      config.mapping_test_partitions = ["labelled_train", "labelled_test"]
    else:
      raise NotImplementedError

  if "Coco" in config.dataset:
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_Coco_dataloaders(config)
  elif config.dataset == "Potsdam":
    dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
      make_Potsdam_dataloaders(config)
  else:
    raise NotImplementedError

  return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def make_Coco_dataloaders(config):
  dataloaders = _create_dataloaders(config, cocostuff.__dict__[config.dataset])

  mapping_assignment_dataloader = \
    _create_mapping_loader(config, cocostuff.__dict__[config.dataset],
                           partitions=config.mapping_assignment_partitions)

  mapping_test_dataloader = \
    _create_mapping_loader(config, cocostuff.__dict__[config.dataset],
                           partitions=config.mapping_test_partitions)

  return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def make_Potsdam_dataloaders(config):
  dataloaders = _create_dataloaders(config, potsdam.__dict__[config.dataset])

  mapping_assignment_dataloader = \
    _create_mapping_loader(config, potsdam.__dict__[config.dataset],
                           partitions=config.mapping_assignment_partitions)

  mapping_test_dataloader = \
    _create_mapping_loader(config, potsdam.__dict__[config.dataset],
                           partitions=config.mapping_test_partitions)

  return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def _create_dataloaders(config, dataset_class):
  # unlike in clustering, each dataloader here returns pairs of images - we
  # need the matrix relation between them
  dataloaders = []
  do_shuffle = (config.num_dataloaders == 1)
  for d_i in range(config.num_dataloaders):
    print("Creating paired dataloader %d out of %d time %s" %
          (d_i, config.num_dataloaders, datetime.now()))
    sys.stdout.flush()

    train_imgs_list = []
    for train_partition in config.train_partitions:
      train_imgs_curr = dataset_class(
        **{"config": config,
           "split": train_partition,
           "purpose": "train"}  # return training tuples, not including labels
      )
      if config.use_doersch_datasets:
        train_imgs_curr = DoerschDataset(config, train_imgs_curr)

      train_imgs_list.append(train_imgs_curr)

    train_imgs = ConcatDataset(train_imgs_list)

    train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=do_shuffle,
                                                   num_workers=0,
                                                   drop_last=False)

    if d_i > 0:
      assert (len(train_dataloader) == len(dataloaders[d_i - 1]))

    dataloaders.append(train_dataloader)

  num_train_batches = len(dataloaders[0])
  print("Length of paired datasets vector %d" % len(dataloaders))
  print("Number of batches per epoch: %d" % num_train_batches)
  sys.stdout.flush()

  return dataloaders


def _create_mapping_loader(config, dataset_class, partitions):
  imgs_list = []
  for partition in partitions:
    imgs_curr = dataset_class(
      **{"config": config,
         "split": partition,
         "purpose": "test"}  # return testing tuples, image and label
    )
    if config.use_doersch_datasets:
      imgs_curr = DoerschDataset(config, imgs_curr)
    imgs_list.append(imgs_curr)

  imgs = ConcatDataset(imgs_list)
  dataloader = torch.utils.data.DataLoader(imgs,
                                           batch_size=config.batch_sz,
                                           # full batch
                                           shuffle=False,
                                           # no point since not trained on
                                           num_workers=0,
                                           drop_last=False)
  return dataloader
