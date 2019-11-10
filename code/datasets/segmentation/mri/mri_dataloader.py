import sys
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

import mri_dataset


def segmentation_create_dataloaders(config):

    config.train_partitions = ["all"]
    config.mapping_assignment_partitions = ["all"]
    config.mapping_test_partitions = ["all"]
    
    dataloaders = _create_dataloaders(config, mri_dataset.__dict__[config.dataset])
    
    mapping_assignment_dataloader = \
    _create_mapping_loader(config, mri_dataset.__dict__[config.dataset],
                           partitions=config.mapping_assignment_partitions)
                           
    mapping_test_dataloader = \
    _create_mapping_loader(config, mri_dataset.__dict__[config.dataset],
                           partitions=config.mapping_test_partitions)
        
    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader

def _create_dataloaders(config, dataset_class):
    # unlike in clustering, each dataloader here returns pairs of images - we
    # need the matrix relation between them
    dataloaders = []
    do_shuffle = (config.num_dataloaders == 1)
    for d_i in xrange(config.num_dataloaders):
        print("Creating paired dataloader %d out of %d time %s" % \
            (d_i, config.num_dataloaders, datetime.now()))
        sys.stdout.flush()

        train_imgs_list = []
        for train_partition in config.train_partitions:
            train_imgs_curr = dataset_class(
            **{"config": config,
            "split": train_partition,
            "purpose": "train"}  # return training tuples, not including labels
            )   
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