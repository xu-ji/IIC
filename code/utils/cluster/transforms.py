from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from torch.autograd import Variable


def custom_greyscale_to_tensor(include_rgb):
  def _inner(img):
    grey_img_tensor = tf.to_tensor(tf.to_grayscale(img, num_output_channels=1))
    result = grey_img_tensor  # 1, 96, 96 in [0, 1]
    assert (result.size(0) == 1)

    if include_rgb:  # greyscale last
      img_tensor = tf.to_tensor(img)
      result = torch.cat([img_tensor, grey_img_tensor], dim=0)
      assert (result.size(0) == 4)

    return result

  return _inner


def custom_cutout(min_box=None, max_box=None):
  def _inner(img):
    w, h = img.size

    # find left, upper, right, lower
    box_sz = np.random.randint(min_box, max_box + 1)
    half_box_sz = int(np.floor(box_sz / 2.))
    x_c = np.random.randint(half_box_sz, w - half_box_sz)
    y_c = np.random.randint(half_box_sz, h - half_box_sz)
    box = (
      x_c - half_box_sz, y_c - half_box_sz, x_c + half_box_sz,
      y_c + half_box_sz)

    img.paste(0, box=box)
    return img

  return _inner


def sobel_process(imgs, include_rgb, using_IR=False):
  bn, c, h, w = imgs.size()

  if not using_IR:
    if not include_rgb:
      assert (c == 1)
      grey_imgs = imgs
    else:
      assert (c == 4)
      grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
      rgb_imgs = imgs[:, :3, :, :]
  else:
    if not include_rgb:
      assert (c == 2)
      grey_imgs = imgs[:, 0, :, :].unsqueeze(1)  # underneath IR
      ir_imgs = imgs[:, 1, :, :].unsqueeze(1)
    else:
      assert (c == 5)
      rgb_imgs = imgs[:, :3, :, :]
      grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
      ir_imgs = imgs[:, 4, :, :].unsqueeze(1)

  sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  conv1.weight = nn.Parameter(
    torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
  dx = conv1(Variable(grey_imgs)).data

  sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  conv2.weight = nn.Parameter(
    torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
  dy = conv2(Variable(grey_imgs)).data

  sobel_imgs = torch.cat([dx, dy], dim=1)
  assert (sobel_imgs.shape == (bn, 2, h, w))

  if not using_IR:
    if include_rgb:
      sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)
      assert (sobel_imgs.shape == (bn, 5, h, w))
  else:
    if include_rgb:
      # stick both rgb and ir back on in right order (sobel sandwiched inside)
      sobel_imgs = torch.cat([rgb_imgs, sobel_imgs, ir_imgs], dim=1)
    else:
      # stick ir back on in right order (on top of sobel)
      sobel_imgs = torch.cat([sobel_imgs, ir_imgs], dim=1)

  return sobel_imgs


def per_img_demean(img):
  assert (len(img.size()) == 3 and img.size(0) == 3)  # 1 RGB image, tensor
  mean = img.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / \
         (img.size(1) * img.size(2))

  return img - mean  # expands


def sobel_make_transforms(config, random_affine=False,
                          cutout=False,
                          cutout_p=None,
                          cutout_max_box=None,
                          affine_p=None):
  tf1_list = []
  tf2_list = []
  tf3_list = []
  if config.crop_orig:
    tf1_list += [
      torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                        config.rand_crop_sz]))),
      torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                    config.input_sz]))),
    ]
    tf3_list += [
      torchvision.transforms.CenterCrop(tuple(np.array([config.rand_crop_sz,
                                                        config.rand_crop_sz]))),
      torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                    config.input_sz]))),
    ]

  print(
    "(_sobel_multioutput_make_transforms) config.include_rgb: %s" %
    config.include_rgb)
  tf1_list.append(custom_greyscale_to_tensor(config.include_rgb))
  tf3_list.append(custom_greyscale_to_tensor(config.include_rgb))

  if config.fluid_warp:
    # 50-50 do rotation or not
    print("adding rotation option for imgs_tf: %d" % config.rot_val)
    tf2_list += [torchvision.transforms.RandomApply(
      [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

    imgs_tf_crops = []
    for crop_sz in config.rand_crop_szs_tf:
      print("adding crop size option for imgs_tf: %d" % crop_sz)
      imgs_tf_crops.append(torchvision.transforms.RandomCrop(crop_sz))
    tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]
  else:
    # default
    tf2_list += [
      torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                        config.rand_crop_sz])))]

  if random_affine:
    print("adding affine with p %f" % affine_p)
    tf2_list.append(torchvision.transforms.RandomApply(
      [torchvision.transforms.RandomAffine(18,
                                           scale=(0.9, 1.1),
                                           translate=(0.1, 0.1),
                                           shear=10,
                                           resample=Image.BILINEAR,
                                           fillcolor=0)], p=affine_p)
    )

  assert (not (cutout and config.cutout))
  if cutout or config.cutout:
    assert (not config.fluid_warp)
    if config.cutout:
      cutout_p = config.cutout_p
      cutout_max_box = config.cutout_max_box

    print("adding cutout with p %f max box %f" % (cutout_p,
                                                  cutout_max_box))
    # https://github.com/uoguelph-mlrg/Cutout/blob/master/images
    # /cutout_on_cifar10.jpg
    tf2_list.append(
      torchvision.transforms.RandomApply(
        [custom_cutout(min_box=int(config.rand_crop_sz * 0.2),
                       max_box=int(config.rand_crop_sz *
                                   cutout_max_box))],
        p=cutout_p)
    )
  else:
    print("not using cutout")

  tf2_list += [
    torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                  config.input_sz]))),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.125)
  ]

  tf2_list.append(custom_greyscale_to_tensor(config.include_rgb))

  if config.demean:
    print("demeaning data")
    tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
    tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
    tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
  else:
    print("not demeaning data")

  if config.per_img_demean:
    print("per image demeaning data")
    tf1_list.append(per_img_demean)
    tf2_list.append(per_img_demean)
    tf3_list.append(per_img_demean)
  else:
    print("not per image demeaning data")

  tf1 = torchvision.transforms.Compose(tf1_list)
  tf2 = torchvision.transforms.Compose(tf2_list)
  tf3 = torchvision.transforms.Compose(tf3_list)

  return tf1, tf2, tf3


def greyscale_make_transforms(config):
  tf1_list = []
  tf3_list = []
  tf2_list = []

  # tf1 and 3 transforms
  if config.crop_orig:
    # tf1 crop
    if config.tf1_crop == "random":
      print("selected random crop for tf1")
      tf1_crop_fn = torchvision.transforms.RandomCrop(config.tf1_crop_sz)
    elif config.tf1_crop == "centre_half":
      print("selected centre_half crop for tf1")
      tf1_crop_fn = torchvision.transforms.RandomChoice([
        torchvision.transforms.RandomCrop(config.tf1_crop_sz),
        torchvision.transforms.CenterCrop(config.tf1_crop_sz)
      ])
    elif config.tf1_crop == "centre":
      print("selected centre crop for tf1")
      tf1_crop_fn = torchvision.transforms.CenterCrop(config.tf1_crop_sz)
    else:
      assert (False)
    tf1_list += [tf1_crop_fn]

    if config.tf3_crop_diff:
      print("tf3 crop size is different to tf1")
      tf3_list += [torchvision.transforms.CenterCrop(config.tf3_crop_sz)]
    else:
      print("tf3 crop size is same as tf1")
      tf3_list += [torchvision.transforms.CenterCrop(config.tf1_crop_sz)]

  tf1_list += [torchvision.transforms.Resize(config.input_sz),
               torchvision.transforms.ToTensor()]
  tf3_list += [torchvision.transforms.Resize(config.input_sz),
               torchvision.transforms.ToTensor()]

  # tf2 transforms
  if config.rot_val > 0:
    # 50-50 do rotation or not
    print("adding rotation option for imgs_tf: %d" % config.rot_val)
    if config.always_rot:
      print("always_rot")
      tf2_list += [torchvision.transforms.RandomRotation(config.rot_val)]
    else:
      print("not always_rot")
      tf2_list += [torchvision.transforms.RandomApply(
        [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

  if config.crop_other:
    imgs_tf_crops = []
    for tf2_crop_sz in config.tf2_crop_szs:
      if config.tf2_crop == "random":
        print("selected random crop for tf2")
        tf2_crop_fn = torchvision.transforms.RandomCrop(tf2_crop_sz)
      elif config.tf2_crop == "centre_half":
        print("selected centre_half crop for tf2")
        tf2_crop_fn = torchvision.transforms.RandomChoice([
          torchvision.transforms.RandomCrop(tf2_crop_sz),
          torchvision.transforms.CenterCrop(tf2_crop_sz)
        ])
      elif config.tf2_crop == "centre":
        print("selected centre crop for tf2")
        tf2_crop_fn = torchvision.transforms.CenterCrop(tf2_crop_sz)
      else:
        assert (False)

      print("adding crop size option for imgs_tf: %d" % tf2_crop_sz)
      imgs_tf_crops.append(tf2_crop_fn)

    tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]

  tf2_list += [torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                             config.input_sz])))]

  if not config.no_flip:
    print("adding flip")
    tf2_list += [torchvision.transforms.RandomHorizontalFlip()]
  else:
    print("not adding flip")

  if not config.no_jitter:
    print("adding jitter")
    tf2_list += [
      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.4, hue=0.125)]
  else:
    print("not adding jitter")

  tf2_list += [torchvision.transforms.ToTensor()]

  # admin transforms
  if config.demean:
    print("demeaning data")
    tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
    tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
    tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                     std=config.data_std))
  else:
    print("not demeaning data")

  if config.per_img_demean:
    print("per image demeaning data")
    tf1_list.append(per_img_demean)
    tf2_list.append(per_img_demean)
    tf3_list.append(per_img_demean)
  else:
    print("not per image demeaning data")

  tf1 = torchvision.transforms.Compose(tf1_list)
  tf2 = torchvision.transforms.Compose(tf2_list)
  tf3 = torchvision.transforms.Compose(tf3_list)

  return tf1, tf2, tf3
