# based on
# https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/datasets
# /cocostuff.py

from __future__ import print_function

import os.path as osp
import pickle
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data

from util import cocostuff_fine_to_coarse
from .util.cocostuff_fine_to_coarse import generate_fine_to_coarse
from ...utils.segmentation.render import render
from ...utils.segmentation.transforms import \
  pad_and_or_crop, random_affine, custom_greyscale_numpy

__all__ = ["Coco10kFull", "Coco10kFew", "Coco164kFull", "Coco164kFew",
           "Coco164kCuratedFew", "Coco164kCuratedFull"]

RENDER_DATA = False


class _Coco(data.Dataset):
  """Base class
  This contains fields and methods common to all COCO datasets:
  (COCO-fine) (182)
  COCO-coarse (27)
  COCO-few (6)
  (COCOStuff-fine) (91)
  COCOStuff-coarse (15)
  COCOStuff-few (3)
  
  For both 10k and 164k (though latter is unimplemented)
  """

  def __init__(self, config=None, split=None, purpose=None, preload=False):
    super(_Coco, self).__init__()

    self.split = split
    self.purpose = purpose

    self.root = config.dataset_root

    self.single_mode = hasattr(config, "single_mode") and config.single_mode

    # always used (labels fields used to make relevancy mask for train)
    self.gt_k = config.gt_k
    self.pre_scale_all = config.pre_scale_all
    self.pre_scale_factor = config.pre_scale_factor
    self.input_sz = config.input_sz

    self.include_rgb = config.include_rgb
    self.no_sobel = config.no_sobel

    assert ((not hasattr(config, "mask_input")) or (not config.mask_input))
    self.mask_input = False

    # only used if purpose is train
    if purpose == "train":
      self.use_random_scale = config.use_random_scale
      if self.use_random_scale:
        self.scale_max = config.scale_max
        self.scale_min = config.scale_min

      self.jitter_tf = tvt.ColorJitter(brightness=config.jitter_brightness,
                                       contrast=config.jitter_contrast,
                                       saturation=config.jitter_saturation,
                                       hue=config.jitter_hue)

      self.flip_p = config.flip_p  # 0.5

      self.use_random_affine = config.use_random_affine
      if self.use_random_affine:
        self.aff_min_rot = config.aff_min_rot
        self.aff_max_rot = config.aff_max_rot
        self.aff_min_shear = config.aff_min_shear
        self.aff_max_shear = config.aff_max_shear
        self.aff_min_scale = config.aff_min_scale
        self.aff_max_scale = config.aff_max_scale

    assert (not preload)

    self.files = []
    self.images = []
    self.labels = []

    if not osp.exists(config.fine_to_coarse_dict):
      generate_fine_to_coarse(config.fine_to_coarse_dict)

    with open(config.fine_to_coarse_dict, "rb") as dict_f:
      d = pickle.load(dict_f)
      self._fine_to_coarse_dict = d["fine_index_to_coarse_index"]

    cv2.setNumThreads(0)

  def _prepare_train(self, index, img, label):
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    # shrink original images, for memory purposes, otherwise no point
    if self.pre_scale_all:
      assert (self.pre_scale_factor < 1.)
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    # basic augmentation transforms for both img1 and img2
    if self.use_random_scale:
      # bilinear interp requires float img
      scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                     self.scale_min
      img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    # random crop to input sz
    img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
    label, _ = pad_and_or_crop(label, self.input_sz, mode="fixed",
                               coords=coords)

    _, mask_img1 = self._filter_label(label)
    # uint8 tensor as masks should be binary, also for consistency with
    # prepare_train, but converted to float32 in main loop because is used
    # multiplicatively in loss
    mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8)).cuda()

    # make img2 different from img1 (img)

    # tf_mat can be:
    # *A, from img2 to img1 (will be applied to img2's heatmap)-> img1 space
    #   input img1 tf: *tf.functional or pil.image
    #   input mask tf: *none
    #   output heatmap: *tf.functional (parallel), inverse of what is used
    #     for inputs, create inverse of this tf in [-1, 1] format

    # B, from img1 to img2 (will be applied to img1's heatmap)-> img2 space
    #   input img1 tf: pil.image
    #   input mask tf: pil.image (discrete)
    #   output heatmap: tf.functional, create copy of this tf in [-1,1] format

    # tf.function tf_mat: translation is opposite to what we'd expect (+ve 1
    # is shift half towards left)
    # but rotation is correct (-sin in top right = counter clockwise)

    # flip is [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # img2 = flip(affine1_to_2(img1))
    # => img1_space = affine1_to_2^-1(flip^-1(img2_space))
    #               = affine2_to_1(flip^-1(img2_space))
    # so tf_mat_img2_to_1 = affine2_to_1 * flip^-1 (order matters as not diag)
    # flip^-1 = flip

    # no need to tf label, as we're doing option A, mask needed in img1 space

    # converting to PIL does not change underlying np datatype it seems
    img1 = Image.fromarray(img.astype(np.uint8))

    # (img2) do jitter, no tf_mat change
    img2 = self.jitter_tf(img1)  # not in place, new memory
    img1 = np.array(img1)
    img2 = np.array(img2)

    # channels still last
    if not self.no_sobel:
      img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)
      img2 = custom_greyscale_numpy(img2, include_rgb=self.include_rgb)

    img1 = img1.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    # convert both to channel-first tensor format
    # make them all cuda tensors now, except label, for optimality
    img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).cuda()

    # mask if required
    if self.mask_input:
      masked = 1 - mask_img1
      img1[:, masked] = 0
      img2[:, masked] = 0

    # (img2) do affine if nec, tf_mat changes
    if self.use_random_affine:
      affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                       "min_shear": self.aff_min_shear,
                       "max_shear": self.aff_max_shear,
                       "min_scale": self.aff_min_scale,
                       "max_scale": self.aff_max_scale}
      img2, affine1_to_2, affine2_to_1 = random_affine(img2,
                                                       **affine_kwargs)  #
      # tensors
    else:
      affine2_to_1 = torch.zeros([2, 3]).to(torch.float32).cuda()  # identity
      affine2_to_1[0, 0] = 1
      affine2_to_1[1, 1] = 1

    # (img2) do random flip, tf_mat changes
    if np.random.rand() > self.flip_p:
      img2 = torch.flip(img2, dims=[2])  # horizontal, along width

      # applied affine, then flip, new = flip * affine * coord
      # (flip * affine)^-1 is just flip^-1 * affine^-1.
      # No order swap, unlike functions...
      # hence top row is negated
      affine2_to_1[0, :] *= -1.

    if RENDER_DATA:
      render(img1, mode="image", name=("train_data_img1_%d" % index))
      render(img2, mode="image", name=("train_data_img2_%d" % index))
      render(affine2_to_1, mode="matrix",
             name=("train_data_affine2to1_%d" % index))
      render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))

    return img1, img2, affine2_to_1, mask_img1

  def _prepare_train_single(self, index, img, label):
    # Returns one pair only, i.e. without transformed second image.
    # Used for standard CNN training (baselines).
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    # shrink original images, for memory purposes, otherwise no point
    if self.pre_scale_all:
      assert (self.pre_scale_factor < 1.)
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    if self.use_random_scale:
      # bilinear interp requires float img
      scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                     self.scale_min
      img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    # random crop to input sz
    img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
    label, _ = pad_and_or_crop(label, self.input_sz, mode="fixed",
                               coords=coords)

    _, mask_img1 = self._filter_label(label)
    # uint8 tensor as masks should be binary, also for consistency with
    # prepare_train, but converted to float32 in main loop because is used
    # multiplicatively in loss
    mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8)).cuda()

    # converting to PIL does not change underlying np datatype it seems
    img1 = Image.fromarray(img.astype(np.uint8))

    img1 = self.jitter_tf(img1)  # not in place, new memory
    img1 = np.array(img1)

    # channels still last
    if not self.no_sobel:
      img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)

    img1 = img1.astype(np.float32) / 255.

    # convert both to channel-first tensor format
    # make them all cuda tensors now, except label, for optimality
    img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()

    # mask if required
    if self.mask_input:
      masked = 1 - mask_img1
      img1[:, masked] = 0

    if self.use_random_affine:
      affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                       "min_shear": self.aff_min_shear,
                       "max_shear": self.aff_max_shear,
                       "min_scale": self.aff_min_scale,
                       "max_scale": self.aff_max_scale}
      img1, _, _ = random_affine(img1, **affine_kwargs)  # tensors

    if np.random.rand() > self.flip_p:
      img1 = torch.flip(img1, dims=[2])  # horizontal, along width

    if RENDER_DATA:
      render(img1, mode="image", name=("train_data_img1_%d" % index))
      render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))

    return img1, mask_img1

  def _prepare_test(self, index, img, label):
    # This returns cpu tensors.
    #   Image: 3D with channels last, float32, in range [0, 1] (normally done
    #     by ToTensor).
    #   Label map: 2D, flat int64, [0 ... sef.gt_k - 1]
    # label is passed in canonical [0 ... 181] indexing

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    # shrink original images, for memory purposes, otherwise no point
    if self.pre_scale_all:
      assert (self.pre_scale_factor < 1.)
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    # center crop to input sz
    img, _ = pad_and_or_crop(img, self.input_sz, mode="centre")
    label, _ = pad_and_or_crop(label, self.input_sz, mode="centre")

    # finish
    if not self.no_sobel:
      img = custom_greyscale_numpy(img, include_rgb=self.include_rgb)

    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)

    if RENDER_DATA:
      render(label, mode="label", name=("test_data_label_pre_%d" % index))

    # convert to coarse if required, reindex to [0, gt_k -1], and get mask
    label, mask = self._filter_label(label)

    # mask if required
    if self.mask_input:
      masked = 1 - mask
      img[:, masked] = 0

    if RENDER_DATA:
      render(img, mode="image", name=("test_data_img_%d" % index))
      render(label, mode="label", name=("test_data_label_post_%d" % index))
      render(mask, mode="mask", name=("test_data_mask_%d" % index))

    # dataloader must return tensors (conversion forced in their code anyway)
    return img, torch.from_numpy(label), torch.from_numpy(mask.astype(np.uint8))

  def __getitem__(self, index):
    image_id = self.files[index]
    image, label = self._load_data(image_id)

    if self.purpose == "train":
      if not self.single_mode:
        return self._prepare_train(index, image, label)
      else:
        return self._prepare_train_single(index, image, label)
    else:
      assert (self.purpose == "test")
      return self._prepare_test(index, image, label)

  def __len__(self):
    return len(self.files)

  def _check_gt_k(self):
    raise NotImplementedError()

  def _filter_label(self):
    raise NotImplementedError()

  def _set_files(self):
    raise NotImplementedError()

  def _load_data(self, image_id):
    raise NotImplementedError()


# ------------------------------------------------------------------------------
# Handles which images are eligible

class _Coco10k(_Coco):
  """Base class
  This contains fields and methods common to all COCO 10k datasets:
  (COCO-fine) (182)
  COCO-coarse (27)
  COCO-few (6)
  (COCOStuff-fine) (91)
  COCOStuff-coarse (15)
  COCOStuff-few (3)
  """

  def __init__(self, **kwargs):
    super(_Coco10k, self).__init__(**kwargs)
    self._set_files()

  def _set_files(self):
    if self.split in ["train", "test", "all"]:
      # deterministic order - important - so >1 dataloader actually meaningful
      file_list = osp.join(self.root, "imageLists", self.split + ".txt")
      file_list = tuple(open(file_list, "r"))
      file_list = [id_.rstrip() for id_ in file_list]
      self.files = file_list
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    image_path = osp.join(self.root, "images", image_id + ".jpg")
    label_path = osp.join(self.root, "annotations", image_id + ".mat")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)

    label = sio.loadmat(label_path)["S"].astype(np.int32)  # [0, 182]
    label -= 1  # unlabeled (0 -> -1)
    # label should now be [-1, 0 ... 181], 91 each

    return image, label


class _Coco164k(_Coco):
  """Base class
  This contains fields and methods common to all COCO 164k datasets
  This is too huge to train in reasonable time
  """

  def __init__(self, **kwargs):
    super(_Coco164k, self).__init__(**kwargs)
    self._set_files()

  def _set_files(self):
    # Create data list by parsing the "images" folder
    if self.split in ["train2017", "val2017"]:
      file_list = sorted(
        glob(osp.join(self.root, "images", self.split, "*.jpg")))
      file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
      self.files = file_list
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    # Set paths
    image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
    label_path = osp.join(self.root, "annotations", self.split,
                          image_id + ".png")
    # Load an image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    label[label == 255] = -1  # to be consistent with 10k

    return image, label


class _Coco164kCuratedFew(_Coco):
  """Base class
  This contains fields and methods common to all COCO 164k curated few datasets:
  
  (curated) Coco164kFew_Stuff
  (curated) Coco164kFew_Stuff_People
  (curated) Coco164kFew_Stuff_Animals
  (curated) Coco164kFew_Stuff_People_Animals 

  """

  def __init__(self, **kwargs):
    super(_Coco164kCuratedFew, self).__init__(**kwargs)

    # work out name
    config = kwargs["config"]
    assert (config.use_coarse_labels)  # we only deal with coarse labels
    self.include_things_labels = config.include_things_labels  # people
    self.incl_animal_things = config.incl_animal_things  # animals

    version = config.coco_164k_curated_version

    name = "Coco164kFew_Stuff"
    if self.include_things_labels and self.incl_animal_things:
      name += "_People_Animals"
    elif self.include_things_labels:
      name += "_People"
    elif self.incl_animal_things:
      name += "_Animals"

    self.name = (name + "_%d" % version)

    print("Specific type of _Coco164kCuratedFew dataset: %s" % self.name)

    self._set_files()

  def _set_files(self):
    # Create data list by parsing the "images" folder
    if self.split in ["train2017", "val2017"]:
      file_list = osp.join(self.root, "curated", self.split, self.name + ".txt")
      file_list = tuple(open(file_list, "r"))
      file_list = [id_.rstrip() for id_ in file_list]

      self.files = file_list
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    # same as _Coco164k
    # Set paths
    image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
    label_path = osp.join(self.root, "annotations", self.split,
                          image_id + ".png")
    # Load an image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    label[label == 255] = -1  # to be consistent with 10k

    return image, label


class _Coco164kCuratedFull(_Coco):
  """Base class
  This contains fields and methods common to all COCO 164k curated full 
  datasets:
  
  (curated) Coco164kFull_Stuff_Coarse

  """

  def __init__(self, **kwargs):
    super(_Coco164kCuratedFull, self).__init__(**kwargs)

    # work out name
    config = kwargs["config"]
    assert (config.use_coarse_labels)  # we only deal with coarse labels

    assert (not config.include_things_labels)
    assert (not config.incl_animal_things)

    version = config.coco_164k_curated_version

    self.name = "Coco164kFull_Stuff_Coarse_%d" % version

    print("Specific type of _Coco164kCuratedFull dataset: %s" % self.name)

    self._set_files()

  def _set_files(self):
    # Create data list by parsing the "images" folder
    if self.split in ["train2017", "val2017"]:
      file_list = osp.join(self.root, "curated", self.split,
                           self.name + ".txt")
      file_list = tuple(open(file_list, "r"))
      file_list = [id_.rstrip() for id_ in file_list]

      self.files = file_list
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    # same as _Coco164k
    # Set paths
    image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
    label_path = osp.join(self.root, "annotations", self.split,
                          image_id + ".png")
    # Load an image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    label[label == 255] = -1  # to be consistent with 10k

    return image, label


# ------------------------------------------------------------------------------
# Handles Full vs Few

class _CocoFull(_Coco):
  """ 
  This contains methods for the following datasets 
  (Full = original labels, coarse or fine)
  
  (COCO-fine) (182)
  COCO-coarse (27)
  (COCOStuff-fine) (91)
  COCOStuff-coarse (15)
  """

  def __init__(self, **kwargs):
    super(_CocoFull, self).__init__(**kwargs)

    config = kwargs["config"]

    # if coarse, index corresponds to order in cocostuff_fine_to_coarse.py
    self.use_coarse_labels = config.use_coarse_labels
    self.include_things_labels = config.include_things_labels

    self._check_gt_k()

  def _fine_to_coarse(self, label_map):
    # label_map is in fine indexing
    # can't be in place!

    new_label_map = np.zeros(label_map.shape, dtype=label_map.dtype)

    # -1 stays -1
    for c in xrange(182):
      new_label_map[label_map == c] = self._fine_to_coarse_dict[c]

    return new_label_map

  def _check_gt_k(self):
    if self.use_coarse_labels:
      if self.include_things_labels:
        assert (self.gt_k == 27)
      else:
        assert (self.gt_k == 15)
    else:
      if self.include_things_labels:
        assert (self.gt_k == 182)
      else:
        assert (self.gt_k == 91)

  def _filter_label(self, label):
    # expects np array in fine labels ([0, 181]) and returns np arrays
    # convert to coarse if required, and reindex to [0, gt_k -1], and get mask
    # do we care about what is in masked portion of label map - no
    # in eval, mask used to select, others ignored

    # things: 91 classes (0-90), 12 superclasses (0-11)
    # stuff: 91 classes (91-181), 15 superclasses (12-26)

    if self.use_coarse_labels:
      label = self._fine_to_coarse(label)
      if self.include_things_labels:
        first_allowed_index = 0
      else:
        first_allowed_index = 12  # first coarse stuff index
    else:
      if self.include_things_labels:
        first_allowed_index = 0
      else:
        first_allowed_index = 91  # first fine stuff index

    # always excludes unlabelled (<= -1)
    mask = (label >= first_allowed_index)
    assert (mask.dtype == np.bool)
    # put in [0, gt_k], gt_k can be 27, 15, 182, 91
    label -= first_allowed_index

    return label, mask


class _CocoFew(_Coco):
  """
  This contains methods for the following datasets 
  
  COCO-few (6)
  COCOStuff-few (3)
  """

  def __init__(self, **kwargs):
    super(_CocoFew, self).__init__(**kwargs)

    config = kwargs["config"]
    assert (config.use_coarse_labels)  # we only deal with coarse labels
    self.include_things_labels = config.include_things_labels
    self.incl_animal_things = config.incl_animal_things

    self._check_gt_k()

    # indexes correspond to order in these lists
    self.label_names = [
      "sky-stuff",
      "plant-stuff",
      "ground-stuff",
    ]

    # CHANGED. Can have animals and/or people.
    if self.include_things_labels:
      self.label_names += ["person-things"]

    if self.incl_animal_things:
      self.label_names += ["animal-things"]

    assert (len(self.label_names) == self.gt_k)

    # make dict that maps fine labels to our labels
    self._fine_to_few_dict = self._make_fine_to_few_dict()

  def _make_fine_to_few_dict(self):
    # only make indices
    self.label_orig_coarse_inds = []
    for label_name in self.label_names:
      orig_coarse_ind = cocostuff_fine_to_coarse._sorted_coarse_names.index(
        label_name)
      self.label_orig_coarse_inds.append(orig_coarse_ind)

    print("label_orig_coarse_inds for this dataset: ")
    print(self.label_orig_coarse_inds)

    # excludes -1 (fine - see usage in filter label - as with Coco10kFull)
    _fine_to_few_dict = {}
    for c in xrange(182):
      orig_coarse_ind = self._fine_to_coarse_dict[c]
      if orig_coarse_ind in self.label_orig_coarse_inds:
        new_few_ind = self.label_orig_coarse_inds.index(orig_coarse_ind)
        # print("assigning fine %d coarse %d to new ind %d" % (c,
        #                                                     orig_coarse_ind,
        #                                                     new_few_ind))
      else:
        new_few_ind = -1
      _fine_to_few_dict[c] = new_few_ind

    # print("fine to few dict:")
    # print(_fine_to_few_dict)
    return _fine_to_few_dict

  def _check_gt_k(self):
    # Can have animals and/or people.
    expected_gt_k = 3
    if self.include_things_labels:
      expected_gt_k += 1
    if self.incl_animal_things:
      expected_gt_k += 1

    assert (self.gt_k == expected_gt_k)

  def _filter_label(self, label):
    # expects np array in fine labels ([-1, 181]) and returns np arrays
    # use coarse labels, reindex to [-1, gt_k -1], and get mask
    # do we care about what is in masked portion of label map - no
    # in eval, mask used to select, others ignored

    # min = label.min()
    # max = label.max()
    # if min < -1:
    #  print("smaller than expected %d" % min)
    #  assert(False)
    # if max >= 182:
    #  print("bigger than expected %d" % max)
    #  assert(False)

    # can't be in place!

    # -1 stays -1
    new_label_map = np.zeros(label.shape, dtype=label.dtype)

    for c in xrange(182):
      new_label_map[label == c] = self._fine_to_few_dict[c]

    mask = (new_label_map >= 0)
    assert (mask.dtype == np.bool)

    return new_label_map, mask


# ------------------------------------------------------------------------------
# All 4 combinations of 10k-164k, Full-Few (Full includes coarse or fine)
class Coco10kFull(_Coco10k, _CocoFull):
  def __init__(self, **kwargs):
    super(Coco10kFull, self).__init__(**kwargs)


class Coco10kFew(_Coco10k, _CocoFew):
  def __init__(self, **kwargs):
    super(Coco10kFew, self).__init__(**kwargs)


class Coco164kFull(_Coco164k, _CocoFull):
  def __init__(self, **kwargs):
    super(Coco164kFull, self).__init__(**kwargs)


class Coco164kFew(_Coco164k, _CocoFew):
  def __init__(self, **kwargs):
    super(Coco164kFew, self).__init__(**kwargs)


# Only 2 top level class options for curated datasets
class Coco164kCuratedFew(_Coco164kCuratedFew, _CocoFew):
  def __init__(self, **kwargs):
    super(Coco164kCuratedFew, self).__init__(**kwargs)


class Coco164kCuratedFull(_Coco164kCuratedFull, _CocoFull):
  def __init__(self, **kwargs):
    super(Coco164kCuratedFull, self).__init__(**kwargs)
