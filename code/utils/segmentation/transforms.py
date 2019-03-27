import cv2
import numpy as np
import torch
import torch.nn.functional as F


def custom_greyscale_numpy(img, include_rgb=True):
  # Takes and returns a channel-last numpy array, uint8

  # use channels last for cvtColor
  h, w, c = img.shape
  grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w,
                                                           1)  # new memory

  if include_rgb:
    img = np.concatenate([img, grey_img], axis=2)
  else:
    img = grey_img

  return img


def pad_if_too_small(data, sz):
  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape

  if not (h >= sz and w >= sz):
    # img is smaller than sz
    # we are missing by at least 1 pixel in at least 1 edge
    new_h, new_w = max(h, sz), max(w, sz)
    new_data = np.zeros([new_h, new_w, c], dtype=data.dtype)

    # will get correct centre, 5 -> 2
    centre_h, centre_w = int(new_h / 2.), int(new_w / 2.)
    h_start, w_start = centre_h - int(h / 2.), centre_w - int(w / 2.)

    new_data[h_start:(h_start + h), w_start:(w_start + w), :] = data
  else:
    new_data = data
    new_h, new_w = h, w

  if reshape:
    new_data = new_data.reshape((new_h, new_w))

  return new_data


def pad_and_or_crop(orig_data, sz, mode=None, coords=None):
  data = pad_if_too_small(orig_data, sz)

  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape
  if mode == "centre":
    h_c = int(h / 2.)
    w_c = int(w / 2.)
  elif mode == "fixed":
    assert (coords is not None)
    h_c, w_c = coords
  elif mode == "random":
    h_c_min = int(sz / 2.)
    w_c_min = int(sz / 2.)

    if sz % 2 == 1:
      h_c_max = h - 1 - int(sz / 2.)
      w_c_max = w - 1 - int(sz / 2.)
    else:
      h_c_max = h - int(sz / 2.)
      w_c_max = w - int(sz / 2.)

    h_c = np.random.randint(low=h_c_min, high=(h_c_max + 1))
    w_c = np.random.randint(low=w_c_min, high=(w_c_max + 1))

  h_start = h_c - int(sz / 2.)
  w_start = w_c - int(sz / 2.)
  data = data[h_start:(h_start + sz), w_start:(w_start + sz), :]

  if reshape:
    data = data.reshape((sz, sz))

  return data, (h_c, w_c)


def random_affine(img, min_rot=None, max_rot=None, min_shear=None,
                  max_shear=None, min_scale=None, max_scale=None):
  # Takes and returns torch cuda tensors with channels 1st (1 img)
  # rot and shear params are in degrees
  # tf matrices need to be float32, returned as tensors
  # we don't do translations

  # https://github.com/pytorch/pytorch/issues/12362
  # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
  # -hard-coded-vs-numpy-linalg-inv

  # https://github.com/pytorch/vision/blob/master/torchvision/transforms
  # /functional.py#L623
  # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
  #                        [ sin(a)*scale    cos(a + shear)*scale     0]
  #                        [     0                  0          1]
  # used by opencv functional _get_affine_matrix and
  # skimage.transform.AffineTransform

  assert (len(img.shape) == 3)
  a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
  shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
  scale = np.random.rand() * (max_scale - min_scale) + min_scale

  affine1_to_2 = np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                           [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                           [0., 0., 1.]], dtype=np.float32)  # 3x3

  affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

  affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
  affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cuda(), \
                               torch.from_numpy(affine2_to_1).cuda()

  img = perform_affine_tf(img.unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))
  img = img.squeeze(dim=0)

  return img, affine1_to_2, affine2_to_1


def perform_affine_tf(data, tf_matrices):
  # expects 4D tensor, we preserve gradients if there are any

  n_i, k, h, w = data.shape
  n_i2, r, c = tf_matrices.shape
  assert (n_i == n_i2)
  assert (r == 2 and c == 3)

  grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size
  data_tf = F.grid_sample(data, grid,
                          padding_mode="zeros")  # this can ONLY do bilinear

  return data_tf


def random_translation_multiple(data, half_side_min, half_side_max):
  n, c, h, w = data.shape

  # pad last 2, i.e. spatial, dimensions, equally in all directions
  data = F.pad(data,
               (half_side_max, half_side_max, half_side_max, half_side_max),
               "constant", 0)
  assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

  # random x, y displacement
  t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
  polarities = np.random.choice([-1, 1], size=(2,), replace=True)
  t *= polarities

  # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
  t += half_side_max

  data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
  assert (data.shape[2:] == (h, w))

  return data


def random_translation(img, half_side_min, half_side_max):
  # expects 3d (cuda) tensor with channels first
  c, h, w = img.shape

  # pad last 2, i.e. spatial, dimensions, equally in all directions
  img = F.pad(img, (half_side_max, half_side_max, half_side_max, half_side_max),
              "constant", 0)
  assert (img.shape[1:] == (2 * half_side_max + h, 2 * half_side_max + w))

  # random x, y displacement
  t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
  polarities = np.random.choice([-1, 1], size=(2,), replace=True)
  t *= polarities

  # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
  t += half_side_max

  img = img[:, t[1]:(t[1] + h), t[0]:(t[0] + w)]
  assert (img.shape[1:] == (h, w))

  return img
